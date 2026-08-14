"""Place-level frontend: purpose-trained VPR descriptors, one per FRAME.

Why this exists: held-out localisation was poor (tracker as), and the kNN
ceiling showed the bottleneck is the DESCRIPTOR — a mean of detection-crop
embeddings destroys scene layout and was never trained for place identity.
This module embeds whole frames with a visual-place-recognition network
trained for viewpoint robustness (EigenPlaces, ICCV 2023; CosPlace as the
fallback), which is exactly the measured failure mode (the "other pass"
aliasing of (aq)).

Granularity principle (the reason this ADDS a frontend rather than replacing
crops): representation granularity must match query granularity. Object
queries keep the crop frontend; the place/localisation factor gets this one.

Output piggybacks the crop-file schema — one row per frame, `frame_idx`
unique, `class_id = -1` — so every downstream tool (`heldout_eval --encoders
name=file`, `time_recall --encoder`, `crosstalk_scaling --content`,
`recall_cdf`) works unchanged: `frame_descriptors()` mean-pools by frame_idx,
and a one-row-per-frame file passes through as-is.

    python -m vsa_cognitive_mapping.vpr_frontend \
        --dataset vsa_cognitive_mapping/configs/classroom_seq.json
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vsa_cognitive_mapping.sequences import load_sequence  # noqa: E402

HUBS = {
    "eigenplaces": ("gmberton/eigenplaces", "EigenPlaces (ICCV 2023, viewpoint-robust)"),
    "cosplace": ("gmberton/cosplace", "CosPlace (CVPR 2022)"),
}
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def resolve_device(spec):
    if spec != "auto":
        return spec
    return "cuda" if torch.cuda.is_available() else "cpu"


def preprocess(img, short):
    """PIL RGB -> float tensor (3,H,W), ImageNet-normalised, shortest side =
    `short` (aspect preserved; the VPR nets pool adaptively, so the spatial
    size is a speed/quality knob, not a requirement)."""
    w, h = img.size
    s = short / min(w, h)
    img = img.convert("RGB").resize((int(round(w * s)), int(round(h * s))))
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x).permute(2, 0, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True,
                    help="sequence config JSON (classroom/school_run1/TUM alike)")
    ap.add_argument("--model", default="eigenplaces", choices=list(HUBS))
    ap.add_argument("--backbone", default="ResNet50")
    ap.add_argument("--dim", type=int, default=2048)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--resize", type=int, default=384,
                    help="shortest image side fed to the net; larger = slower, "
                         "slightly better. 384 is the CPU compromise")
    ap.add_argument("--limit", type=int, default=0, help="smoke runs")
    ap.add_argument("--save-every", type=int, default=500)
    args = ap.parse_args()

    seq = load_sequence(args.dataset)
    device = resolve_device(args.device)
    repo, desc = HUBS[args.model]
    print(f"{seq.name}: {len(seq)} frames; model {desc} "
          f"({args.backbone}, {args.dim}-d) on {device}, resize {args.resize}")

    # Newer torch.hub interactively prompts "do you trust this repository?"
    # — a subprocess (Colab cell) has no stdin and EOFs. trust_repo=True on
    # our call is not enough: EigenPlaces' hubconf itself hub-loads
    # gmberton/cosplace without the flag, so force it for every nested
    # hub load in this process.
    _hub_load = torch.hub.load
    torch.hub.load = lambda *a, **kw: _hub_load(*a, **{**kw, "trust_repo": True})
    model = torch.hub.load(repo, "get_trained_model",
                           backbone=args.backbone, fc_output_dim=args.dim)
    model.eval().to(device)

    n = len(seq) if not args.limit else min(args.limit, len(seq))
    fids = seq.frame_ids()[:n]
    ts = (seq.timestamps_s()[:n] * 1e9).astype(np.int64)
    out_path = os.path.join(seq.out_dir, f"crop_embeddings_{args.model}.pt")
    os.makedirs(seq.out_dir, exist_ok=True)

    E = torch.zeros(n, args.dim, dtype=torch.float32)
    done = 0

    def save(tag=""):
        torch.save({"embedding": E[:done].clone(),
                    "frame_idx": torch.tensor(fids[:done].astype(np.int64)),
                    "timestamp_ns": torch.tensor(ts[:done]),
                    "class_id": torch.full((done,), -1, dtype=torch.long),
                    "confidence": torch.ones(done),
                    "encoder": f"{args.model}:{args.backbone}:{args.dim}",
                    "resize": args.resize, "frame_level": True},
                   out_path + tag)

    with torch.no_grad():
        i = 0
        while i < n:
            j = min(i + args.batch, n)
            batch = torch.stack([preprocess(seq.image(k), args.resize)
                                 for k in range(i, j)]).to(device)
            E[i:j] = model(batch).float().cpu()
            done = j
            i = j
            if done % args.save_every < args.batch:
                save(".part")
                print(f"  {done}/{n}", flush=True)
    save()
    if os.path.exists(out_path + ".part"):
        os.remove(out_path + ".part")
    print(f"saved {tuple(E.shape)} -> {out_path}")

    # quick geometry read so every embed run reports its conditioning
    X = E[:done].numpy().astype(np.float64)
    U = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    m = min(done, 800)
    sel = np.linspace(0, done - 1, m).astype(int)
    S = U[sel] @ U[sel].T
    off = S[~np.eye(m, dtype=bool)]
    Xc = X - X.mean(0, keepdims=True)
    lam = np.linalg.svd(Xc, compute_uv=False) ** 2
    pr = float(lam.sum() ** 2 / (lam ** 2).sum()) if lam.sum() > 0 else 0.0
    print(f"geometry: mean off-diag cos {off.mean():+.3f} (|cos| {np.abs(off).mean():.3f}), "
          f"eff-rank {pr:.1f}/{X.shape[1]}")


if __name__ == "__main__":
    main()
