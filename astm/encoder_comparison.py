"""CNN vs ViT vs untrained: where does our anisotropy actually come from?

Two controls from the anisotropy literature, run on identical crops:

**Godey et al. (EACL 2024)** report that anisotropy tracks self-attention:
transformers and ViTs are anisotropic at every layer, while CNNs largely are
not -- ResNet-50 gets *more* isotropic with depth. Our crosstalk was measured
on YOLOv8n, which is a CNN. If a CNN is already this anisotropic, the driver
cannot be self-attention, and the "same pathology as NLP" framing is wrong.

**Liang et al. (NeurIPS 2022)** report the cone effect in *untrained* networks:
random-init ResNet reaches ~0.99 mean pairwise cosine, persisting on noise
input. If an untrained backbone is worse than a trained one, the pathology is
architectural and initialisation-driven rather than learned -- which is a very
different paper.

Every encoder sees the SAME boxes, read back from ``detections_crops.csv``, so
the only variable is the encoder. Nothing here re-detects.

Run:
    python -m vsa_cognitive_mapping.encoder_comparison --encoders resnet50 dinov2 resnet50-untrained
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATASET = "lorinachey/spot-telluride-workshop-dataset"


# --------------------------------------------------------------------------
# encoders -- each returns (name, dim, callable(list[PIL]) -> (B, dim) tensor)
# --------------------------------------------------------------------------


def resolve_device(spec="auto"):
    if spec != "auto":
        return spec
    return "cuda" if torch.cuda.is_available() else "cpu"


def _torchvision_cnn(arch, pretrained, device="cpu"):
    import torchvision.models as tvm
    from torchvision import transforms

    if arch == "resnet50":
        w = tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        m = tvm.resnet50(weights=w)
        m.fc = torch.nn.Identity()
        dim = 2048
    elif arch == "vit_b_16":
        w = tvm.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        m = tvm.vit_b_16(weights=w)
        m.heads = torch.nn.Identity()
        dim = 768
    else:
        raise ValueError(arch)
    m.eval().to(device)

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    @torch.no_grad()
    def run(imgs):
        x = torch.stack([tf(im.convert("RGB")) for im in imgs]).to(device)
        return m(x).float().cpu()

    return dim, run


DINO_VARIANTS = {                      # short name -> HF id, output dim
    "small": ("facebook/dinov2-small", 384),
    "base": ("facebook/dinov2-base", 768),
    "large": ("facebook/dinov2-large", 1024),
    "giant": ("facebook/dinov2-giant", 1536),
}


def _dinov2(name="facebook/dinov2-small", device="cpu", img_size=224):
    """DINOv2 ViT-S/14 CLS embedding, preprocessed exactly the way the upstream
    ``init-am`` pipeline does it (``scripts/classroom/detect_and_embed_classroom.py``
    -> ``preprocess_dino_batch``): square BILINEAR resize to 224, /255, ImageNet
    mean/std.

    Deliberately NOT the HF ``AutoImageProcessor`` default, which resizes the
    shortest edge to 256 and then centre-crops 224 -- on a whole frame that is
    harmless, but on an object box with an extreme aspect ratio (a standing
    person, a long desk) it silently discards much of the object we are trying
    to describe. Square resize distorts aspect but keeps all the content, and
    matching upstream keeps this comparable to their DINO runs.

    ``facebook/dinov2-small`` is the same checkpoint as torch.hub's
    ``dinov2_vits14`` (384-d); using the HF copy avoids a second download.
    """
    from PIL import Image
    from transformers import AutoModel

    m = AutoModel.from_pretrained(name).eval().to(device)
    dim = m.config.hidden_size
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    @torch.no_grad()
    def run(imgs):
        arr = [np.asarray(im.convert("RGB").resize((img_size, img_size),
                                                   Image.Resampling.BILINEAR),
                          dtype=np.float32) / 255.0 for im in imgs]
        b = (np.stack(arr) - MEAN) / STD
        x = torch.from_numpy(b).permute(0, 3, 1, 2).contiguous().to(device)
        return m(pixel_values=x).last_hidden_state[:, 0].float().cpu()   # CLS token

    return dim, run


def build_encoder(key, device="cpu", img_size=224):
    """`key` is either a plain name or `dinov2:<variant>` (small|base|large|giant).
    Plain `dinov2` means `dinov2:small`, which is what every result in the repo
    so far was measured with."""
    if key == "resnet50":
        return _torchvision_cnn("resnet50", True, device)
    if key == "resnet50-untrained":
        return _torchvision_cnn("resnet50", False, device)
    if key == "vit_b_16":
        return _torchvision_cnn("vit_b_16", True, device)
    if key == "vit_b_16-untrained":
        return _torchvision_cnn("vit_b_16", False, device)
    if key == "dinov2" or key.startswith("dinov2:"):
        variant = key.split(":", 1)[1] if ":" in key else "small"
        if variant not in DINO_VARIANTS:
            raise ValueError(f"dinov2 variant must be one of {sorted(DINO_VARIANTS)}")
        return _dinov2(DINO_VARIANTS[variant][0], device, img_size)
    raise ValueError(f"unknown encoder {key}")


# --------------------------------------------------------------------------


def load_boxes(out_dir):
    """Read back the exact boxes the YOLO crop stage used."""
    rows = []
    with open(os.path.join(out_dir, "detections_crops.csv"), newline="") as f:
        for r in csv.DictReader(f):
            rows.append((int(r["frame_idx"]), int(r["class_id"]),
                         float(r["x1"]), float(r["y1"]),
                         float(r["x2"]), float(r["y2"])))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="outputs/classroom")
    ap.add_argument("--dataset", default=None,
                    help="sequence config JSON (preferred; works for local folders "
                         "too and sets --out-dir). See configs/TEMPLATE_new_dataset.json")
    ap.add_argument("--repo", default=DATASET)
    ap.add_argument("--rgb-config", default="rgb_d455",
                    help="HF config name, used only when --dataset is absent")
    ap.add_argument("--encoders", nargs="+",
                    default=["resnet50", "dinov2", "resnet50-untrained"])
    ap.add_argument("--pad-frac", type=float, default=0.08,
                    help="must match the YOLO crop stage for a controlled comparison")
    ap.add_argument("--batch", type=int, default=32,
                    help="raise this on a GPU: 128-256 is comfortable for "
                         "dinov2:base, 64-128 for large")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--img-size", type=int, default=224,
                    help="DINOv2 input size; must be a multiple of the patch "
                         "size 14. 224 matches upstream and every result so far")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap detections (0 = all); use for a fast smoke run")
    args = ap.parse_args()

    seq = None
    if args.dataset:
        from vsa_cognitive_mapping.sequences import load_sequence
        seq = load_sequence(args.dataset)
        if args.out_dir == "outputs/classroom":
            args.out_dir = seq.out_dir

    rows = load_boxes(args.out_dir)
    if args.limit:
        rows = rows[:args.limit]
    print(f"{len(rows)} boxes read from detections_crops.csv")

    if seq is not None:
        idx_all = seq.frame_ids()
        get_image = lambda r: seq.image(int(r))          # noqa: E731
    else:
        from datasets import load_dataset
        ds = load_dataset(args.repo, args.rgb_config, split="train")
        idx_all = np.array(ds["frame_idx"], dtype=np.int64)
        get_image = lambda r: ds[int(r)]["image"]        # noqa: E731
    pos = {int(v): i for i, v in enumerate(idx_all)}

    by_frame = defaultdict(list)
    for k, r in enumerate(rows):
        by_frame[r[0]].append(k)
    frames = sorted(by_frame)
    print(f"{len(frames)} distinct frames to re-open\n")

    device = resolve_device(args.device)
    if args.img_size % 14:
        raise SystemExit(f"--img-size {args.img_size} must be a multiple of 14 "
                         f"(DINOv2's patch size)")
    print(f"device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else "")
          + f", batch {args.batch}, img_size {args.img_size}")

    for key in args.encoders:
        print(f"--- {key} ---", flush=True)
        dim, run = build_encoder(key, device, args.img_size)
        E = torch.zeros(len(rows), dim, dtype=torch.float32)
        buf_img, buf_k = [], []

        def flush():
            if not buf_img:
                return
            out = run(buf_img)
            for j, k in enumerate(buf_k):
                E[k] = out[j]
            buf_img.clear(); buf_k.clear()

        for n, fi in enumerate(frames):
            i = pos.get(fi)
            if i is None:
                continue
            img = get_image(i)
            W, H = img.size
            for k in by_frame[fi]:
                _, _, x1, y1, x2, y2 = rows[k]
                pad = args.pad_frac * max(x2 - x1, y2 - y1)
                cx1 = max(0, int(round(x1 - pad))); cy1 = max(0, int(round(y1 - pad)))
                cx2 = min(W, int(round(x2 + pad))); cy2 = min(H, int(round(y2 + pad)))
                if cx2 - cx1 < 2 or cy2 - cy1 < 2:
                    continue
                buf_img.append(img.crop((cx1, cy1, cx2, cy2))); buf_k.append(k)
                if len(buf_img) >= args.batch:
                    flush()
            if n % 100 == 0:
                print(f"  frame {n}/{len(frames)}", flush=True)
        flush()

        # `dinov2:base` -> crop_embeddings_dinov2-base.pt, so variants coexist
        # instead of silently overwriting each other
        safe = key.replace(":", "-")
        out_path = os.path.join(args.out_dir, f"crop_embeddings_{safe}.pt")
        torch.save({"embedding": E,
                    "frame_idx": torch.tensor([r[0] for r in rows]),
                    "class_id": torch.tensor([r[1] for r in rows]),
                    "encoder": key, "dim": dim, "device": device,
                    "img_size": args.img_size,
                    "pad_frac": args.pad_frac}, out_path)
        print(f"  saved {tuple(E.shape)} -> {out_path}\n", flush=True)


if __name__ == "__main__":
    main()
