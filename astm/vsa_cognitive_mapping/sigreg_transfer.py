"""Build the SIGReg-transferred crop-embedding file (WP4 step 1).

Applies the trained JEPA+SIGReg head (Linear 384->128 + BatchNorm1d, eval
mode with running statistics) to the cached per-DETECTION DINOv2 crop
embeddings, producing ``crop_embeddings_sigreg.pt`` with the same dict shape
as the source so it drops into ``crosstalk_scaling --content`` and
``heldout_eval --encoders`` unchanged.

PROVENANCE / OFF-DISTRIBUTION CAVEAT (carry this everywhere the numbers go):
the head was trained on WHOLE-FRAME DINOv2 CLS embeddings of a 3 Hz /
235-frame subsample of the classroom walk (tracker entry (ae)), with a
JEPA+SIGReg objective (70 training transitions). Here it is applied to
PER-CROP embeddings of all 2,429 frames — a transfer, not its training
distribution. Output is 128-d against dinov2's 384-d input, so dimension is
NOT matched across the comparison.

The DINOv2 network is never loaded; only the small linear head is applied.

Usage (from repo root):
    python -m vsa_cognitive_mapping.sigreg_transfer
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_CKPT = r"C:\vsacm_jepa\checkpoints\jepa_run1_scratch\final.pt"
DEFAULT_OUT_DIR = os.path.join("outputs", "classroom")
BN_EPS = 1e-5  # torch.nn.BatchNorm1d default


def apply_head(X: np.ndarray, sd) -> np.ndarray:
    """Linear + BatchNorm1d (eval mode, running stats) as float64 numpy."""
    W = sd["head.proj.weight"].numpy().astype(np.float64)      # (128, 384)
    b = sd["head.proj.bias"].numpy().astype(np.float64)        # (128,)
    g = sd["head.bn.weight"].numpy().astype(np.float64)
    beta = sd["head.bn.bias"].numpy().astype(np.float64)
    mu = sd["head.bn.running_mean"].numpy().astype(np.float64)
    var = sd["head.bn.running_var"].numpy().astype(np.float64)
    Z = X @ W.T + b
    return (Z - mu) / np.sqrt(var + BN_EPS) * g + beta


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--source", default="crop_embeddings_dinov2.pt")
    ap.add_argument("--dest", default="crop_embeddings_sigreg.pt")
    args = ap.parse_args()

    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if "model" in sd:                       # some runs nest the state dict
        sd = sd["model"]

    src = torch.load(os.path.join(args.out_dir, args.source), weights_only=False)
    X = src["embedding"].numpy().astype(np.float64)            # (n_det, 384)
    Y = apply_head(X, sd)                                      # (n_det, 128)

    out = {
        "embedding": torch.from_numpy(Y.astype(np.float32)),
        "frame_idx": src["frame_idx"].clone(),
        "class_id": src["class_id"].clone(),
        "encoder": "sigreg-head-on-dinov2",
        "dim": int(Y.shape[1]),
        "pad_frac": src.get("pad_frac"),
        "note": ("JEPA+SIGReg head (Linear 384->128 + BatchNorm1d eval mode, "
                 f"eps={BN_EPS}) from {args.ckpt} applied per-detection to "
                 f"{args.source}. OFF-DISTRIBUTION: the head was trained on "
                 "whole-frame CLS embeddings of a 3 Hz / 235-frame subsample "
                 "(70 transitions, tracker entry (ae)); applied here to "
                 "per-crop embeddings — a transfer. 128-d output vs 384-d "
                 "input: dimension is not matched across encoders."),
    }
    dest = os.path.join(args.out_dir, args.dest)
    torch.save(out, dest)

    # sanity summary
    n = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-12)
    rng = np.random.RandomState(0)
    idx = rng.choice(len(n), size=min(500, len(n)), replace=False)
    S = n[idx] @ n[idx].T
    off = S[~np.eye(len(idx), dtype=bool)]
    print(f"wrote {dest}: {Y.shape[0]} detections x {Y.shape[1]} dims")
    print(f"  per-dim mean |mean|: {np.abs(Y.mean(axis=0)).mean():.4f}  "
          f"per-dim std mean: {Y.std(axis=0).mean():.4f}")
    print(f"  mean pairwise cos (500-row subsample): {off.mean():+.4f} "
          f"(mean |cos| {np.abs(off).mean():.4f})")


if __name__ == "__main__":
    main()
