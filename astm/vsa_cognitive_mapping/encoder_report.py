"""Compare encoders on identical crops: isotropy, rogue dimensions, retrieval.

Tests two literature predictions at once.

**Godey et al. (EACL 2024)** -- anisotropy tracks self-attention, so ViTs should
be anisotropic and CNNs largely fine. Here YOLOv8n and ResNet-50 are CNNs and
DINOv2 ViT-S/14 is a ViT, all describing the same object boxes.

**Liang et al. (NeurIPS 2022)** -- the cone effect appears in *untrained*
networks. If random-init ResNet-50 is not clearly worse than the trained one,
the anisotropy is learned rather than architectural.

Caveat carried in the output: dimensionality differs across encoders (256 / 384
/ 2048), so effective rank is reported both absolutely and as a fraction of the
available dimensions. Mean cosine and IsoScore are dimension-sensitive too --
compare within a column, cautiously across rows.

Run:
    python -m vsa_cognitive_mapping.encoder_report --out-dir outputs/classroom
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vsa_cognitive_mapping.isotropy_ladder import (  # noqa: E402
    chance, cos_contributions, eff_rank, isoscore, rung_zscore, top1_retrieval)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="outputs/classroom")
    ap.add_argument("--gaps", type=int, nargs="*", default=[30, 300, 900])
    args = ap.parse_args()

    base = torch.load(os.path.join(args.out_dir, "crop_embeddings.pt"),
                      weights_only=False)
    cls = base["class_id"].numpy()
    fr = base["frame_idx"].numpy()

    encs = [("yolov8n (CNN, detection)", "crop_embeddings.pt"),
            ("resnet50 (CNN, ImageNet)", "crop_embeddings_resnet50.pt"),
            ("dinov2-S/14 (ViT, SSL)", "crop_embeddings_dinov2.pt"),
            ("resnet50 UNTRAINED (CNN)", "crop_embeddings_resnet50-untrained.pt")]

    ch = chance(cls)
    print(f"{len(cls)} crops, {len(np.unique(cls))} classes, chance top1 = {ch:.3f}\n")

    hdr = (f"{'encoder':<27}{'dim':>6}{'cos':>8}{'effrk':>7}{'/dim':>7}"
           f"{'Iso':>8}{'top1%':>7}{'d50%':>6}"
           + "".join(f"{'@' + str(g):>8}" for g in args.gaps)
           + "".join(f"{'z@' + str(g):>8}" for g in args.gaps))
    print(hdr); print("-" * len(hdr))

    rows = []
    for label, fn in encs:
        p = os.path.join(args.out_dir, fn)
        if not os.path.exists(p):
            print(f"{label:<27} MISSING ({fn})")
            continue
        d = torch.load(p, weights_only=False)
        X = d["embedding"].numpy().astype(np.float64)
        D = X.shape[1]

        contrib, cm = cos_contributions(X)
        order = np.argsort(contrib)[::-1]
        cumul = np.cumsum(contrib[order]) / cm if cm != 0 else np.zeros(D)
        top_share = float(contrib[order[0]] / cm) if cm != 0 else 0.0
        d50 = int(np.searchsorted(cumul, 0.5) + 1)

        pr, k95 = eff_rank(X)
        iso = isoscore(X)
        raw_acc = [top1_retrieval(X, cls, fr, g) for g in args.gaps]
        Z = rung_zscore(X)
        z_acc = [top1_retrieval(Z, cls, fr, g) for g in args.gaps]

        print(f"{label:<27}{D:>6}{cm:>+8.3f}{pr:>7.1f}{pr / D:>7.3f}{iso:>8.4f}"
              f"{top_share:>7.1%}{d50:>6}"
              + "".join(f"{a:>8.3f}" for a in raw_acc)
              + "".join(f"{a:>8.3f}" for a in z_acc))

        rows.append(dict(encoder=label, dim=D, cos_mean=cm, eff_rank=pr,
                         eff_rank_frac=pr / D, k95=k95, isoscore=iso,
                         timkey_top_dim_share=top_share, timkey_dims_for_50pct=d50,
                         top1_raw={str(g): a for g, a in zip(args.gaps, raw_acc)},
                         top1_zscored={str(g): a for g, a in zip(args.gaps, z_acc)}))

    print(f"\ntop1% = share of the mean cosine carried by the single worst dimension "
          f"(Timkey: 76-99% in GPT-2/BERT/XLNet)")
    print(f"d50%  = dimensions needed for half the mean cosine (Timkey: ~1)")
    print(f"@g    = class top-1 retrieval excluding neighbours within g frames; "
          f"z@g = same after per-dimension z-scoring")

    out = os.path.join(args.out_dir, "encoder_report.json")
    with open(out, "w") as f:
        json.dump({"chance": ch, "n": int(len(cls)), "rows": rows}, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
