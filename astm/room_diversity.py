"""Does semantic diversity predict effective rank? (paper-plan Q1/Q7 control)

The low-rank hypothesis says our anisotropy is *low-rank-from-few-classes*
rather than the rogue-dimension pathology NLP reports. If that is right, a
scene with richer, more evenly distributed semantic content should spread its
features over more directions -- higher effective rank, lower crosstalk -- and
an impoverished one should collapse further.

`classroom` and `school_run1` are the free test: same robot, same camera, same
detector, different spaces.

Two controls that make the comparison mean anything:

* **Matched N.** Effective rank, mean cosine and IsoScore all move with sample
  size, and school_run1 has 4.6x more detections. Everything is reported at a
  common subsample as well as at full size.
* **Diversity measured, not assumed.** Class entropy and the effective number
  of classes (exp of entropy) say which scene is actually richer, rather than
  guessing from the name.

Run:
    python -m vsa_cognitive_mapping.room_diversity
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
    chance, cos_contributions, eff_rank, isoscore, rung_pca_whiten,
    rung_zscore, top1_retrieval)


def diversity(cls):
    """Shannon entropy of the class distribution, and its exponential -- the
    'effective number of classes', which is the quantity the low-rank story
    actually predicts on."""
    _, cnt = np.unique(cls, return_counts=True)
    p = cnt / cnt.sum()
    H = float(-(p * np.log(p)).sum())
    return H, float(np.exp(H)), len(cnt)


def summarise(tag, X, cls, fr, gap, do_retrieval=True):
    contrib, cm = cos_contributions(X)
    order = np.argsort(contrib)[::-1]
    cumul = np.cumsum(contrib[order]) / cm
    pr, k95 = eff_rank(X)
    iso = isoscore(X)
    H, effK, nK = diversity(cls)
    row = dict(tag=tag, n=int(len(X)), dim=int(X.shape[1]), cos_mean=cm,
               eff_rank=pr, k95=k95, isoscore=iso,
               timkey_top_share=float(contrib[order[0]] / cm),
               timkey_d50=int(np.searchsorted(cumul, 0.5) + 1),
               entropy=H, eff_classes=effK, n_classes=nK, chance=chance(cls))
    if do_retrieval:
        row["top1_raw"] = top1_retrieval(X, cls, fr, gap)
        row["top1_zscored"] = top1_retrieval(rung_zscore(X), cls, fr, gap)
        row["top1_whiten128"] = top1_retrieval(rung_pca_whiten(X, 128), cls, fr, gap)
    return row


def load(out_dir):
    d = torch.load(os.path.join(out_dir, "crop_embeddings.pt"), weights_only=False)
    return (d["embedding"].numpy().astype(np.float64),
            d["class_id"].numpy(), d["frame_idx"].numpy())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", nargs="+",
                    default=["outputs/classroom", "outputs/school_run1"])
    ap.add_argument("--gap", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    loaded = {}
    for s in args.scenes:
        X, c, f = load(s)
        loaded[os.path.basename(s)] = (X, c, f)
        print(f"{os.path.basename(s):<14} {len(X):>6} crops, "
              f"{len(np.unique(c))} classes")
    n_match = min(len(v[0]) for v in loaded.values())
    print(f"\nmatched subsample N = {n_match}\n")

    rows = []
    for name, (X, c, f) in loaded.items():
        rows.append(summarise(f"{name} (full)", X, c, f, args.gap))
        if len(X) > n_match:
            rng = np.random.RandomState(args.seed)
            # contiguous-in-time subsample would bias the gap control, so take
            # an evenly spaced slice: preserves the temporal span, thins density
            sel = np.linspace(0, len(X) - 1, n_match).astype(int)
            rows.append(summarise(f"{name} (matched)", X[sel], c[sel], f[sel], args.gap))

    hdr = (f"{'scene'                  :<26}{'n':>7}{'cos':>8}{'effrk':>7}{'k95':>5}"
           f"{'Iso':>8}{'H':>7}{'effK':>7}{'d50':>5}{'top1':>7}{'z':>7}{'wht128':>8}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['tag']:<26}{r['n']:>7}{r['cos_mean']:>+8.3f}{r['eff_rank']:>7.1f}"
              f"{r['k95']:>5}{r['isoscore']:>8.4f}{r['entropy']:>7.2f}"
              f"{r['eff_classes']:>7.1f}{r['timkey_d50']:>5}"
              f"{r.get('top1_raw', float('nan')):>7.3f}"
              f"{r.get('top1_zscored', float('nan')):>7.3f}"
              f"{r.get('top1_whiten128', float('nan')):>8.3f}")

    print("\nH = class entropy (nats); effK = exp(H) = effective number of classes")
    print("prediction: richer scene (higher effK) -> higher eff-rank, lower crosstalk")

    # name by scene set + gap: a fixed filename silently overwrote the two-scene
    # table when this was re-run at a different gap
    tag = "-".join(os.path.basename(s) for s in args.scenes)
    out = f"outputs/classroom/room_diversity_{tag}_gap{args.gap}.json"
    with open(out, "w") as fh:
        json.dump(rows, fh, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
