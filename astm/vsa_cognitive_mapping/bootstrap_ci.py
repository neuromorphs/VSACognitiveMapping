"""Bootstrap CIs for the isotropy-ladder retrieval cells (T4d, robustness batch).

Resamples CROPS (rows) with replacement and recomputes the full treatment +
leave-one-out gap-excluded top-1 retrieval per replicate, so the CI covers both
the sampling noise of the crop set and the treatment statistics (mean/std/PCA)
refit on each resample.

Notes on honesty:
  * A duplicated crop keeps its frame_idx, and ``top1_retrieval`` masks every
    pair within ``gap`` frames — |diff| = 0 <= gap — so a bootstrap duplicate can
    never retrieve itself. No self-match leakage.
  * ``raw`` here means the untreated features, identical to the ladder's "raw"
    rung (L2-normalisation happens inside top1_retrieval either way).
  * CIs are percentile (2.5/97.5) over ``--reps`` replicates.

Writes/updates the output JSON incrementally every ``--save-every`` reps so an
interrupted run still leaves usable partial results.

Run (repo root):
    python vsa_cognitive_mapping/bootstrap_ci.py --out-dir outputs/classroom \
        --gaps 300 --out-json outputs/robustness/bootstrap_classroom.json
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
    chance, rung_pca_whiten, rung_zscore, top1_retrieval)


def treatments(X):
    """The three key ladder cells: raw, z-scored (per-dim), PCA-whiten 128."""
    return {"raw": X,
            "zscored": rung_zscore(X),
            "whitened128": rung_pca_whiten(X, 128)}


def cells(X, cls, fr, gaps):
    return {tn: {str(g): top1_retrieval(F, cls, fr, g) for g in gaps}
            for tn, F in treatments(X).items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="outputs/classroom")
    ap.add_argument("--crops-name", default="crop_embeddings.pt")
    ap.add_argument("--gaps", type=int, nargs="+", default=[300])
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    d = torch.load(os.path.join(args.out_dir, args.crops_name),
                   weights_only=False)
    X = d["embedding"].numpy().astype(np.float64)
    cls = d["class_id"].numpy()
    fr = d["frame_idx"].numpy()
    n = len(X)
    print(f"{n} crops, {X.shape[1]}-d, {len(np.unique(cls))} classes, "
          f"chance {chance(cls):.3f}; gaps {args.gaps}; {args.reps} reps")

    point = cells(X, cls, fr, args.gaps)
    print("point estimates:", json.dumps(point))

    rng = np.random.RandomState(args.seed)
    boots = {tn: {str(g): [] for g in args.gaps} for tn in point}

    def save():
        rep = {"n_crops": n, "dim": int(X.shape[1]),
               "chance": chance(cls), "gaps": args.gaps,
               "reps_done": len(next(iter(
                   next(iter(boots.values())).values()))),
               "reps_requested": args.reps, "seed": args.seed,
               "crops_file": os.path.join(args.out_dir, args.crops_name),
               "point": point, "cells": {}}
        for tn in boots:
            rep["cells"][tn] = {}
            for g in boots[tn]:
                v = np.asarray(boots[tn][g], float)
                if len(v):
                    rep["cells"][tn][g] = {
                        "point": point[tn][g],
                        "boot_mean": float(v.mean()),
                        "boot_std": float(v.std(ddof=1)) if len(v) > 1 else None,
                        "ci95": [float(np.percentile(v, 2.5)),
                                 float(np.percentile(v, 97.5))],
                        "n_boot": int(len(v))}
        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(rep, f, indent=2)

    for r in range(args.reps):
        idx = rng.randint(0, n, size=n)          # resample crops w/ replacement
        c = cells(X[idx], cls[idx], fr[idx], args.gaps)
        for tn in c:
            for g in c[tn]:
                boots[tn][g].append(c[tn][g])
        if (r + 1) % args.save_every == 0 or r == args.reps - 1:
            save()
            print(f"  rep {r + 1}/{args.reps} saved")

    save()
    print("\n==== 95% bootstrap CIs (percentile) ====")
    for tn in boots:
        for g in boots[tn]:
            v = np.asarray(boots[tn][g], float)
            print(f"  {tn:<12} gap {g:>4}: point {point[tn][g]:.4f}  "
                  f"CI [{np.percentile(v, 2.5):.4f}, "
                  f"{np.percentile(v, 97.5):.4f}]  "
                  f"(boot mean {v.mean():.4f} +/- {v.std(ddof=1):.4f})")
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
