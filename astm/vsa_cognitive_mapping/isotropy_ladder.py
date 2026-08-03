"""Isotropisation ablation ladder + the Timkey rogue-dimension diagnostic.

Answers one question the paper plan (Q2) turns on: **is our anisotropy the same
pathology NLP reports, or a different one?**

Timkey & van Schijndel (EMNLP 2021) showed transformer anisotropy is mostly the
work of 1-5 "rogue" dimensions -- remove five and the mean cosine collapses --
and that the cheapest effective fix is per-dimension standardisation, not
whitening. If our features behave the same way, the story is "known anisotropy,
applied to VSAs". If instead the mean cosine is spread evenly across many
dimensions, it is a genuinely spectral, low-rank-from-few-classes pathology,
which is a different and more interesting claim.

The decomposition is exact, not sampled. With u_i the L2-normalised vectors,

    mean_{i != j} cos(u_i, u_j)  =  sum_d [ (sum_i u_i[d])^2 - sum_i u_i[d]^2 ]
                                    / (n (n - 1))

so each dimension's additive contribution to the mean off-diagonal cosine falls
out in O(nD) and the contributions sum exactly to the measured mean.

The ladder then measures every rung on BOTH axes that matter -- isotropy *and*
downstream retrieval -- because the plan predicts an optimum rather than a
monotone (strict isotropy is incompatible with cluster structure, and object
classes are clusters).

Run:
    python -m vsa_cognitive_mapping.isotropy_ladder --out-dir outputs/classroom
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------


def unit(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def cos_contributions(X):
    """Exact per-dimension additive contribution to the mean off-diagonal cosine.

    Returns (contrib, mean_cos) with contrib.sum() == mean_cos up to rounding.
    """
    U = unit(X)
    n = len(U)
    s1 = U.sum(axis=0)                      # (D,)
    s2 = (U ** 2).sum(axis=0)               # (D,)
    contrib = (s1 ** 2 - s2) / (n * (n - 1))
    return contrib, float(contrib.sum())


def eff_rank(X):
    """Participation ratio of the covariance spectrum, and the 95%-variance count."""
    Xc = X - X.mean(axis=0, keepdims=True)
    lam = np.linalg.svd(Xc, compute_uv=False) ** 2
    if lam.sum() <= 0:
        return 0.0, 0
    pr = float(lam.sum() ** 2 / (lam ** 2).sum())
    k95 = int(np.searchsorted(np.cumsum(lam) / lam.sum(), 0.95) + 1)
    return pr, k95


def isoscore(X):
    """IsoScore (Rudman et al., Findings ACL 2022). 0 = maximally anisotropic,
    1 = isotropic. Defined on the PCA-reoriented variance profile, so it is
    invariant to rotation -- unlike mean cosine."""
    Xc = X - X.mean(axis=0, keepdims=True)
    D = Xc.shape[1]
    lam = np.linalg.svd(Xc, compute_uv=False) ** 2 / max(len(Xc) - 1, 1)
    nrm = np.linalg.norm(lam)
    if nrm <= 0:
        return 0.0
    lam_hat = np.sqrt(D) * lam / nrm                       # ||lam_hat|| = sqrt(D)
    delta = np.linalg.norm(lam_hat - 1.0) / np.sqrt(2 * (D - np.sqrt(D)))
    phi = (D - delta ** 2 * (D - np.sqrt(D))) ** 2 / D ** 2
    return float((phi * D - 1) / (D - 1))


def top1_retrieval(F, cls, fr, gap, block=512):
    """Leave-one-out top-1 cosine retrieval, class-level, excluding any neighbour
    within `gap` frames. The gap is the whole point: without it the nearest
    neighbour is the same object one frame later."""
    U = unit(F)
    hits = tot = 0
    for s in range(0, len(U), block):
        S = U[s:s + block] @ U.T
        S[np.abs(fr[s:s + block, None] - fr[None, :]) <= gap] = -2.0
        best = S.argmax(1)
        ok = S.max(1) > -1.5
        hits += int((cls[best][ok] == cls[s:s + block][ok]).sum())
        tot += int(ok.sum())
    return hits / max(tot, 1)


def chance(cls):
    _, cnt = np.unique(cls, return_counts=True)
    p = cnt / cnt.sum()
    return float((p ** 2).sum())


# --------------------------------------------------------------------------
# ladder rungs
# --------------------------------------------------------------------------


def rung_centre(X):
    return X - X.mean(axis=0, keepdims=True)


def rung_zscore(X):
    """Per-dimension standardisation -- Timkey's recommended fix, and the
    cheapest rung on the ladder."""
    Xc = X - X.mean(axis=0, keepdims=True)
    return Xc / (Xc.std(axis=0, keepdims=True) + 1e-9)


def rung_drop_top(X, k, order):
    """Zero the k dimensions contributing most to the mean cosine."""
    Xc = X - X.mean(axis=0, keepdims=True)
    Y = Xc.copy()
    Y[:, order[:k]] = 0.0
    return Y


def rung_pca_whiten(X, k):
    Xc = X - X.mean(axis=0, keepdims=True)
    _, s, vt = np.linalg.svd(Xc, full_matrices=False)
    Z = Xc @ vt[:k].T
    return Z / (Z.std(axis=0, keepdims=True) + 1e-9)


def rung_zca(X, eps=1e-6):
    """Full ZCA: identity covariance, but kept in the ORIGINAL basis (unlike PCA
    whitening, which also rotates). Expected to overshoot -- it equalises the
    nuisance directions with the discriminative ones."""
    Xc = X - X.mean(axis=0, keepdims=True)
    C = np.cov(Xc, rowvar=False)
    lam, V = np.linalg.eigh(C)
    lam = np.maximum(lam, eps)
    W = V @ np.diag(1.0 / np.sqrt(lam)) @ V.T
    return Xc @ W


def rung_random_orth(X, seed=0):
    """Control rung. A random orthogonal map cannot change any cosine, so every
    metric that is rotation-invariant must be unchanged. If IsoScore moves here,
    the implementation is wrong."""
    rng = np.random.RandomState(seed)
    Q, _ = np.linalg.qr(rng.randn(X.shape[1], X.shape[1]))
    return X @ Q


# --------------------------------------------------------------------------


def analyse(name, X, cls, fr, gaps, order):
    rungs = {
        "raw": X,
        "centred": rung_centre(X),
        "L2-normalised": unit(X),
        "z-scored (per-dim)": rung_zscore(X),
        "drop top-1 dim": rung_drop_top(X, 1, order),
        "drop top-3 dims": rung_drop_top(X, 3, order),
        "drop top-5 dims": rung_drop_top(X, 5, order),
        "drop top-10 dims": rung_drop_top(X, 10, order),
        "PCA-whiten 32": rung_pca_whiten(X, 32),
        "PCA-whiten 64": rung_pca_whiten(X, 64),
        "PCA-whiten 128": rung_pca_whiten(X, 128),
        "ZCA (full)": rung_zca(X),
        "random orthogonal": rung_random_orth(X),
    }
    ch = chance(cls)
    hdr = (f"{'rung':<21}{'cos':>8}{'effrank':>9}{'k95':>6}{'IsoScore':>10}"
           + "".join(f"{'top1@' + str(g):>11}" for g in gaps))
    print(f"\n=== {name} ===   (chance top1 = {ch:.3f})")
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for rn, F in rungs.items():
        _, cm = cos_contributions(F)
        pr, k95 = eff_rank(F)
        iso = isoscore(F)
        accs = {g: top1_retrieval(F, cls, fr, g) for g in gaps}
        rows.append(dict(rung=rn, cos_mean=cm, eff_rank=pr, k95=k95,
                         isoscore=iso, top1={str(g): accs[g] for g in gaps}))
        print(f"{rn:<21}{cm:>+8.3f}{pr:>9.1f}{k95:>6d}{iso:>10.4f}"
              + "".join(f"{accs[g]:>11.3f}" for g in gaps))
    return {"chance": ch, "rows": rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="outputs/classroom")
    ap.add_argument("--gaps", type=int, nargs="*", default=[30, 300])
    ap.add_argument("--crops-name", default="crop_embeddings.pt")
    args = ap.parse_args()

    d = torch.load(os.path.join(args.out_dir, args.crops_name), weights_only=False)
    X = d["embedding"].numpy().astype(np.float64)
    cls = d["class_id"].numpy()
    fr = d["frame_idx"].numpy()
    print(f"{len(X)} crops, {X.shape[1]}-d, {len(np.unique(cls))} classes")

    # ---- Timkey diagnostic on the raw features ---------------------------
    contrib, cm = cos_contributions(X)
    order = np.argsort(contrib)[::-1]
    frac = np.cumsum(contrib[order]) / cm
    print(f"\nTIMKEY DIAGNOSTIC (crops)   mean off-diagonal cosine = {cm:+.4f}")
    print(f"  top dim  {order[0]:>4}  contributes {contrib[order[0]]:+.4f} "
          f"= {contrib[order[0]] / cm:6.1%} of the mean cosine")
    for k in (1, 3, 5, 10, 20, 50):
        print(f"  top {k:>2} dims cumulative: {frac[k - 1]:6.1%}")
    print(f"  dims needed for 50% of the mean cosine: "
          f"{int(np.searchsorted(frac, 0.5) + 1)} of {X.shape[1]}")
    print(f"  dims needed for 90%: {int(np.searchsorted(frac, 0.9) + 1)}")

    report = {"timkey_crops": {
        "mean_cos": cm,
        "top_dim": int(order[0]),
        "top_dim_share": float(contrib[order[0]] / cm),
        "cumulative_share": {str(k): float(frac[k - 1]) for k in (1, 3, 5, 10, 20, 50)},
        "dims_for_50pct": int(np.searchsorted(frac, 0.5) + 1),
        "dims_for_90pct": int(np.searchsorted(frac, 0.9) + 1),
        "dim": int(X.shape[1])}}

    report["crops"] = analyse("CROP features (per detection)", X, cls, fr,
                              args.gaps, order)

    # ---- same ladder on the frame features, for contrast -----------------
    fp = os.path.join(args.out_dir, "embeddings.pt")
    if os.path.exists(fp):
        fd = torch.load(fp, weights_only=True)
        Xf = fd["embedding"].numpy().astype(np.float64)
        cf, mc = cos_contributions(Xf)
        of = np.argsort(cf)[::-1]
        ff = np.cumsum(cf[of]) / mc
        print(f"\nTIMKEY DIAGNOSTIC (frames)  mean off-diagonal cosine = {mc:+.4f}")
        print(f"  top dim contributes {cf[of[0]] / mc:6.1%};  "
              f"top 5 = {ff[4]:6.1%};  dims for 50% = "
              f"{int(np.searchsorted(ff, 0.5) + 1)} of {Xf.shape[1]}")
        report["timkey_frames"] = {
            "mean_cos": float(mc), "top_dim_share": float(cf[of[0]] / mc),
            "cumulative_share": {str(k): float(ff[k - 1]) for k in (1, 3, 5, 10, 20, 50)},
            "dims_for_50pct": int(np.searchsorted(ff, 0.5) + 1),
            "dim": int(Xf.shape[1])}

    out = os.path.join(args.out_dir, "isotropy_ladder.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
