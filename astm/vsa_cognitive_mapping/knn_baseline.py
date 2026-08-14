"""kNN-at-matched-bytes baseline for the blocked held-out place recall (T6).

Pre-answers "why not a vector database?" with a measured table on the
IDENTICAL task that `heldout_eval.py` defines: the memory set stores
(descriptor, pose) pairs; a query answers the pose of the nearest stored
descriptor by cosine similarity; scoring uses the same oracle / constant
brackets, the same blocked split (contiguous held-out segments plus
+-`gap`-frame eviction), and the same seeds.

Systems
-------
exact       full float32 descriptors, exact cosine nearest neighbour.
pq{m}       product quantisation: descriptor split into m contiguous
            subvectors, 8-bit (256-codeword) k-means codebook per subvector
            trained on the MEMORY SET ONLY (per split, per seed); stored
            vectors are replaced by their PQ reconstruction; retrieval is the
            same cosine NN of the raw query against the reconstructions
            (asymmetric-distance style).

Everything task-defining is imported and REUSED from `heldout_eval`
(`frame_descriptors`, `treat`, `make_split`) -- this file must never redefine
the task. Stated assumptions of that reuse:

  * treatment statistics (mean/std for `zscored`) are fit on ALL frames,
    including held-out ones, exactly as `heldout_eval.treat` does when the
    VSA path calls it. The small shared-statistics leak is carried
    identically in both systems, so the comparison stays fair.
  * split = blocked, frac 0.30, gap 45 frames, rng = RandomState(1000+seed),
    seeds 0-4 -- byte-identical splits to the (ai) table (833 stored /
    819 queries on the classroom).
  * signal = (const - median) / (const - oracle) with oracle = median
    query->nearest-stored-pose distance and const = median distance to the
    centroid of stored poses -- computed per seed exactly as heldout_eval.

Byte accounting (per system, all stated in the JSON):

  poses       n_mem x 2 x 4 B float32 (every kNN system must store them).
  exact       n_mem x D x 4 B float32 descriptors + poses.
  pq{m}       n_mem x m x 1 B codes + poses + codebook 256 x D x 4 B
              (sum over subvectors of 256 x (D/m) x 4 = 256 x D x 4,
              independent of m; reported separately because it is a
              fixed model cost, not a per-vector cost).
  VSA (quoted from heldout_eval_5seed.json, entry (ai)) -- one hd=4096
              complex128 trace = 65,536 B, PLUS the honest note that the
              decode grid as implemented materialises grid^2 x hd complex64
              (48x48x4096x8 B = 75.5 MB); the grid is derivable from the
              2 x hd position-encoder phases (32 KB float32) so it is
              compute, not storage, but the as-run figure is reported too.

Run from the repo root:

    python -m vsa_cognitive_mapping.knn_baseline
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vsa_cognitive_mapping.heldout_eval import (  # noqa: E402  (task reuse)
    frame_descriptors, treat, make_split)
from vsa_cognitive_mapping.classroom_pipeline import (  # noqa: E402
    load_poses_interpolated)


# ---------------------------------------------------------------- PQ pieces

def _kmeans(X, k, rng, iters=25):
    """Plain-Lloyd k-means, numpy only. Empty clusters re-seeded to a random
    training point. Returns centroids (k, d)."""
    n = X.shape[0]
    k = min(k, n)
    C = X[rng.choice(n, size=k, replace=False)].copy()
    x2 = (X ** 2).sum(1)[:, None]
    for _ in range(iters):
        d2 = x2 - 2.0 * X @ C.T + (C ** 2).sum(1)[None, :]
        a = d2.argmin(1)
        for j in range(k):
            m = a == j
            if m.any():
                C[j] = X[m].mean(0)
            else:
                C[j] = X[rng.randint(n)]
    return C


def pq_fit(Xmem, m, rng, k=256, iters=25):
    """Train one 8-bit codebook per contiguous subvector on the memory set."""
    n, D = Xmem.shape
    assert D % m == 0, f"D={D} not divisible by m={m}"
    sub = D // m
    return [
        _kmeans(np.ascontiguousarray(Xmem[:, i * sub:(i + 1) * sub]),
                k, rng, iters)
        for i in range(m)
    ]


def pq_reconstruct(X, books):
    """Encode X with the codebooks and return the reconstruction (n, D)."""
    m = len(books)
    sub = X.shape[1] // m
    R = np.empty_like(X)
    for i, C in enumerate(books):
        Xi = np.ascontiguousarray(X[:, i * sub:(i + 1) * sub])
        d2 = ((Xi ** 2).sum(1)[:, None] - 2.0 * Xi @ C.T
              + (C ** 2).sum(1)[None, :])
        R[:, i * sub:(i + 1) * sub] = C[d2.argmin(1)]
    return R


# ---------------------------------------------------------------- scoring

def _unit(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def knn_answer(Mdesc, Mxy, Qdesc):
    """Cosine NN: each query answers the pose of its most similar stored
    descriptor. Returns (n_query, 2) answered poses."""
    S = _unit(Qdesc) @ _unit(Mdesc).T
    return Mxy[S.argmax(1)]


def brackets(XY, mem, qry):
    """oracle / constant, computed exactly as heldout_eval does per seed."""
    dx = XY[qry, 0][:, None] - XY[mem, 0][None, :]
    dy = XY[qry, 1][:, None] - XY[mem, 1][None, :]
    oracle = float(np.median(np.hypot(dx, dy).min(axis=1)))
    cen = XY[mem].mean(axis=0)
    const = float(np.median(np.hypot(XY[qry, 0] - cen[0],
                                     XY[qry, 1] - cen[1])))
    return oracle, const


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="outputs/classroom")
    ap.add_argument("--result-dir", default="outputs/knn_baseline")
    ap.add_argument("--crop-file", default="crop_embeddings_dinov2.pt")
    ap.add_argument("--encoder-name", default="dinov2")
    ap.add_argument("--treatments", nargs="+", default=["raw", "zscored"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--frac", type=float, default=0.3)
    ap.add_argument("--gap", type=int, default=45)
    ap.add_argument("--pq-m", type=int, nargs="+",
                    default=[48, 24, 16, 8, 4, 2, 1],
                    help="subvector counts to sweep (48 -> ~48 B/vec codes, "
                         "16 -> ~16 B/vec codes, then down to 1 B/vec to "
                         "find where accuracy visibly degrades, if it does)")
    ap.add_argument("--kmeans-iters", type=int, default=25)
    args = ap.parse_args()

    os.makedirs(args.result_dir, exist_ok=True)

    # poses: identical path to heldout_eval (timestamps from the yolov8n crop
    # file, LIO-SAM odometry interpolated to them)
    base = torch.load(os.path.join(args.out_dir, "crop_embeddings.pt"),
                      weights_only=False)
    ts_of = {}
    for f, t in zip(base["frame_idx"].numpy(), base["timestamp_ns"].numpy()):
        ts_of.setdefault(int(f), int(t))

    uf, E = frame_descriptors(args.out_dir, args.crop_file)
    ts = np.array([ts_of[int(f)] for f in uf], np.int64)
    x, y, _ = load_poses_interpolated(ts)
    XY = np.stack([x, y], axis=1)
    D = E.shape[1]
    print(f"{len(uf)} frames, descriptor dim {D}, encoder "
          f"{args.encoder_name}, seeds {args.seeds}")

    results = []
    for tname in args.treatments:
        Et = treat(E, tname).astype(np.float64)

        # accumulators: system -> list of per-seed dicts
        per_system = {}

        for s in args.seeds:
            rng = np.random.RandomState(1000 + s)
            mem, qry = make_split(uf, "blocked", args.frac, args.gap, rng)
            if mem.sum() < 20 or qry.sum() < 20:
                print(f"  seed {s}: degenerate split, skipped")
                continue
            oracle, const = brackets(XY, mem, qry)
            n_mem, n_qry = int(mem.sum()), int(qry.sum())
            Mdesc, Qdesc, Mxy = Et[mem], Et[qry], XY[mem]

            def score(ans):
                err = np.hypot(ans[:, 0] - XY[qry, 0], ans[:, 1] - XY[qry, 1])
                med = float(np.median(err))
                sig = (const - med) / max(const - oracle, 1e-9)
                return med, sig

            # ---- exact float32 kNN
            med, sig = score(knn_answer(Mdesc.astype(np.float32).astype(
                np.float64), Mxy, Qdesc))
            per_system.setdefault("exact_f32", []).append(dict(
                seed=s, median=med, signal=sig, oracle=oracle, const=const,
                n_mem=n_mem, n_query=n_qry))

            # ---- PQ sweep (codebooks trained on the memory set only)
            for m in args.pq_m:
                if D % m != 0:
                    continue
                pq_rng = np.random.RandomState(7000 + 100 * s + m)
                books = pq_fit(Mdesc, m, pq_rng, iters=args.kmeans_iters)
                Mrec = pq_reconstruct(Mdesc, books)
                med, sig = score(knn_answer(Mrec, Mxy, Qdesc))
                per_system.setdefault(f"pq{m}", []).append(dict(
                    seed=s, median=med, signal=sig, oracle=oracle,
                    const=const, n_mem=n_mem, n_query=n_qry))
            print(f"  [{tname}] seed {s} done "
                  f"(n_mem={n_mem}, n_qry={n_qry}, "
                  f"oracle={oracle:.2f}, const={const:.2f})")

        # ---- aggregate + byte accounting, save incrementally per treatment
        for sysname, rows in per_system.items():
            sig = np.array([r["signal"] for r in rows])
            med = np.array([r["median"] for r in rows])
            nm = int(np.mean([r["n_mem"] for r in rows]))
            pose_bytes = nm * 2 * 4
            if sysname == "exact_f32":
                desc_bytes, codebook_bytes = nm * D * 4, 0
            else:
                m = int(sysname[2:])
                desc_bytes = nm * m * 1
                codebook_bytes = 256 * D * 4
            entry = dict(
                task="blocked heldout place recall (heldout_eval definition)",
                encoder=args.encoder_name, treatment=tname, system=sysname,
                signal_mean=float(sig.mean()), signal_std=float(sig.std()),
                median_mean=float(med.mean()), median_std=float(med.std()),
                oracle=float(np.mean([r["oracle"] for r in rows])),
                constant=float(np.mean([r["const"] for r in rows])),
                n_mem=nm,
                n_query=int(np.mean([r["n_query"] for r in rows])),
                n_seeds=len(rows), seeds=[r["seed"] for r in rows],
                desc_bytes=desc_bytes, pose_bytes=pose_bytes,
                codebook_bytes=codebook_bytes,
                total_bytes=desc_bytes + pose_bytes + codebook_bytes,
                per_seed=rows)
            results.append(entry)
            fn = os.path.join(args.result_dir,
                              f"knn_{args.encoder_name}_{tname}_{sysname}.json")
            with open(fn, "w") as f:
                json.dump(entry, f, indent=2)

    # ---- final combined artifact
    out = os.path.join(args.result_dir,
                       f"knn_baseline_{args.encoder_name}.json")
    with open(out, "w") as f:
        json.dump(dict(
            note="kNN / PQ baselines on the IDENTICAL blocked held-out task; "
                 "VSA comparison rows are quoted from "
                 "outputs/classroom/heldout_eval_5seed.json entry (ai), "
                 "NOT re-run here.",
            vsa_comparison=dict(
                trace_bytes=4096 * 16,
                decode_grid_bytes_as_run=48 * 48 * 4096 * 8,
                position_encoder_bytes=2 * 4096 * 4,
                source="outputs/classroom/heldout_eval_5seed.json"),
            results=results), f, indent=2)
    print(f"\nwrote {out}")

    # ---- console table
    hdr = (f"{'treatment':<10}{'system':<11}{'sig_mean':>9}{'sig_sd':>8}"
           f"{'median':>8}{'bytes(desc+pose)':>18}{'+codebook':>11}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in results:
        print(f"{r['treatment']:<10}{r['system']:<11}"
              f"{r['signal_mean']:>9.1%}{r['signal_std']:>8.1%}"
              f"{r['median_mean']:>8.3f}"
              f"{r['desc_bytes'] + r['pose_bytes']:>18,}"
              f"{r['total_bytes']:>11,}")
    print("\nVSA (ai) quoted rows, hd=4096 trace = 65,536 B "
          "(+ decode grid 75,497,472 B as run; grid derivable from 32,768 B "
          "of encoder phases):")
    print("  dinov2 raw      31.7% +- 15.4%   median 2.295 m")
    print("  dinov2 zscored  71.6% +-  7.5%   median 1.142 m")


if __name__ == "__main__":
    main()
