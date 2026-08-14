"""Cross-traverse recall: memory from some walks, queries from a DIFFERENT walk.

This is the benchmark protocol (7-Scenes-style): the memory stores frames
from the official train traverses; every query comes from a physically
separate test traverse. No blocked splits, no eviction gap, no split-seed
caveats — the held-out-ness is real.

Protocol hardening vs the single-sequence tools: descriptor treatment
statistics (mean/std/whitening basis) are fit on the MEMORY traverses only
and applied frozen to queries. The single-sequence tools fit them on all
frames (documented as acceptable there); a benchmark reviewer would call
that a leak, and memory-only stats are also the online-deployable choice.

Reported per system, always with bytes: 2D floor-plane error (what the VSA
grid decodes) and — when the pose CSVs carry tx/ty/tz — full 3D translation
error of the RETRIEVED frame vs the query's ground truth. The 3D number is
the one comparable to published retrieval baselines (DenseVLAD ~0.21 m
median on chess); never put the 2D number next to those.

    python -m vsa_cognitive_mapping.cross_recall \
        --mem-datasets vsa_cognitive_mapping/configs/7scenes_chess_seq01.json ... \
        --query-datasets vsa_cognitive_mapping/configs/7scenes_chess_seq03.json ... \
        --encoder crop_embeddings_eigenplaces.pt --merge-check
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vsa_cognitive_mapping.sequences import load_sequence  # noqa: E402
from vsa_cognitive_mapping.heldout_eval import frame_descriptors  # noqa: E402
from vsa_cognitive_mapping.classroom_pipeline import ClassroomEncoders  # noqa: E402
from vsa_cognitive_mapping.vsa import random_project_to_phasor  # noqa: E402
from vsa_cognitive_mapping.knn_baseline import pq_fit, pq_reconstruct, _unit  # noqa: E402
from vsa_cognitive_mapping.recall_cdf import topk_err  # noqa: E402


# ------------------------------------------------------------ traverse load

def load_traverse(cfg_path, encoder):
    seq = load_sequence(cfg_path)
    uf, E = frame_descriptors(seq.out_dir, encoder)
    p = seq.poses()
    if p is None:
        raise SystemExit(f"{cfg_path}: no poses")
    fids = seq.frame_ids()
    pose_of = {int(f): (p[0][i], p[1][i], p[2][i]) for i, f in enumerate(fids)}
    keep = np.array([int(f) in pose_of for f in uf])
    uf, E = uf[keep], E[keep]
    XYP = np.array([pose_of[int(f)] for f in uf])

    # full 3D translation, if the pose CSV carries it (prepare_7scenes does)
    T3 = None
    cfg = json.load(open(cfg_path)) if isinstance(cfg_path, str) else cfg_path
    spec = cfg.get("poses", {})
    if spec.get("kind") == "csv":
        with open(spec["path"], newline="") as f:
            rd = csv.DictReader(f)
            if rd.fieldnames and {"tx", "ty", "tz", "frame"} <= set(rd.fieldnames):
                t3_of = {int(float(r["frame"])):
                         (float(r["tx"]), float(r["ty"]), float(r["tz"]))
                         for r in rd}
                T3 = np.array([t3_of[int(f)] for f in uf])
    return dict(name=seq.name, uf=uf, E=E, XY=XYP[:, :2], PSI=XYP[:, 2], T3=T3)


def concat(travs):
    return (np.concatenate([t["E"] for t in travs]),
            np.concatenate([t["XY"] for t in travs]),
            np.concatenate([t["PSI"] for t in travs]),
            (np.concatenate([t["T3"] for t in travs])
             if all(t["T3"] is not None for t in travs) else None),
            np.concatenate([np.full(len(t["uf"]), i)
                            for i, t in enumerate(travs)]))


# ---------------------------------------- memory-only treatment statistics

def fit_treatment(Xmem, name):
    prm = dict(name=name, mu=Xmem.mean(0))
    if name == "zscored":
        prm["sd"] = Xmem.std(0) + 1e-8
    elif name.startswith("whitened"):
        k = int(name.replace("whitened", ""))
        Xc = Xmem - prm["mu"]
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        k = min(k, Vt.shape[0])
        prm["W"] = Vt[:k].T / (S[:k] / np.sqrt(len(Xmem) - 1) + 1e-8)
    return prm


def apply_treatment(X, prm):
    n = prm["name"]
    if n == "raw":
        return X
    if n == "centred":
        return X - prm["mu"]
    if n == "zscored":
        return (X - prm["mu"]) / prm["sd"]
    return (X - prm["mu"]) @ prm["W"]


# ------------------------------------------------------------------ metrics

def cdf_row(e):
    return dict(median=float(np.median(e)),
                le025=float(np.mean(e <= 0.25)), le05=float(np.mean(e <= 0.5)),
                le1=float(np.mean(e <= 1)), worst=float(e.max()))


def fmt(name, e, nbytes=None):
    r = cdf_row(e)
    b = f"{nbytes / 1024:>9,.0f} KB" if nbytes else " " * 12
    return (f"{name:<34}{b}  median {r['median']:5.2f} m | <=0.25m "
            f"{r['le025']:5.1%} | <=0.5m {r['le05']:5.1%} | <=1m "
            f"{r['le1']:5.1%} | worst {r['worst']:4.1f}")


def err3d(T3m, T3q, ans_idx):
    return np.linalg.norm(T3m[ans_idx] - T3q, axis=1)


# --------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mem-datasets", nargs="+", required=True)
    ap.add_argument("--query-datasets", nargs="+", required=True)
    ap.add_argument("--encoder", default="crop_embeddings_eigenplaces.pt")
    ap.add_argument("--treatments", nargs="+",
                    default=["raw", "centred", "zscored", "whitened128"])
    ap.add_argument("--primary", default="zscored",
                    help="treatment used for the PQ rows and merge check")
    ap.add_argument("--hd-dim", type=int, default=4096)
    ap.add_argument("--grid", type=int, default=48)
    ap.add_argument("--length-scale", type=float, default=0.75)
    ap.add_argument("--bind-yaw", action="store_true")
    ap.add_argument("--merge-check", action="store_true")
    ap.add_argument("--pq-m", type=int, nargs="+", default=[16, 8, 4])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    mem_t = [load_traverse(c, args.encoder) for c in args.mem_datasets]
    qry_t = [load_traverse(c, args.encoder) for c in args.query_datasets]
    Em, XYm, PSIm, T3m, src_m = concat(mem_t)
    Eq, XYq, PSIq, T3q, _ = concat(qry_t)
    n_mem, n_qry, D = len(Em), len(Eq), Em.shape[1]
    have3d = T3m is not None and T3q is not None

    dxy = np.hypot(XYq[:, 0][:, None] - XYm[None, :, 0],
                   XYq[:, 1][:, None] - XYm[None, :, 1])
    oracle = float(np.median(dxy.min(1)))
    cen = XYm.mean(0)
    const = float(np.median(np.hypot(*(XYq - cen).T)))

    print(f"memory: {n_mem} frames from {len(mem_t)} traverse(s) "
          f"({', '.join(t['name'] for t in mem_t)})")
    print(f"queries: {n_qry} frames from {len(qry_t)} SEPARATE traverse(s) "
          f"({', '.join(t['name'] for t in qry_t)})")
    print(f"2D brackets: oracle {oracle:.3f} m, constant-predictor {const:.3f} m")
    print("treatment statistics: fit on MEMORY traverses only, frozen for "
          "queries (no query leak)\n")

    ext = [min(XYm[:, 0].min(), XYq[:, 0].min()) - 1,
           max(XYm[:, 0].max(), XYq[:, 0].max()) + 1,
           min(XYm[:, 1].min(), XYq[:, 1].min()) - 1,
           max(XYm[:, 1].max(), XYq[:, 1].max()) + 1]
    enc = ClassroomEncoders(args.hd_dim, 0, args.length_scale, 20.0)
    gx = np.linspace(ext[0], ext[1], args.grid)
    gy = np.linspace(ext[2], ext[3], args.grid)
    G = np.empty((args.grid ** 2, args.hd_dim), np.complex64)
    k = 0
    for yy in gy:
        for xx in gx:
            G[k] = enc.ctx_pos(float(xx), float(yy)).values.astype(np.complex64)
            k += 1
    P = np.stack([enc.ctx_pos(float(a), float(b)).values for a, b in XYm])
    Hm = Hq = None
    if args.bind_yaw:
        Hm = np.stack([enc.ctx_head(float(a)).values for a in PSIm])
        Hq = np.stack([enc.ctx_head(float(a)).values for a in PSIq])

    vsa_bytes = args.hd_dim * 8            # one complex64 trace; bases from seed
    knn_bytes = n_mem * D * 4 + n_mem * 2 * 4
    results = dict(mem=[t["name"] for t in mem_t],
                   qry=[t["name"] for t in qry_t],
                   n_mem=n_mem, n_qry=n_qry, encoder=args.encoder,
                   bind_yaw=args.bind_yaw, oracle_2d=oracle, const_2d=const,
                   rows=[])

    def emit(system, treatment, e2, e3, nbytes):
        row = dict(system=system, treatment=treatment, bytes=int(nbytes),
                   err2d=cdf_row(e2), err3d=cdf_row(e3) if e3 is not None else None)
        results["rows"].append(row)
        tag = f"{system} · {treatment}"
        print(fmt(tag + "  [2D]", e2, nbytes))
        if e3 is not None:
            print(fmt(tag + "  [3D]", e3))

    Zq_primary = Zm_primary = None
    for tr in args.treatments:
        prm = fit_treatment(Em, tr)
        Xm, Xq = apply_treatment(Em, prm), apply_treatment(Eq, prm)

        # kNN ceiling on the same (memory-stat-treated) descriptors
        nn = (_unit(Xq) @ _unit(Xm).T).argmax(1)
        e2 = np.hypot(*(XYm[nn] - XYq).T)
        emit("kNN exact", tr, e2, err3d(T3m, T3q, nn) if have3d else None,
             knn_bytes)

        # VSA memory
        z, _ = random_project_to_phasor(
            torch.from_numpy(np.ascontiguousarray(
                np.concatenate([Xm, Xq]))).float(), d=args.hd_dim, seed=0)
        Z = z.numpy().astype(np.complex128)
        Zm, Zq = Z[:n_mem], Z[n_mem:]
        M = (Zm * P * Hm).mean(0) if args.bind_yaw else (Zm * P).mean(0)
        M /= max(np.abs(M).max(), 1e-12)
        R = M[None, :] / Zq
        if args.bind_yaw:
            R = R / Hq
        S = (R @ np.conj(G).T).real
        j = S.argmax(1)
        pk = np.stack([gx[j % args.grid], gy[j // args.grid]], 1)
        e2 = np.hypot(*(pk - XYq).T)
        # decoded cell -> nearest memory frame -> that frame's full pose
        near = np.array([np.argmin(np.hypot(*(XYm - p).T)) for p in pk])
        emit("VSA argmax", tr, e2,
             err3d(T3m, T3q, near) if have3d else None, vsa_bytes)
        e2t = topk_err(S, XYq, gx, gy, args.grid)
        emit("VSA top-3 modes", tr, e2t, None, vsa_bytes)
        if tr == args.primary:
            Zm_primary, Zq_primary = Zm, Zq
            Xm_primary, Xq_primary = Xm, Xq
        print()

    # ---- PQ at matched bytes (primary treatment) -------------------------
    rng = np.random.RandomState(0)
    for m in args.pq_m:
        if D % m:
            print(f"pq{m}: skipped (D={D} not divisible)")
            continue
        books = pq_fit(Xm_primary, m, rng)
        Rm = pq_reconstruct(Xm_primary, books)
        nn = (_unit(Xq_primary) @ _unit(Rm).T).argmax(1)
        e2 = np.hypot(*(XYm[nn] - XYq).T)
        nbytes = n_mem * m + 256 * D * 4 + n_mem * 2 * 4
        emit(f"PQ m={m} (codes+codebook+poses)", args.primary, e2,
             err3d(T3m, T3q, nn) if have3d else None, nbytes)
    print(f"\n(VSA trace = {vsa_bytes / 1024:.0f} KB regardless of N; "
          f"kNN = {knn_bytes / 1e6:.1f} MB at N={n_mem})")

    # ---- multi-session merge --------------------------------------------
    if args.merge_check and len(mem_t) > 1:
        print("\n=== multi-session merge (one memory per traverse, added) ===")
        parts = []
        for i in range(len(mem_t)):
            sel = src_m == i
            Zi = Zm_primary[sel]
            Pi = P[sel]
            parts.append((Zi * Pi).sum(0) if not args.bind_yaw
                         else (Zi * Pi * Hm[sel]).sum(0))
        M_merged = sum(parts) / n_mem
        M_joint = ((Zm_primary * P * Hm).sum(0) if args.bind_yaw
                   else (Zm_primary * P).sum(0)) / n_mem
        exact = float(np.abs(M_merged - M_joint).max())
        Mn = M_merged / max(np.abs(M_merged).max(), 1e-12)
        R = Mn[None, :] / Zq_primary
        if args.bind_yaw:
            R = R / Hq
        S2 = (R @ np.conj(G).T).real
        Mj = M_joint / max(np.abs(M_joint).max(), 1e-12)
        Rj = Mj[None, :] / Zq_primary
        if args.bind_yaw:
            Rj = Rj / Hq
        Sj = (Rj @ np.conj(G).T).real
        same = int((S2.argmax(1) == Sj.argmax(1)).sum())
        print(f"{len(mem_t)} sessions ({[int((src_m == i).sum()) for i in range(len(mem_t))]} "
              f"frames each): max |merged − joint| = {exact:.2e}; "
              f"identical answers on the full test battery: {same}/{n_qry}")
        results["merge"] = dict(exact=exact, identical=same, n=n_qry)

    out = args.out or os.path.join(
        "outputs", f"cross_recall_{mem_t[0]['name'].rsplit('_', 1)[0]}"
                   f"{'_yaw' if args.bind_yaw else ''}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
