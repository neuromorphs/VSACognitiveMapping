"""Held-out recall: query the memory with frames it never stored.

Nothing in this project has been held out. Every replayed frame was already in
the memory, which measures faithful storage rather than generalisation -- and
that is not a pedantic complaint: an UNTRAINED ResNet-50 matched DINOv2's median
recall under closed-set scoring, because storing "arbitrary vector -> pose" and
querying with that same arbitrary vector works whether or not the vector means
anything.

What the memory does
--------------------
It does not estimate pose. Robot pose comes from LIO-SAM (LiDAR + IMU); object
positions come from D455 depth deprojected through that pose. The VSA is handed
a pose and stores appearance bound to it, so a query returns *the pose stored
with the most similar stored appearance*. This evaluation therefore measures
"can an unseen view recall a sensible place", not visual localisation.

Split regimes -- the point of this script
-----------------------------------------
``random``      Uniformly random held-out frames. **Leaks.** At ~15 fps the
                neighbours of a held-out frame are still in the memory, and
                near-in-time frames stay highly similar even after whitening
                (|cos| ~0.29 at <4 s apart against a 0.011 random floor). Kept
                precisely so the leak is visible next to the honest numbers.
``blocked``     Random held-out frames, and every memory frame within
                ``--gap`` frames of a held-out one is evicted. No near-duplicate
                of a query remains stored. This is the honest random split.
``contiguous``  The last fraction of the run held out in one block. Tests
                extrapolation to a part of the walk never seen.

Every number is bracketed:

  ORACLE  distance from the query pose to the NEAREST stored frame's pose --
          the best any descriptor could do, since the memory can only return a
          pose it holds
  CHANCE  median distance from the query pose to a uniformly random stored pose

and reported as ``signal`` = (chance - measured) / (chance - oracle): 0% is
random, 100% is the ceiling. Raw metres are meaningless in a 6.9 m room.

    python -m vsa_cognitive_mapping.heldout_eval --out-dir outputs/classroom
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vsa_cognitive_mapping.vsa import pca_components, random_project_to_phasor  # noqa: E402
from vsa_cognitive_mapping.classroom_pipeline import (  # noqa: E402
    ClassroomEncoders, load_poses_interpolated)

ENCODERS = [("yolov8n", "crop_embeddings.pt"),
            ("resnet50", "crop_embeddings_resnet50.pt"),
            ("dinov2", "crop_embeddings_dinov2.pt"),
            ("untrained", "crop_embeddings_resnet50-untrained.pt")]

# extra names resolvable via --encoders (not part of the default run)
KNOWN_ENCODERS = dict(ENCODERS, sigreg="crop_embeddings_sigreg.pt")

TREATMENTS = ["raw", "centred", "zscored", "whitened128"]


def frame_descriptors(out_dir, fname):
    """Per-frame descriptor = mean of that frame's crop embeddings, L2-normed."""
    d = torch.load(os.path.join(out_dir, fname), weights_only=False)
    X = d["embedding"].numpy().astype(np.float64)
    fr = d["frame_idx"].numpy()
    uf, inv = np.unique(fr, return_inverse=True)
    D = np.zeros((len(uf), X.shape[1]))
    np.add.at(D, inv, X)
    D /= np.maximum(np.bincount(inv, minlength=len(uf))[:, None], 1)
    D /= (np.linalg.norm(D, axis=1, keepdims=True) + 1e-12)
    return uf, D


def treat(E, kind):
    if kind == "raw":
        return E
    if kind == "centred":
        return E - E.mean(axis=0, keepdims=True)
    if kind == "zscored":
        c = E - E.mean(axis=0, keepdims=True)
        return c / (c.std(axis=0, keepdims=True) + 1e-9)
    if kind.startswith("whitened"):
        k = int(kind.replace("whitened", ""))
        s, _ = pca_components(E, n_components=min(k, E.shape[1], E.shape[0] - 1))
        return s
    raise ValueError(kind)


def make_split(fid, mode, frac, gap, rng, n_blocks=8):
    """-> (memory_mask, query_mask). Disjoint; memory may be shrunk by eviction."""
    n = len(fid)
    if mode == "contiguous":
        cut = int(round((1 - frac) * n))
        q = np.zeros(n, bool); q[cut:] = True
        return ~q, q
    if mode == "random":
        q = np.zeros(n, bool)
        q[rng.choice(n, size=int(round(frac * n)), replace=False)] = True
        return ~q, q
    if mode == "blocked":
        # Contiguous SEGMENTS, not scattered frames. Scattering 30% of frames
        # uniformly leaves an average spacing of ~3 frames, so any eviction
        # radius worth having deletes the whole memory -- the first attempt at
        # this produced an empty table. Blocks keep the memory intact while
        # still guaranteeing no near-duplicate of a query remains stored.
        seg = max(1, int(round(frac * n / n_blocks)))
        q = np.zeros(n, bool)
        starts = []
        for _ in range(n_blocks * 20):
            if sum(q) >= frac * n:
                break
            s = rng.randint(0, max(1, n - seg))
            if any(abs(s - p) < seg + 2 * gap for p in starts):
                continue
            starts.append(s); q[s:s + seg] = True
        qf = fid[q]
        close = np.abs(fid[:, None] - qf[None, :]).min(axis=1) <= gap
        return (~q) & (~close), q
    raise ValueError(mode)


def evaluate(Z, XY, mem, qry, enc, grid, extent, hd,
             psi=None, bind_yaw=False, Z_mem_override=None):
    """Build the memory over `mem`, decode position for every `qry` frame.

    Optional variants (both default OFF; the default path is unchanged):
      bind_yaw=True     store content (x) ctx_pos (x) ctx_head(psi_i); at query
                        time unbind BOTH the content and ctx_head(psi_q) —
                        psi_q comes from the pose stream, i.e. yaw is treated
                        as KNOWN at query time — then decode position only on
                        the same (x, y) grid. One extra elementwise multiply
                        per stored/query frame; no joint (x, y, psi) decode.
      Z_mem_override    (n_mem, hd) complex: replaces the stored frames'
                        content phasors (used for place signatures). Queries
                        always use their own single-view content Z[qry].
    """
    gx = np.linspace(extent[0], extent[1], grid)
    gy = np.linspace(extent[2], extent[3], grid)
    G = np.empty((grid * grid, hd), np.complex64)
    k = 0
    for yy in gy:
        for xx in gx:
            G[k] = enc.ctx_pos(float(xx), float(yy)).values.astype(np.complex64); k += 1

    Zm = Z[mem] if Z_mem_override is None else Z_mem_override
    P = np.stack([enc.ctx_pos(float(x), float(y)).values for x, y in XY[mem]])
    if bind_yaw:
        H = np.stack([enc.ctx_head(float(a)).values for a in psi[mem]])
        M = (Zm * P * H).mean(axis=0)
    else:
        M = (Zm * P).mean(axis=0)
    M /= max(np.abs(M).max(), 1e-12)

    R = M[None, :] / Z[qry]                      # unbind each query's content
    if bind_yaw:                                 # ...and its (known) heading
        Hq = np.stack([enc.ctx_head(float(a)).values for a in psi[qry]])
        R = R / Hq
    S = (R @ np.conj(G).T).real                  # (n_query, cells)
    j = S.argmax(1)
    peak = np.stack([gx[j % grid], gy[j // grid]], axis=1)
    err = np.hypot(peak[:, 0] - XY[qry, 0], peak[:, 1] - XY[qry, 1])
    # How many distinct answers did it actually give? A representation whose
    # vectors are nearly parallel leaves the same residual for every query and
    # decodes to one cell -- a constant predictor that still scores well against
    # a random-frame baseline. `spread` catches that: 1 unique cell = constant.
    return err, len(np.unique(j)), float(np.mean(np.std(peak, axis=0)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="outputs/classroom")
    ap.add_argument("--odom-config", default="odometry_lio_sam")
    ap.add_argument("--dataset", default=None,
                    help="sequence config JSON (sequences.py). When given, poses "
                         "come from the sequence (TUM/CSV readers included) "
                         "instead of the HuggingFace odometry, after the pose "
                         "sanity check; --out-dir defaults to the config's. This "
                         "is what lets the through-memory held-out test run on a "
                         "second environment.")
    ap.add_argument("--modes", nargs="+", default=["random", "blocked", "contiguous"])
    ap.add_argument("--treatments", nargs="+", default=TREATMENTS)
    ap.add_argument("--frac", type=float, default=0.3, help="held-out fraction")
    ap.add_argument("--gap", type=int, default=45,
                    help="blocked mode: evict stored frames within this many "
                         "frames of any query (45 ~ 3 s at 15 fps)")
    ap.add_argument("--hd-dim", type=int, default=4096)
    ap.add_argument("--grid", type=int, default=48)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--bind-yaw", action="store_true",
                    help="store content (x) ctx_pos(x,y) (x) ctx_head(psi); "
                         "at query time unbind content AND ctx_head(psi_q) "
                         "(yaw is taken from the pose stream, i.e. KNOWN at "
                         "query time), then decode position on the same grid")
    ap.add_argument("--place-radius", type=float, default=None,
                    help="replace each stored frame's content with a PLACE "
                         "SIGNATURE: the L2-normalised mean of the (treated) "
                         "descriptors of all MEMORY frames within this many "
                         "metres of its pose (any heading), bound to ctx_pos "
                         "only. Queries keep their single-view descriptor. "
                         "No heading factor at all.")
    ap.add_argument("--encoders", nargs="+", default=None,
                    help="subset/extend the encoder list: known names "
                         f"({', '.join(KNOWN_ENCODERS)}) or name=file pairs "
                         "(file relative to --out-dir). Default: the standard "
                         "four-encoder list.")
    args = ap.parse_args()

    if args.bind_yaw and args.place_radius is not None:
        ap.error("--bind-yaw and --place-radius are alternative remedies; "
                 "run them separately to compare against the baseline")

    if args.encoders:
        pairs = []
        for tok in args.encoders:
            if "=" in tok:
                n, f = tok.split("=", 1)
            elif tok in KNOWN_ENCODERS:
                n, f = tok, KNOWN_ENCODERS[tok]
            else:
                ap.error(f"unknown encoder '{tok}'; use one of "
                         f"{sorted(KNOWN_ENCODERS)} or name=file")
            pairs.append((n, f))
    else:
        pairs = ENCODERS

    seq = None
    if args.dataset:
        from vsa_cognitive_mapping.sequences import load_sequence
        seq = load_sequence(args.dataset)
        if args.out_dir == "outputs/classroom":
            args.out_dir = seq.out_dir
        q = seq.pose_quality()
        if q is None:
            raise SystemExit("this dataset has no poses; the held-out place "
                             "metric cannot run without them")
        if not q.usable:
            raise SystemExit("pose stream rejected:\n" + q.report())

    have = [(n, f) for n, f in pairs
            if os.path.exists(os.path.join(args.out_dir, f))]
    uf0, _ = frame_descriptors(args.out_dir, have[0][1])
    # Timestamps from the FIRST encoder file itself, not crop_embeddings.pt:
    # frame-level frontends (e.g. VPR) cover detection-free frames that the
    # crop file doesn't, which made this a KeyError (same fix as time_recall).
    base = torch.load(os.path.join(args.out_dir, have[0][1]),
                      weights_only=False)
    ts_of = {}
    for f, t in zip(base["frame_idx"].numpy(), base["timestamp_ns"].numpy()):
        ts_of.setdefault(int(f), int(t))
    if seq is not None:
        px, py, pyaw = seq.poses()
        pose_of = {int(fi): (px[i], py[i], pyaw[i])
                   for i, fi in enumerate(seq.frame_ids())}
        missing = [int(f) for f in uf0 if int(f) not in pose_of]
        if missing:
            raise SystemExit(f"{len(missing)} crop frames have no pose in the "
                             f"sequence (first: {missing[:5]}) — frame-id "
                             f"mismatch between the crop file and the config")
        XYP = np.array([pose_of[int(f)] for f in uf0])
        XY, PSI = XYP[:, :2], XYP[:, 2]
    else:
        ts = np.array([ts_of[int(f)] for f in uf0], np.int64)
        x, y, psi = load_poses_interpolated(ts, odom_config=args.odom_config)
        XY = np.stack([x, y], axis=1)
        PSI = np.asarray(psi)
    x, y = XY[:, 0], XY[:, 1]        # extent from whichever pose source was used
    ext = [x.min() - 1.0, x.max() + 1.0, y.min() - 1.0, y.max() + 1.0]
    encs = ClassroomEncoders(args.hd_dim, 0, 0.75, 20.0)
    if args.bind_yaw:
        variant = "bind_yaw"
    elif args.place_radius is not None:
        variant = f"place_r{args.place_radius:g}"
    else:
        variant = "baseline"
    print(f"{len(uf0)} frames, room {np.ptp(x):.2f} x {np.ptp(y):.2f} m, "
          f"hd={args.hd_dim}, grid={args.grid}, held out {args.frac:.0%}, "
          f"variant={variant}")
    if args.bind_yaw:
        print("NOTE: --bind-yaw stores content (x) ctx_pos (x) ctx_head(psi); "
              "the query unbinds ctx_head(psi_q) with psi_q taken from the "
              "pose stream — yaw is KNOWN at query time; only position is "
              "decoded.")
    if args.place_radius is not None:
        print(f"NOTE: --place-radius {args.place_radius:g} m: each stored "
              "frame's content is the L2-normalised mean of the treated "
              "descriptors of all MEMORY frames within that radius (any "
              "heading), bound to ctx_pos only; queries keep their "
              "single-view descriptor.")
    print()

    D_true = np.hypot(XY[:, 0][:, None] - XY[:, 0][None, :],
                      XY[:, 1][:, None] - XY[:, 1][None, :])

    results = []
    for mode in args.modes:
        print(f"===== split: {mode} =====")
        hdr = (f"{'encoder':<11}{'treatment':<13}{'n_mem':>7}{'n_qry':>7}"
               f"{'oracle':>7}{'const':>7}{'median':>8}{'med_sd':>8}{'signal':>8}"
               f"{'sig_sd':>8}{'cells':>7}{'spread':>8}")
        print(hdr); print("-" * len(hdr))
        for enc_name, fn in have:
            uf, E = frame_descriptors(args.out_dir, fn)
            for tname in args.treatments:
                Et = treat(E, tname)
                z, _ = random_project_to_phasor(
                    torch.from_numpy(np.ascontiguousarray(Et)).float(),
                    d=args.hd_dim, seed=0)
                Z = z.numpy().astype(np.complex128)
                meds, sigs, nm, nq, cells, spreads = [], [], [], [], [], []
                for s in args.seeds:
                    rng = np.random.RandomState(1000 + s)
                    mem, qry = make_split(uf, mode, args.frac, args.gap, rng)
                    if mem.sum() < 20 or qry.sum() < 20:
                        continue
                    Zov = None
                    if args.place_radius is not None:
                        # Place signature per STORED frame: mean of the treated
                        # descriptors of memory frames within R metres (any
                        # heading; each frame is its own neighbour so counts
                        # are >= 1), L2-normalised, then projected with the
                        # SAME seed-0 matrix as the queries.
                        mi = np.flatnonzero(mem)
                        A = (D_true[np.ix_(mi, mi)] <= args.place_radius
                             ).astype(np.float64)
                        Sig = A @ Et[mi]
                        Sig /= A.sum(axis=1, keepdims=True)
                        Sig /= (np.linalg.norm(Sig, axis=1, keepdims=True)
                                + 1e-12)
                        zp, _ = random_project_to_phasor(
                            torch.from_numpy(np.ascontiguousarray(Sig)).float(),
                            d=args.hd_dim, seed=0)
                        Zov = zp.numpy().astype(np.complex128)
                    d, nuniq, spr = evaluate(Z, XY, mem, qry, encs,
                                             args.grid, ext, args.hd_dim,
                                             psi=PSI, bind_yaw=args.bind_yaw,
                                             Z_mem_override=Zov)
                    sub = D_true[np.ix_(qry, mem)]
                    oracle = np.median(sub.min(axis=1))
                    # The honest floor is not "a random stored frame" but "ignore
                    # the query and always answer the centroid of stored poses" --
                    # a constant predictor beats random in a small room, so
                    # scoring against random flatters degenerate representations.
                    cen = XY[mem].mean(axis=0)
                    const = float(np.median(np.hypot(XY[qry, 0] - cen[0],
                                                     XY[qry, 1] - cen[1])))
                    med = float(np.median(d))
                    meds.append(med); nm.append(mem.sum()); nq.append(qry.sum())
                    cells.append(nuniq); spreads.append(spr)
                    sigs.append((const - med) / max(const - oracle, 1e-9))
                    last = (oracle, const)
                    if mode == "contiguous":
                        break                     # deterministic split
                if not meds:
                    continue
                print(f"{enc_name:<11}{tname:<13}{int(np.mean(nm)):>7}{int(np.mean(nq)):>7}"
                      f"{last[0]:>7.2f}{last[1]:>7.2f}{np.mean(meds):>8.3f}"
                      f"{np.std(meds):>8.3f}{np.mean(sigs):>8.1%}"
                      f"{np.std(sigs):>8.1%}{int(np.mean(cells)):>7}"
                      f"{np.mean(spreads):>8.2f}")
                extra = {} if variant == "baseline" else {"variant": variant}
                results.append(dict(mode=mode, encoder=enc_name, treatment=tname,
                                    **extra,
                                    median=float(np.mean(meds)),
                                    median_std=float(np.std(meds)),
                                    signal_vs_constant=float(np.mean(sigs)),
                                    signal_std=float(np.std(sigs)),
                                    oracle=float(last[0]), constant=float(last[1]),
                                    distinct_cells=int(np.mean(cells)),
                                    peak_spread_m=float(np.mean(spreads)),
                                    n_mem=int(np.mean(nm)), n_query=int(np.mean(nq))))
        print()

    out = os.path.join(args.out_dir, "heldout_eval.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out}")
    print("\nsignal = (const - median) / (const - oracle).  0% = no better than ignoring")
    print("the query and answering the centroid of stored poses; 100% = the ceiling.")
    print("NEGATIVE means worse than that constant predictor.")
    print("oracle = nearest STORED pose, so it rises as 'blocked' evicts neighbours.")
    print("cells  = distinct decoded grid cells over all queries; 1 means the")
    print("         representation is degenerate and answers the same place every time.")


if __name__ == "__main__":
    main()
