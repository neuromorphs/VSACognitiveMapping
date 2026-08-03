"""A Kalman-style filter that runs entirely in phasor space.

The decoded position jumps between frames in a way the robot physically cannot.
Each frame is currently answered independently, so nothing enforces continuity.
A filter fixes that -- and the FPE encoding makes both filter steps native, so
neither requires decoding:

**Predict.** ``ctx_pos`` is a group homomorphism, ``S(a) (x) S(b) = S(a+b)``, so
moving the belief by an odometry delta is one bind::

    S_pred = S_prev  (x)  ctx_pos(dx, dy)

exact, O(D), no grid involved.

**Update.** Bundling is a weighted average in phasor space, so blending the
prediction with the measurement is one superposition::

    S_post = normalise( a * S_meas + (1 - a) * S_pred )

`a` is the Kalman gain. The grid decode is needed only to *display* a position,
not to run the filter.

Choosing the gain
-----------------
``--gain fixed`` uses a constant `a` (what the upstream ``--kalman-weight``
does). ``--gain adaptive`` sets it per frame from how peaked the measurement's
similarity field is, ``z = (peak - mean) / std`` over the grid, mapped through
a logistic. A confident recall pulls hard; a flat, ambiguous field is mostly
ignored and the prediction carries. That reuses the same evidence signal the
abstention machinery is built on.

Honest odometry
---------------
Propagating with the *true* pose deltas would make dead reckoning exact and the
filter trivially perfect -- using the answer to smooth the answer. So the
odometry is corrupted with per-step Gaussian noise plus a small systematic bias
(``--odom-noise``, ``--odom-bias``), giving a track that drifts the way real
odometry does. The question then is the real one: **does associative recall pull
a drifting track back?**

This is also consistent with what the system actually is. Robot pose already
comes from LIO-SAM, and the VSA is a memory rather than a localiser, so
"relative motion from odometry, absolute correction from memory" is the
architecture, not a shortcut.

    python -m vsa_cognitive_mapping.vsa_kalman --out-dir outputs/classroom
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
from vsa_cognitive_mapping.heldout_eval import frame_descriptors, treat, make_split  # noqa: E402


def normalise(v):
    """Back onto the unit circle per component -- the phasor manifold."""
    return v / (np.abs(v) + 1e-12)


def decode(S, G, gx, gy, grid):
    s = (np.conj(G) @ S).real
    j = int(np.argmax(s))
    z = (s[j] - s.mean()) / (s.std() + 1e-12)
    return np.array([gx[j % grid], gy[j // grid]]), float(z)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="outputs/classroom")
    ap.add_argument("--odom-config", default="odometry_lio_sam")
    ap.add_argument("--encoder", default="crop_embeddings_dinov2.pt")
    ap.add_argument("--treatment", default="zscored",
                    help="raw | centred | zscored | whitened128 (zscored wins on "
                         "held-out splits -- see heldout_eval)")
    ap.add_argument("--hd-dim", type=int, default=4096)
    ap.add_argument("--grid", type=int, default=48)
    ap.add_argument("--split", default="blocked", choices=["blocked", "contiguous", "all"],
                    help="which frames go INTO the memory; 'all' stores everything")
    ap.add_argument("--frac", type=float, default=0.3)
    ap.add_argument("--gap", type=int, default=45)
    ap.add_argument("--gain", default="adaptive", choices=["fixed", "adaptive"])
    ap.add_argument("--alpha", type=float, default=0.35, help="fixed gain")
    ap.add_argument("--z-mid", type=float, default=6.0, help="adaptive: z at gain 0.5")
    ap.add_argument("--z-slope", type=float, default=0.6)
    ap.add_argument("--odom-noise", type=float, default=0.02,
                    help="per-step Gaussian sigma on each delta component (m)")
    ap.add_argument("--odom-bias", type=float, default=0.004,
                    help="systematic per-step drift added to each delta (m)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sweep", action="store_true",
                    help="scan the gain, fixed and adaptive, before the headline run")
    ap.add_argument("--odom-sweep", action="store_true",
                    help="scan odometry quality to find where recall starts to help")
    args = ap.parse_args()

    uf, E = frame_descriptors(args.out_dir, args.encoder)
    base = torch.load(os.path.join(args.out_dir, "crop_embeddings.pt"), weights_only=False)
    ts_of = {}
    for f, t in zip(base["frame_idx"].numpy(), base["timestamp_ns"].numpy()):
        ts_of.setdefault(int(f), int(t))
    ts = np.array([ts_of[int(f)] for f in uf], np.int64)
    x, y, _ = load_poses_interpolated(ts, odom_config=args.odom_config)
    XY = np.stack([x, y], 1)
    n = len(uf)

    Et = treat(E, args.treatment)
    z, _ = random_project_to_phasor(
        torch.from_numpy(np.ascontiguousarray(Et)).float(), d=args.hd_dim, seed=0)
    Z = z.numpy().astype(np.complex128)

    rng = np.random.RandomState(args.seed)
    if args.split == "all":
        mem = np.ones(n, bool)
    else:
        mem, _q = make_split(uf, args.split, args.frac, args.gap, rng)
    enc = ClassroomEncoders(args.hd_dim, 0, 0.75, 20.0)
    ext = [x.min() - 1.0, x.max() + 1.0, y.min() - 1.0, y.max() + 1.0]
    gx = np.linspace(ext[0], ext[1], args.grid)
    gy = np.linspace(ext[2], ext[3], args.grid)
    G = np.empty((args.grid * args.grid, args.hd_dim), np.complex64)
    k = 0
    for yy in gy:
        for xx in gx:
            G[k] = enc.ctx_pos(float(xx), float(yy)).values.astype(np.complex64); k += 1

    P = np.stack([enc.ctx_pos(float(a), float(b)).values for a, b in XY[mem]])
    M = (Z[mem] * P).mean(axis=0)
    M /= max(np.abs(M).max(), 1e-12)
    print(f"{n} frames, memory holds {mem.sum()} ({args.split}), "
          f"encoder={args.encoder}, treatment={args.treatment}")

    # --- drifting odometry ------------------------------------------------
    d_true = np.diff(XY, axis=0)
    step = np.hypot(d_true[:, 0], d_true[:, 1])[:, None]
    # Dedicated stream: `rng` has already been advanced by make_split, so
    # sharing it made the headline run and the odometry sweep draw different
    # noise for identical parameters (0.350 m vs 0.777 m dead-reckoning median).
    orng = np.random.RandomState(args.seed + 777)
    d_odo = (d_true
             + orng.normal(0, args.odom_noise, d_true.shape)
             + args.odom_bias * step * np.array([[1.0, 0.6]]))
    dead = np.vstack([XY[0], XY[0] + np.cumsum(d_odo, axis=0)])
    print(f"odometry: noise sigma {args.odom_noise} m/step, bias {args.odom_bias} m/m; "
          f"dead-reckoning final drift {np.hypot(*(dead[-1]-XY[-1])):.2f} m")

    # --- per-frame raw measurements --------------------------------------
    R = M[None, :] / Z                                  # unbind every frame at once
    S_all = (R @ np.conj(G).T).real
    j = S_all.argmax(1)
    meas = np.stack([gx[j % args.grid], gy[j // args.grid]], 1)
    zs = (S_all[np.arange(n), j] - S_all.mean(1)) / (S_all.std(1) + 1e-12)

    print(f"measurement confidence z: min {zs.min():.1f}, median {np.median(zs):.1f}, "
          f"p90 {np.percentile(zs,90):.1f}, max {zs.max():.1f}")

    # Motion vectors are the same for every gain setting, so build them once.
    STEP = np.stack([enc.ctx_pos(float(dx), float(dy)).values for dx, dy in d_odo])
    Rn = normalise(R)                       # measurement vectors, unit modulus

    # --- the filter -------------------------------------------------------
    def run_filter(mode, alpha=None, z_mid=None):
        """The recursion is pure O(D) vector algebra -- decoding every step to a
        grid was 99% of the cost and is not needed to RUN the filter, only to
        report it. Collect the posterior track, then decode it in one matmul."""
        S_post = enc.ctx_pos(float(XY[0, 0]), float(XY[0, 1])).values.copy()
        track = np.empty((n, args.hd_dim), np.complex128)
        track[0] = S_post
        gains = np.zeros(n)
        for i in range(1, n):
            S_pred = S_post * STEP[i - 1]                     # homomorphism
            a = alpha if mode == "fixed" else \
                1.0 / (1.0 + np.exp(-args.z_slope * (zs[i] - z_mid)))
            gains[i] = a
            S_post = normalise(a * Rn[i] + (1.0 - a) * S_pred)
            track[i] = S_post
        S = (track.astype(np.complex64) @ np.conj(G).T).real   # (n, cells), batched
        j = S.argmax(1)
        out = np.stack([gx[j % args.grid], gy[j // args.grid]], 1)
        out[0] = XY[0]
        return out, gains

    def err(a):
        return np.hypot(a[:, 0] - XY[:, 0], a[:, 1] - XY[:, 1])

    if args.odom_sweep:
        # The measurement never depends on the odometry, so R/zs are reused and
        # only the motion model changes. This finds the crossover: how bad must
        # dead reckoning get before associative recall is worth blending in?
        print("\nodometry-quality sweep — the filter can only help once dead")
        print("reckoning is worse than the recall it is being corrected by:")
        hdr = (f"{'noise/bias':<18}{'dead med':>10}{'dead final':>11}"
               f"{'best filt':>11}{'at gain':>9}{'helps?':>8}")
        print(hdr); print("-" * len(hdr))
        for nz, bi in ((0.02, 0.004), (0.05, 0.010), (0.10, 0.025),
                       (0.20, 0.050), (0.40, 0.100)):
            rr = np.random.RandomState(args.seed + 777)   # same stream as above
            dd = (d_true + rr.normal(0, nz, d_true.shape)
                  + bi * step * np.array([[1.0, 0.6]]))
            dead_s = np.vstack([XY[0], XY[0] + np.cumsum(dd, axis=0)])
            e_dead_s = np.median(err(dead_s))
            STEP_s = np.stack([enc.ctx_pos(float(a), float(b)).values for a, b in dd])
            best, best_a = None, None
            for a in (0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.60):
                S_post = enc.ctx_pos(float(XY[0, 0]), float(XY[0, 1])).values.copy()
                trk = np.empty((n, args.hd_dim), np.complex128); trk[0] = S_post
                for i in range(1, n):
                    S_post = normalise(a * Rn[i] + (1.0 - a) * (S_post * STEP_s[i - 1]))
                    trk[i] = S_post
                Sx = (trk.astype(np.complex64) @ np.conj(G).T).real
                jx = Sx.argmax(1)
                tr = np.stack([gx[jx % args.grid], gy[jx // args.grid]], 1); tr[0] = XY[0]
                m = float(np.median(err(tr)))
                if best is None or m < best:
                    best, best_a = m, a
            fin = float(np.hypot(*(dead_s[-1] - XY[-1])))
            print(f"{'s=' + format(nz,'.2f') + ' b=' + format(bi,'.3f'):<18}"
                  f"{e_dead_s:>10.3f}{fin:>11.2f}{best:>11.3f}{best_a:>9.2f}"
                  f"{('YES' if best < e_dead_s else 'no'):>8}")
        print()

    if args.sweep:
        print("\ngain sweep — the filter cannot beat its best input by much, so this")
        print("shows where the optimum sits between odometry and recall:")
        hdr = (f"{'gain':<24}{'median':>9}{'mean':>9}{'p90':>9}"
               f"{'max jump':>10}{'>0.5 m':>9}")
        print(hdr); print("-" * len(hdr))
        for a in (0.02, 0.05, 0.10, 0.20, 0.35, 0.60):
            f_, _ = run_filter("fixed", alpha=a)
            e, jp = err(f_), np.hypot(*np.diff(f_, axis=0).T)
            print(f"{'fixed a=' + format(a, '.2f'):<24}{np.median(e):>9.3f}{e.mean():>9.3f}"
                  f"{np.percentile(e,90):>9.3f}{jp.max():>10.3f}{int((jp>0.5).sum()):>9}")
        for zm in (np.percentile(zs, 75), np.percentile(zs, 90), np.percentile(zs, 97)):
            f_, g_ = run_filter("adaptive", z_mid=zm)
            e, jp = err(f_), np.hypot(*np.diff(f_, axis=0).T)
            print(f"{'adaptive z_mid=' + format(zm, '.1f'):<24}{np.median(e):>9.3f}"
                  f"{e.mean():>9.3f}{np.percentile(e,90):>9.3f}{jp.max():>10.3f}"
                  f"{int((jp>0.5).sum()):>9}   mean gain {g_[1:].mean():.3f}")
        print()

    filt, gains = run_filter(args.gain, alpha=args.alpha, z_mid=args.z_mid)

    e_dead, e_meas, e_filt = err(dead), err(meas), err(filt)
    jump_meas = np.hypot(*np.diff(meas, axis=0).T)
    jump_filt = np.hypot(*np.diff(filt, axis=0).T)
    jump_true = np.hypot(*np.diff(XY, axis=0).T)

    print(f"\ngain: {args.gain}" + (f" (alpha={args.alpha})" if args.gain == "fixed"
          else f" (z_mid={args.z_mid}, mean gain {gains[1:].mean():.3f})"))
    hdr = f"{'track':<22}{'median':>9}{'mean':>9}{'p90':>9}{'max jump':>10}{'>0.5 m jumps':>14}"
    print(hdr); print("-" * len(hdr))
    for name, e, jp in (("dead reckoning", e_dead, np.hypot(*np.diff(dead, axis=0).T)),
                        ("VSA recall (raw)", e_meas, jump_meas),
                        ("VSA Kalman", e_filt, jump_filt),
                        ("true trajectory", np.zeros(n), jump_true)):
        print(f"{name:<22}{np.median(e):>9.3f}{e.mean():>9.3f}{np.percentile(e,90):>9.3f}"
              f"{jp.max():>10.3f}{int((jp > 0.5).sum()):>14}")

    out = os.path.join(args.out_dir, "vsa_kalman.json")
    with open(out, "w") as f:
        json.dump({"encoder": args.encoder, "treatment": args.treatment,
                   "split": args.split, "n_memory": int(mem.sum()), "n_frames": n,
                   "gain": args.gain, "mean_gain": float(gains[1:].mean()),
                   "odom_noise": args.odom_noise, "odom_bias": args.odom_bias,
                   "median": {"dead": float(np.median(e_dead)),
                              "measurement": float(np.median(e_meas)),
                              "filtered": float(np.median(e_filt))},
                   "max_jump": {"measurement": float(jump_meas.max()),
                                "filtered": float(jump_filt.max()),
                                "true": float(jump_true.max())}}, f, indent=2)
    print(f"\nwrote {out}")
    print("'max jump' is the largest frame-to-frame move in the estimate; the true")
    print("trajectory's value is the physical ceiling, so anything above it is an")
    print("artefact the filter should be removing.")


if __name__ == "__main__":
    main()
