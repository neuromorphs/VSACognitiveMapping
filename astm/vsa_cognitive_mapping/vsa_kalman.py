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

Statistics
----------
A single odometry-noise draw moved the headline numbers ~2x between draws, so
results are reported as mean+/-std over ``--noise-seeds`` (default 10 draws).
Everything expensive -- frame descriptors, the phasor projection, the decode
grid, the per-frame measurements (which do not depend on odometry) and the
memory -- is built once and reused. Each noise seed regenerates only the
odometry corruption (``RandomState(noise_seed + 777)``), its STEP vectors, the
dead-reckoning track, and the cheap filter recursion + batched decode. The
memory split stays fixed (``--seed``) across all noise seeds, so between-seed
spread is attributable to odometry noise alone. Aggregates are written to
``vsa_kalman_seeds.json``; the legacy single-run ``vsa_kalman.json`` is only
written when exactly one noise seed is given.

    python -m vsa_cognitive_mapping.vsa_kalman --out-dir outputs/classroom
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

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
    ap.add_argument("--seed", type=int, default=0,
                    help="split seed: fixes WHICH frames the memory holds; kept "
                         "constant across all noise seeds")
    ap.add_argument("--noise-seeds", type=int, nargs="+",
                    default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    help="odometry-noise seeds; results are aggregated as "
                         "mean+/-std over these draws")
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

    # --- per-frame raw measurements (do NOT depend on odometry) -----------
    # Built once and reused across every noise seed, like Z, G and M above.
    R = M[None, :] / Z                                  # unbind every frame at once
    S_all = (R @ np.conj(G).T).real
    j = S_all.argmax(1)
    meas = np.stack([gx[j % args.grid], gy[j // args.grid]], 1)
    zs = (S_all[np.arange(n), j] - S_all.mean(1)) / (S_all.std(1) + 1e-12)

    print(f"measurement confidence z: min {zs.min():.1f}, median {np.median(zs):.1f}, "
          f"p90 {np.percentile(zs,90):.1f}, max {zs.max():.1f}")

    Rn = normalise(R)                       # measurement vectors, unit modulus

    # --- drifting odometry, one draw per noise seed -----------------------
    d_true = np.diff(XY, axis=0)
    step = np.hypot(d_true[:, 0], d_true[:, 1])[:, None]

    def make_odometry(noise_seed, nz, bias):
        """Dedicated stream per noise seed: RandomState(noise_seed + 777),
        independent of the split rng (make_split already advanced `rng`, and
        sharing it once made identical parameters draw different noise:
        0.350 m vs 0.777 m dead-reckoning median)."""
        orng = np.random.RandomState(noise_seed + 777)
        dd = (d_true
              + orng.normal(0, nz, d_true.shape)
              + bias * step * np.array([[1.0, 0.6]]))
        return dd, np.vstack([XY[0], XY[0] + np.cumsum(dd, axis=0)])

    def make_steps(dd):
        """Per-step motion phasors; cheap next to the batched grid decode."""
        return np.stack([enc.ctx_pos(float(dx), float(dy)).values for dx, dy in dd])

    # --- the filter -------------------------------------------------------
    def run_filter(mode, STEP, alpha=None, z_mid=None):
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

    def mstd(v):
        v = np.asarray(v, float)
        return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else 0.0)

    seeds = list(args.noise_seeds)
    attribution = (f"split fixed ({args.split}, --seed {args.seed}) across all "
                   f"noise seeds {seeds}; only the odometry corruption varies "
                   f"(RandomState(noise_seed + 777))")
    print(f"odometry: noise sigma {args.odom_noise} m/step, bias {args.odom_bias} m/m")
    print(attribution)

    GAIN_GRID = (0.02, 0.05, 0.10, 0.20, 0.35, 0.60)
    sweep_json = []
    if args.odom_sweep:
        # The measurement never depends on the odometry, so R/zs are reused and
        # only the motion model changes. This finds the crossover: how bad must
        # dead reckoning get before associative recall is worth blending in?
        # Per (noise, bias) level and per noise seed the gain grid is swept,
        # the per-seed best is taken, THEN mean+/-std is computed over seeds.
        print("\nodometry-quality sweep — the filter can only help once dead")
        print(f"reckoning is worse than the recall correcting it. mean+/-std of the")
        print(f"median error over {len(seeds)} noise seeds; {attribution}")
        hdr = (f"{'noise/bias':<18}{'dead med (m)':>16}{'best filt (m)':>16}"
               f"{'mode gain':>10}{'gain wins':>22}{'helps':>7}")
        print(hdr); print("-" * len(hdr))
        for nz, bi in ((0.02, 0.004), (0.05, 0.010), (0.10, 0.025),
                       (0.20, 0.050), (0.40, 0.100)):
            dead_meds, best_meds, best_gains = [], [], []
            for ns in seeds:
                dd, dead_s = make_odometry(ns, nz, bi)
                dead_meds.append(float(np.median(err(dead_s))))
                STEP_s = make_steps(dd)
                per_gain = {}
                for a in GAIN_GRID:
                    tr, _ = run_filter("fixed", STEP_s, alpha=a)
                    per_gain[a] = float(np.median(err(tr)))
                ba = min(per_gain, key=per_gain.get)
                best_gains.append(ba)
                best_meds.append(per_gain[ba])
            dm, ds = mstd(dead_meds)
            bm, bs = mstd(best_meds)
            wins = Counter(best_gains)
            mode_gain = wins.most_common(1)[0][0]
            helps = int((np.array(best_meds) < np.array(dead_meds)).sum())
            wins_str = ",".join(f"{g:.2f}x{c}" for g, c in wins.most_common())
            print(f"{'s=' + format(nz,'.2f') + ' b=' + format(bi,'.3f'):<18}"
                  f"{format(dm,'.3f') + '+/-' + format(ds,'.3f'):>16}"
                  f"{format(bm,'.3f') + '+/-' + format(bs,'.3f'):>16}"
                  f"{mode_gain:>10.2f}{wins_str:>22}"
                  f"{format(helps) + '/' + format(len(seeds)):>7}")
            sweep_json.append({
                "noise": nz, "bias": bi,
                "dead_median_mean": dm, "dead_median_std": ds,
                "best_filtered_mean": bm, "best_filtered_std": bs,
                "best_gain_mode": mode_gain,
                "gain_wins": {f"{g:.2f}": c for g, c in wins.most_common()},
                "helps_seeds": helps, "n_seeds": len(seeds),
                "per_seed": {"noise_seed": seeds,
                             "dead_median": dead_meds,
                             "best_filtered": best_meds,
                             "best_gain": best_gains}})
        print()

    if args.sweep:
        print(f"\ngain sweep (noise seed {seeds[0]} only) — the filter cannot beat its")
        print("best input by much, so this shows where the optimum sits between")
        print("odometry and recall:")
        dd0, _ = make_odometry(seeds[0], args.odom_noise, args.odom_bias)
        STEP0 = make_steps(dd0)
        hdr = (f"{'gain':<24}{'median':>9}{'mean':>9}{'p90':>9}"
               f"{'max jump':>10}{'>0.5 m':>9}")
        print(hdr); print("-" * len(hdr))
        for a in GAIN_GRID:
            f_, _ = run_filter("fixed", STEP0, alpha=a)
            e, jp = err(f_), np.hypot(*np.diff(f_, axis=0).T)
            print(f"{'fixed a=' + format(a, '.2f'):<24}{np.median(e):>9.3f}{e.mean():>9.3f}"
                  f"{np.percentile(e,90):>9.3f}{jp.max():>10.3f}{int((jp>0.5).sum()):>9}")
        for zm in (np.percentile(zs, 75), np.percentile(zs, 90), np.percentile(zs, 97)):
            f_, g_ = run_filter("adaptive", STEP0, z_mid=zm)
            e, jp = err(f_), np.hypot(*np.diff(f_, axis=0).T)
            print(f"{'adaptive z_mid=' + format(zm, '.1f'):<24}{np.median(e):>9.3f}"
                  f"{e.mean():>9.3f}{np.percentile(e,90):>9.3f}{jp.max():>10.3f}"
                  f"{int((jp>0.5).sum()):>9}   mean gain {g_[1:].mean():.3f}")
        print()

    # --- headline, aggregated over noise seeds ----------------------------
    # Raw recall never sees the odometry, so its stats are seed-independent
    # and reported once.
    e_meas = err(meas)
    jump_meas = np.hypot(*np.diff(meas, axis=0).T)
    jump_true = np.hypot(*np.diff(XY, axis=0).T)

    per_seed = []
    for ns in seeds:
        dd, dead_t = make_odometry(ns, args.odom_noise, args.odom_bias)
        STEP_h = make_steps(dd)
        filt, gains = run_filter(args.gain, STEP_h, alpha=args.alpha, z_mid=args.z_mid)
        e_d, e_f = err(dead_t), err(filt)
        jp_d = np.hypot(*np.diff(dead_t, axis=0).T)
        jp_f = np.hypot(*np.diff(filt, axis=0).T)
        per_seed.append({
            "noise_seed": ns,
            "dead_median": float(np.median(e_d)),
            "dead_mean": float(e_d.mean()),
            "dead_final_drift": float(np.hypot(*(dead_t[-1] - XY[-1]))),
            "dead_jumps_gt_0.5": int((jp_d > 0.5).sum()),
            "filt_median": float(np.median(e_f)),
            "filt_mean": float(e_f.mean()),
            "filt_p90": float(np.percentile(e_f, 90)),
            "filt_max_jump": float(jp_f.max()),
            "filt_jumps_gt_0.5": int((jp_f > 0.5).sum()),
            "mean_gain": float(gains[1:].mean())})

    agg = {k: mstd([p[k] for p in per_seed]) for k in
           ("dead_median", "dead_mean", "dead_final_drift", "dead_jumps_gt_0.5",
            "filt_median", "filt_mean", "filt_p90", "filt_max_jump",
            "filt_jumps_gt_0.5", "mean_gain")}

    print(f"\nheadline over {len(seeds)} noise seeds — gain: {args.gain}"
          + (f" (alpha={args.alpha})" if args.gain == "fixed"
             else f" (z_mid={args.z_mid}, mean gain "
                  f"{agg['mean_gain'][0]:.3f}+/-{agg['mean_gain'][1]:.3f})"))
    print(attribution)
    hdr = (f"{'track':<22}{'median (m)':>16}{'mean (m)':>16}"
           f"{'max jump (m)':>15}{'>0.5 m jumps':>14}")
    print(hdr); print("-" * len(hdr))
    print(f"{'dead reckoning':<22}"
          f"{format(agg['dead_median'][0],'.3f') + '+/-' + format(agg['dead_median'][1],'.3f'):>16}"
          f"{format(agg['dead_mean'][0],'.3f') + '+/-' + format(agg['dead_mean'][1],'.3f'):>16}"
          f"{'—':>15}{format(agg['dead_jumps_gt_0.5'][0],'.1f'):>14}")
    print(f"{'VSA recall (raw)*':<22}{np.median(e_meas):>16.3f}{e_meas.mean():>16.3f}"
          f"{jump_meas.max():>15.3f}{int((jump_meas > 0.5).sum()):>14}")
    print(f"{'VSA Kalman':<22}"
          f"{format(agg['filt_median'][0],'.3f') + '+/-' + format(agg['filt_median'][1],'.3f'):>16}"
          f"{format(agg['filt_mean'][0],'.3f') + '+/-' + format(agg['filt_mean'][1],'.3f'):>16}"
          f"{format(agg['filt_max_jump'][0],'.2f') + '+/-' + format(agg['filt_max_jump'][1],'.2f'):>15}"
          f"{format(agg['filt_jumps_gt_0.5'][0],'.1f') + '+/-' + format(agg['filt_jumps_gt_0.5'][1],'.1f'):>14}")
    print(f"{'true trajectory':<22}{0.0:>16.3f}{0.0:>16.3f}{jump_true.max():>15.3f}"
          f"{int((jump_true > 0.5).sum()):>14}")
    print("* raw recall never sees odometry: seed-independent, reported once.")
    print("mean+/-std is over noise seeds (std: ddof=1); per-seed values in the JSON.")

    out = os.path.join(args.out_dir, "vsa_kalman_seeds.json")
    with open(out, "w") as f:
        json.dump({
            "encoder": args.encoder, "treatment": args.treatment,
            "split": args.split, "split_seed": args.seed,
            "n_memory": int(mem.sum()), "n_frames": n,
            "noise_seeds": seeds, "n_noise_seeds": len(seeds),
            "gain": args.gain, "alpha": args.alpha,
            "odom_noise": args.odom_noise, "odom_bias": args.odom_bias,
            "attribution": attribution,
            "std_definition": "sample std over noise seeds (ddof=1)",
            "raw_recall": {"seed_independent": True,
                           "median": float(np.median(e_meas)),
                           "mean": float(e_meas.mean()),
                           "p90": float(np.percentile(e_meas, 90)),
                           "max_jump": float(jump_meas.max()),
                           "jumps_gt_0.5": int((jump_meas > 0.5).sum())},
            "true_max_jump": float(jump_true.max()),
            "headline": {k: {"mean": agg[k][0], "std": agg[k][1]} for k in agg},
            "headline_per_seed": per_seed,
            "odom_sweep": sweep_json}, f, indent=2)
    print(f"\nwrote {out}")

    if len(seeds) == 1:
        # legacy single-run artifact, only meaningful for a single noise draw
        p = per_seed[0]
        legacy = os.path.join(args.out_dir, "vsa_kalman.json")
        with open(legacy, "w") as f:
            json.dump({"encoder": args.encoder, "treatment": args.treatment,
                       "split": args.split, "n_memory": int(mem.sum()), "n_frames": n,
                       "gain": args.gain, "mean_gain": p["mean_gain"],
                       "noise_seed": seeds[0],
                       "odom_noise": args.odom_noise, "odom_bias": args.odom_bias,
                       "median": {"dead": p["dead_median"],
                                  "measurement": float(np.median(e_meas)),
                                  "filtered": p["filt_median"]},
                       "max_jump": {"measurement": float(jump_meas.max()),
                                    "filtered": p["filt_max_jump"],
                                    "true": float(jump_true.max())}}, f, indent=2)
        print(f"wrote {legacy}")
    print("'max jump' is the largest frame-to-frame move in the estimate; the true")
    print("trajectory's value is the physical ceiling, so anything above it is an")
    print("artefact the filter should be removing.")


if __name__ == "__main__":
    main()
