"""FRAME RECOVERY for VSA map merging on geometric lidar maps — the missing
robustness experiment behind the standing caveat:

    "exactness of the addition-merge is conditional on a shared coordinate
     frame; misaligned traces sum to a superposition of two inconsistent
     maps with no error signal."

This script measures whether that shared frame can be RECOVERED from the map
vectors alone, and what misalignment actually costs downstream.

Setup (same world as merge_comparison_lidar.py): a 32x24 room with three box
obstacles, one 800-step trajectory (seed 7), D=2048 phasor FPE. Two robots A
and B take contiguous trajectory blocks whose overlap fraction f is
controlled:  A = steps [0, 400 + f*400),  B = steps [400 - f*400, 800),  so f
is exactly the fraction of the 800-step trajectory both robots see (f=0
disjoint halves, f=1 identical full trajectories). Robot B builds its
signature in a frame offset by an unknown translation delta, i.e. from poses
x_t + delta.

Key algebraic identities exploited:
  * signature_from_scan(P, x + delta) = phi(delta) (.) signature_from_scan(P, x)
    because phi(a+b) = phi(a) (.) phi(b). Offsetting B's frame therefore
    multiplies B's whole signature-sum by phi(delta) EXACTLY — scans are
    encoded once and reused across every offset trial.
  * Gauge-shift decode (vgpi_online_s.ipynb section 7.1): since S_B_offset
    carries phi(m + delta) mass where S_A carries phi(m), the product
    Q = S_B_offset (.) conj(S_A) has a phi(delta) component (diagonal terms;
    everything else is bundling crosstalk). ONE correlation of Q against a
    codebook of candidate shifts phi(d) over a delta-grid recovers delta —
    aligning two maps costs one correlation, because the offset between them
    is literally a single bound vector.

TRANSLATION ONLY: in the 2-D fractional-power encoding, translation is a bind
(phi(x+d) = phi(x) (.) phi(d)) but rotation is NOT — it acts on the frequency
matrix, not elementwise on the code (a rotation-closed lattice encoder, cf.
LatticeEncoder.rotate, would be needed). Rotational frame recovery is out of
scope here and stated as such.

Scoring, per (overlap f) x (offset magnitude) x (direction) trial:
  * recovery success = |delta_hat - delta| <= 0.25 code units (the
    sub-wavelength merge-tolerance bar from prior work; the sinc kernel's
    first zero is at 1.0 unit).
  * merged-map localization error (median over held-out test scans from
    everywhere in the room) for three merges:
      oracle      A + B merged with the TRUE delta removed (shared frame),
      recovered   A + B merged after removing the DECODED delta_hat,
      misaligned  A + B_offset summed as-is — the "no error signal"
                  condition the caveat warns about.

Outputs: outputs/lidar_merge/frame_recovery.json (all trials) and ONE figure
outputs/lidar_merge/frame_recovery.png with (a) success rate vs overlap
fraction per offset magnitude, (b) merged-map median error for oracle vs
recovered vs misaligned, (c) an example delta-correlation surface.

    python -m vsa_cognitive_mapping.frame_recovery_lidar
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
LIDAR_VSA_SRC = os.path.join(os.path.dirname(os.path.dirname(HERE)),
                             "le-marmotte-vsa-grid-cells-in-silicon", "src")
sys.path.insert(0, LIDAR_VSA_SRC)

from lidar_vsa.vsa import SpatialEncoder                        # noqa: E402
from lidar_vsa.world import (room_with_boxes, free_space_mask,  # noqa: E402
                             smooth_random_trajectory)
from lidar_vsa.encoding import scan_hit_vectors, code_from_hits  # noqa: E402
from lidar_vsa.readout import PositionCodebook, make_grid       # noqa: E402

WIDTH, HEIGHT = 32.0, 24.0
BOXES = [(9.0, 13.0, 5.0, 9.0), (20.0, 26.0, 14.0, 18.0),
         (16.0, 19.0, 4.0, 7.0)]

OVERLAPS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]     # fraction of trajectory shared
MAGS = [0.5, 1.0, 2.0, 4.0, 8.0]                # |delta| in code units
N_DIRS = 3                                       # random directions per magnitude
TOL = 0.25                                       # success bar |dhat - d| <= TOL
EXAMPLE_TRIAL = (0.25, 2.0, 0)                   # (f, mag, dir) for panel (c)


def decode_delta(Q, enc, cb_coarse, grid_coarse, fine_pitch=0.05,
                 fine_half=0.35):
    """argmax_d Re<phi(d), Q> over the coarse delta-grid, then a local refine.

    Returns (delta_hat (2,), coarse spectrum (M,), normalized peak coherence).
    """
    spec = np.real(np.conj(cb_coarse) @ Q)
    d0 = grid_coarse[int(np.argmax(spec))]
    off = np.arange(-fine_half, fine_half + 1e-9, fine_pitch)
    fg = make_grid(d0[0] + off, d0[1] + off)
    fspec = np.real(np.conj(enc.encode(fg[:, 0], fg[:, 1])) @ Q)
    dhat = fg[int(np.argmax(fspec))]
    peak = float(fspec.max() / (np.sqrt(enc.dim) * np.linalg.norm(Q) + 1e-12))
    return dhat, spec, peak


def hit_cells(world_pts, res=0.5):
    """Set of binned boundary cells hit by a robot's scans (spatial coverage)."""
    ij = np.floor(world_pts / res).astype(int)
    return set(map(tuple, ij))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dim", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--beams", type=int, default=360)
    ap.add_argument("--tests", type=int, default=150)
    ap.add_argument("--pitch", type=float, default=0.25,
                    help="room localization codebook pitch (code units)")
    ap.add_argument("--delta-extent", type=float, default=10.0,
                    help="half-extent of the delta search grid")
    ap.add_argument("--delta-pitch", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-dir", default=os.path.join(HERE, "..", "outputs",
                                                      "lidar_merge"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    segments = room_with_boxes(WIDTH, HEIGHT, BOXES)
    angles = np.linspace(0, 2 * np.pi, args.beams, endpoint=False)
    enc = SpatialEncoder(dim=args.dim)
    traj = smooth_random_trajectory(n=args.steps, seed=args.seed,
                                    boxes=BOXES, width=WIDTH, height=HEIGHT)
    half = args.steps // 2

    # ---- encode every scan ONCE; offsets never touch the scans again -------
    print(f"encoding {args.steps} scans (D={args.dim}) ...")
    sigs = np.empty((args.steps, args.dim), complex)
    hit_pts = []                                   # world-frame hit points
    for t in range(args.steps):
        hx, hy = scan_hit_vectors(traj[t], angles, segments)
        P = code_from_hits(enc, hx, hy)            # egocentric scan code
        sigs[t] = P * enc.encode(traj[t][0], traj[t][1])   # signature term
        hit_pts.append(np.stack([traj[t][0] + hx, traj[t][1] + hy], 1))
    csum = np.concatenate([np.zeros((1, args.dim), complex),
                           np.cumsum(sigs, axis=0)])      # block sums for free

    def block(lo, hi):
        return csum[hi] - csum[lo], hi - lo

    # ---- held-out test scans everywhere in the room ------------------------
    tests = []
    while len(tests) < args.tests:
        cand = rng.uniform([0.8, 0.8], [WIDTH - 0.8, HEIGHT - 0.8],
                           (args.tests, 2))
        ok = free_space_mask(cand, WIDTH, HEIGHT, BOXES, margin=0.6)
        tests += [c for c in cand[ok]]
    tests = np.array(tests[:args.tests])
    test_codes = np.empty((args.tests, args.dim), complex)
    for i, p in enumerate(tests):
        hx, hy = scan_hit_vectors(p, angles, segments)
        test_codes[i] = code_from_hits(enc, hx, hy)

    xs = np.arange(0.5, WIDTH, args.pitch)
    ys = np.arange(0.5, HEIGHT, args.pitch)
    cb = PositionCodebook(enc, make_grid(xs, ys))

    def med_err(S):
        est = cb.decode_batch(test_codes, S)
        return float(np.median(np.hypot(*(est - tests).T)))

    # ---- delta search codebook ---------------------------------------------
    ds = np.arange(-args.delta_extent, args.delta_extent + 1e-9,
                   args.delta_pitch)
    grid_d = make_grid(ds, ds)
    cb_d = enc.encode(grid_d[:, 0], grid_d[:, 1])
    print(f"delta search grid: {len(ds)}x{len(ds)} at {args.delta_pitch}u "
          f"(then 0.05u local refine)")

    # ---- sanity check: f=1, no noise — decode must be near-exact -----------
    S_full, n_full = block(0, args.steps)
    d_chk = np.array([1.3, -0.7])
    Q = (enc.encode(*d_chk) * S_full / n_full) * np.conj(S_full / n_full)
    dhat, _, pk = decode_delta(Q, enc, cb_d, grid_d)
    chk = float(np.hypot(*(dhat - d_chk)))
    print(f"sanity (f=1, delta={d_chk}): delta_hat={np.round(dhat, 3)} "
          f"err={chk:.3f}u peak={pk:.3f}"
          + ("  OK" if chk <= 0.1 else "  ** FAILED — decode is broken **"))
    if chk > 0.1:
        sys.exit(1)

    # same directions for every (f, mag) cell -> comparable lines across f
    thetas = {m: rng.uniform(0, 2 * np.pi, N_DIRS) for m in MAGS}

    trials, oracle_rows = [], {}
    example_spec = None
    for f in OVERLAPS:
        a_lo, a_hi = 0, int(round(half + f * half))
        b_lo, b_hi = int(round(half - f * half)), args.steps
        sum_A, nA = block(a_lo, a_hi)
        sum_B, nB = block(b_lo, b_hi)

        cells_A = set().union(*(hit_cells(hit_pts[t]) for t in range(a_lo, a_hi)))
        cells_B = set().union(*(hit_cells(hit_pts[t]) for t in range(b_lo, b_hi)))
        iou = len(cells_A & cells_B) / len(cells_A | cells_B)

        S_oracle = (sum_A + sum_B) / (nA + nB)     # true-delta-removed merge
        e_oracle = med_err(S_oracle)
        oracle_rows[f] = dict(overlap=f, blocks=[[a_lo, a_hi], [b_lo, b_hi]],
                              shared_steps=max(a_hi - b_lo, 0),
                              hit_cell_iou=iou, med_err_oracle=e_oracle)
        print(f"\nf={f:.2f}: A=[{a_lo},{a_hi}) B=[{b_lo},{b_hi}) "
              f"shared={max(a_hi - b_lo, 0)} steps, hit-cell IoU={iou:.2f}, "
              f"oracle merge med err={e_oracle:.2f}u")

        for m in MAGS:
            for k, th in enumerate(thetas[m]):
                delta = m * np.array([np.cos(th), np.sin(th)])
                phi_d = enc.encode(*delta)
                sum_B_off = phi_d * sum_B          # exact offset, no re-encode
                Q = (sum_B_off / nB) * np.conj(sum_A / nA)
                dhat, spec, pk = decode_delta(Q, enc, cb_d, grid_d)
                err_d = float(np.hypot(*(dhat - delta)))
                success = err_d <= TOL

                S_rec = (sum_A + np.conj(enc.encode(*dhat)) * sum_B_off) / (nA + nB)
                S_naive = (sum_A + sum_B_off) / (nA + nB)
                e_rec, e_naive = med_err(S_rec), med_err(S_naive)

                trials.append(dict(
                    overlap=f, mag=m, dir=k, theta=float(th),
                    delta=[float(delta[0]), float(delta[1])],
                    delta_hat=[float(dhat[0]), float(dhat[1])],
                    err_delta=err_d, success=bool(success), peak=pk,
                    med_err_oracle=e_oracle, med_err_recovered=e_rec,
                    med_err_misaligned=e_naive))
                print(f"  |d|={m:<4} dir{k}: err_d={err_d:6.3f}u "
                      f"{'OK ' if success else 'FAIL'} peak={pk:.3f} | "
                      f"med err oracle/rec/naive = "
                      f"{e_oracle:.2f}/{e_rec:.2f}/{e_naive:.2f}u")
                if (f, m, k) == EXAMPLE_TRIAL:
                    example_spec = dict(spec=spec.reshape(len(ds), len(ds)),
                                        delta=delta, dhat=dhat, f=f, m=m)

    # ---- success-rate table (overlap x magnitude) --------------------------
    T = {(t["overlap"], t["mag"]): [] for t in trials}
    for t in trials:
        T[(t["overlap"], t["mag"])].append(t["success"])
    rate = {fm: float(np.mean(v)) for fm, v in T.items()}

    print(f"\nrecovery success rate (|delta_hat - delta| <= {TOL}u), "
          f"{N_DIRS} directions per cell:")
    hdr = "overlap f " + "".join(f"|d|={m:>5} " for m in MAGS) + " hit-IoU"
    print(hdr)
    print("-" * len(hdr))
    for f in OVERLAPS:
        row = f"{f:<10.2f}" + "".join(f"{rate[(f, m)]:>8.0%}  " for m in MAGS)
        print(row + f" {oracle_rows[f]['hit_cell_iou']:.2f}")

    def med(key, sel=lambda t: True):
        v = [t[key] for t in trials if sel(t)]
        return float(np.median(v)) if v else float("nan")

    print("\nmerged-map localization, median over trials of per-trial medians "
          f"({args.tests} held-out scans each):")
    print(f"  oracle (true frame):     {med('med_err_oracle'):.2f}u")
    print(f"  recovered delta_hat:     {med('med_err_recovered'):.2f}u")
    print(f"  misaligned (uncorrected):{med('med_err_misaligned'):.2f}u")

    out_json = os.path.join(args.out_dir, "frame_recovery.json")
    with open(out_json, "w") as fjs:
        json.dump(dict(
            dim=args.dim, steps=args.steps, seed=args.seed, tests=args.tests,
            tol=TOL, delta_grid=dict(extent=args.delta_extent,
                                     pitch=args.delta_pitch, refine=0.05),
            overlaps=[oracle_rows[f] for f in OVERLAPS],
            success_rate={f"f={f}|mag={m}": rate[(f, m)]
                          for f in OVERLAPS for m in MAGS},
            trials=trials), fjs, indent=2)
    print(f"\nwrote {out_json}")

    # ---- ONE figure --------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))

    axa = axes[0]
    for m in MAGS:
        axa.plot(OVERLAPS, [rate[(f, m)] for f in OVERLAPS], "o-",
                 label=f"$|\\delta|$ = {m}u")
    axa.set_xlabel("trajectory overlap fraction f")
    axa.set_ylabel(f"recovery success rate (err ≤ {TOL}u)")
    axa.set_ylim(-0.05, 1.05)
    axa.grid(alpha=0.3)
    axa.legend(fontsize=8, title="offset magnitude", title_fontsize=8)
    axa.set_title("(a) frame recovery vs overlap\n"
                  f"({N_DIRS} random directions per point)", fontsize=10)

    axb = axes[1]
    w = 0.25
    xpos = np.arange(len(MAGS))
    for i, (key, lab, col) in enumerate([
            ("med_err_oracle", "oracle (true frame)", "tab:green"),
            ("med_err_recovered", "recovered $\\hat{\\delta}$", "tab:blue"),
            ("med_err_misaligned", "misaligned (uncorrected)", "tab:red")]):
        vals = [med(key, lambda t, m=m: t["mag"] == m) for m in MAGS]
        axb.bar(xpos + (i - 1) * w, vals, w, label=lab, color=col, alpha=0.85)
    axb.set_xticks(xpos)
    axb.set_xticklabels([f"{m}" for m in MAGS])
    axb.set_xlabel("offset magnitude $|\\delta|$ [code units]")
    axb.set_ylabel("merged-map median localization error [u]")
    axb.legend(fontsize=8)
    axb.grid(alpha=0.3, axis="y")
    axb.set_title("(b) what misalignment costs downstream\n"
                  "(median over all overlaps × directions)", fontsize=10)

    axc = axes[2]
    if example_spec is not None:
        E = args.delta_extent
        im = axc.imshow(example_spec["spec"], origin="lower", cmap="magma",
                        extent=[-E, E, -E, E], aspect="equal")
        axc.plot(*example_spec["delta"], "c+", ms=14, mew=2,
                 label="true $\\delta$")
        axc.plot(*example_spec["dhat"], "wx", ms=9, mew=1.5,
                 label="decoded $\\hat{\\delta}$")
        axc.legend(fontsize=8, loc="upper right")
        fig.colorbar(im, ax=axc, shrink=0.85)
        axc.set_xlabel("$\\delta_x$ [u]")
        axc.set_ylabel("$\\delta_y$ [u]")
        axc.set_title("(c) Re$\\langle\\varphi(d),\\ S_B \\odot \\bar S_A\\rangle$,"
                      " one correlation\n"
                      f"(f={example_spec['f']}, $|\\delta|$={example_spec['m']}u)",
                      fontsize=10)
    fig.suptitle("Recovering the shared frame of two VSA lidar maps from the map "
                 "vectors alone (translation gauge; rotation is not a bind in 2-D FPE)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_png = os.path.join(args.out_dir, "frame_recovery.png")
    fig.savefig(out_png, dpi=140)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
