"""Map JOINING head-to-head on GEOMETRIC lidar maps: VSA addition vs the
alternatives, same data — the boundary-signature companion to
merge_comparison.py (which does the semantic/object case on Replica).

Data: the le-marmotte lidar-VSA simulator (neuromorphs/
le-marmotte-vsa-grid-cells-in-silicon, branch interactive-demo, package
src/lidar_vsa). A room with obstacles, one long trajectory split into K
contiguous blocks = K single-session robots. Each robot builds its own map
from its scans at KNOWN poses (ground-truth-anchored: this isolates the
MERGE step, per the standing caveat — alignment/frame recovery is orthogonal
and not solved by any join below). The joined map then localizes held-out
scans from everywhere in the room.

Methods (what is transmitted, how joining works, how localization works):
  vsa      one (D-dim signature-sum, scan-count) per robot; join =
           element-wise addition + count add. Bit-exactness measured
           against the jointly built signature. Localization: one inner
           product sweep against a position codebook (readout.py).
  gridmap  per-robot lidar hit-count grid; join = cell-wise addition (also
           exact). Localization: template scoring of the test scan's hits
           against the (blurred) joined grid over candidate positions —
           the occupancy-grid analogue.
  scandb   raw scan database (pose + 360 ranges per kept frame); join =
           concatenation (exact, unbounded growth). Localization: nearest
           stored scan by range-vector correlation -> its pose.

All joins assume a SHARED WORLD FRAME (stated on output).

    python -m vsa_cognitive_mapping.merge_comparison_lidar --robots 4
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

from lidar_vsa.vsa import SpatialEncoder                     # noqa: E402
from lidar_vsa.world import (room_with_boxes, free_space_mask,  # noqa: E402
                             smooth_random_trajectory)
from lidar_vsa.encoding import (lidar_position_code,          # noqa: E402
                                scan_hit_vectors, signature_from_scan)
from lidar_vsa.readout import (PositionCodebook, make_grid,   # noqa: E402
                               signature_similarity_grid)

WIDTH, HEIGHT = 32.0, 24.0
BOXES = [(9.0, 13.0, 5.0, 9.0), (20.0, 26.0, 14.0, 18.0),
         (16.0, 19.0, 4.0, 7.0)]


def blur(grid, sigma_cells):
    """Separable Gaussian blur via FFT (numpy-only)."""
    ny, nx = grid.shape
    fy = np.fft.fftfreq(ny)
    fx = np.fft.fftfreq(nx)
    ky = np.exp(-2 * (np.pi * fy * sigma_cells) ** 2)
    kx = np.exp(-2 * (np.pi * fx * sigma_cells) ** 2)
    return np.real(np.fft.ifft2(np.fft.fft2(grid) * ky[:, None] * kx[None, :]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robots", type=int, default=4)
    ap.add_argument("--dim", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--beams", type=int, default=360)
    ap.add_argument("--tests", type=int, default=300)
    ap.add_argument("--pitch", type=float, default=0.25,
                    help="decode grid pitch (code units)")
    ap.add_argument("--occ-res", type=float, default=0.25,
                    help="occupancy grid cell size")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-dir", default=os.path.join(HERE, "..", "outputs",
                                                      "lidar_merge"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    segments = room_with_boxes(WIDTH, HEIGHT, BOXES)
    angles = np.linspace(0, 2 * np.pi, args.beams, endpoint=False)
    enc = SpatialEncoder(dim=args.dim)

    # one long walk -> K contiguous single-session blocks
    traj = smooth_random_trajectory(n=args.steps, seed=args.seed,
                                    boxes=BOXES, width=WIDTH, height=HEIGHT)
    blocks = np.array_split(np.arange(args.steps), args.robots)

    # every method consumes the identical scans (same-frontend rule)
    scans = [lidar_position_code(enc, traj[t], angles, segments)
             for t in range(args.steps)]
    hits = [scan_hit_vectors(traj[t], angles, segments)
            for t in range(args.steps)]
    ranges = np.array([np.hypot(hx, hy) for hx, hy in hits])

    print(f"{args.steps} scans -> {args.robots} robots "
          f"({[len(b) for b in blocks]} each); shared world frame ASSUMED "
          f"(alignment is orthogonal to every join below)\n")

    # held-out test poses everywhere in free space (not on the trajectory)
    tests = []
    while len(tests) < args.tests:
        cand = rng.uniform([0.8, 0.8], [WIDTH - 0.8, HEIGHT - 0.8],
                           (args.tests, 2))
        ok = free_space_mask(cand, WIDTH, HEIGHT, BOXES, margin=0.6)
        tests += [c for c in cand[ok]]
    tests = np.array(tests[:args.tests])
    test_codes = np.array([lidar_position_code(enc, p, angles, segments)
                           for p in tests])
    test_hits = [scan_hit_vectors(p, angles, segments) for p in tests]
    test_ranges = np.array([np.hypot(hx, hy) for hx, hy in test_hits])

    rows = []

    # ---- vsa: per-robot signature sums, join by addition ------------------
    parts, counts = [], []
    for b in blocks:
        s = np.zeros(args.dim, complex)
        for t in b:
            s += signature_from_scan(scans[t], enc, traj[t])
        parts.append(s)
        counts.append(len(b))
    S_join = sum(parts) / sum(counts)
    S_joint = sum(signature_from_scan(scans[t], enc, traj[t])
                  for t in range(args.steps)) / args.steps
    exact = float(np.abs(S_join - S_joint).max())

    xs = np.arange(0.5, WIDTH, args.pitch)
    ys = np.arange(0.5, HEIGHT, args.pitch)
    cb = PositionCodebook(enc, make_grid(xs, ys))
    est = cb.decode_batch(test_codes, S_join)
    err_vsa = np.hypot(*(est - tests).T)
    rows.append(dict(method="VSA signature + addition",
                     bytes_per_robot=args.dim * 16,
                     join_op="element-wise add",
                     assoc_decisions=0,
                     exact_vs_joint=f"{exact:.1e}",
                     med_err=float(np.median(err_vsa)),
                     r_at_05=float(np.mean(err_vsa <= 0.5))))

    # ---- gridmap: per-robot hit-count grids, join by addition -------------
    gx = np.arange(0, WIDTH + args.occ_res, args.occ_res)
    gy = np.arange(0, HEIGHT + args.occ_res, args.occ_res)
    nx, ny = len(gx), len(gy)
    grids = []
    for b in blocks:
        g = np.zeros((ny, nx), np.float32)
        for t in b:
            hx, hy = hits[t]
            ix = np.clip(((traj[t][0] + hx) / args.occ_res).astype(int), 0, nx - 1)
            iy = np.clip(((traj[t][1] + hy) / args.occ_res).astype(int), 0, ny - 1)
            np.add.at(g, (iy, ix), 1)
        grids.append(g)
    G_join = sum(grids)
    occ = blur(np.log1p(G_join), sigma_cells=1.0)      # log-count, blurred

    cand = cb.grid
    err_grid = np.empty(args.tests)
    for i, (hx, hy) in enumerate(test_hits):
        wx = cand[:, 0][:, None] + hx[None, :]
        wy = cand[:, 1][:, None] + hy[None, :]
        ix = np.clip((wx / args.occ_res).astype(int), 0, nx - 1)
        iy = np.clip((wy / args.occ_res).astype(int), 0, ny - 1)
        score = occ[iy, ix].sum(1)
        err_grid[i] = np.hypot(*(cand[np.argmax(score)] - tests[i]))
    rows.append(dict(method="hit-count grid + addition",
                     bytes_per_robot=int(nx * ny * 2),
                     join_op="cell-wise add",
                     assoc_decisions=0,
                     exact_vs_joint="exact (by construction)",
                     med_err=float(np.median(err_grid)),
                     r_at_05=float(np.mean(err_grid <= 0.5))))

    # ---- scandb: raw scan database, join by concatenation -----------------
    mu = ranges.mean(1, keepdims=True)
    Z = ranges - mu
    Zt = test_ranges - test_ranges.mean(1, keepdims=True)
    sims = Zt @ Z.T / (np.linalg.norm(Zt, axis=1)[:, None]
                       * np.linalg.norm(Z, axis=1)[None, :] + 1e-12)
    est_db = traj[np.argmax(sims, 1)]
    err_db = np.hypot(*(est_db - tests).T)
    rows.append(dict(method="raw scan database union",
                     bytes_per_robot=int(args.steps / args.robots
                                         * (args.beams + 2) * 4),
                     join_op="concatenate",
                     assoc_decisions=0,
                     exact_vs_joint="exact (trivially)",
                     med_err=float(np.median(err_db)),
                     r_at_05=float(np.mean(err_db <= 0.5))))

    hdr = (f"{'method':<28}{'bytes/robot':>12}{'join':>20}"
           f"{'assoc':>7}{'med err':>9}{'R@0.5':>8}   exact-vs-joint")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['method']:<28}{r['bytes_per_robot']/1024:>10.1f}KB"
              f"{r['join_op']:>20}{r['assoc_decisions']:>7}"
              f"{r['med_err']:>8.2f}u{r['r_at_05']:>8.0%}   "
              f"{r['exact_vs_joint']}")

    out = os.path.join(args.out_dir, f"merge_comparison_lidar_k{args.robots}.json")
    with open(out, "w") as f:
        json.dump(dict(robots=args.robots, rows=rows, steps=args.steps,
                       dim=args.dim, tests=args.tests), f, indent=2)
    print(f"\nwrote {out}")

    # ---- figure: per-robot maps, joined map, error CDFs -------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from lidar_vsa.world import draw_room

    fxs = np.arange(0, WIDTH, 0.15)
    fys = np.arange(0, HEIGHT, 0.15)
    ncol = args.robots + 2
    fig, axes = plt.subplots(2, ncol, figsize=(3.1 * ncol, 6.4))
    for i, (p, c, b) in enumerate(zip(parts, counts, blocks)):
        F = signature_similarity_grid(enc, p / c, fxs, fys)
        ax = axes[0, i]
        ax.imshow(F, origin="lower", extent=[0, WIDTH, 0, HEIGHT],
                  cmap="magma")
        ax.plot(*traj[b].T, "c-", lw=0.7, alpha=0.8)
        draw_room(ax, segments)
        ax.set_title(f"robot {i + 1}: S$_{i + 1}$ ({c} scans)", fontsize=9)
        ax.set_xticks([]), ax.set_yticks([])
    F = signature_similarity_grid(enc, S_join, fxs, fys)
    ax = axes[0, args.robots]
    ax.imshow(F, origin="lower", extent=[0, WIDTH, 0, HEIGHT], cmap="magma")
    draw_room(ax, segments)
    ax.set_title(f"joined = $\\Sigma$ S$_k$  (32 KB,\nmax|Δ| vs joint = {exact:.0e})",
                 fontsize=9)
    ax.set_xticks([]), ax.set_yticks([])
    ax = axes[0, args.robots + 1]
    ax.imshow(np.log1p(G_join), origin="lower",
              extent=[0, WIDTH, 0, HEIGHT], cmap="magma")
    draw_room(ax, segments)
    ax.set_title("grid join (cell add)", fontsize=9)
    ax.set_xticks([]), ax.set_yticks([])

    axl = axes[1, 0]
    for e, lab in ((err_vsa, "VSA signature + addition"),
                   (err_grid, "hit-count grid + addition"),
                   (err_db, "raw scan DB union")):
        axl.plot(np.sort(e), np.linspace(0, 1, len(e)),
                 label=f"{lab} (med {np.median(e):.2f}u)")
    axl.set_xlim(0, 3), axl.set_ylim(0, 1)
    axl.set_xlabel("localization error of held-out scans [code units]")
    axl.set_ylabel("CDF")
    axl.legend(fontsize=7, loc="lower right")
    axl.grid(alpha=0.3)
    ax = axes[1, 1]
    ax.scatter(*tests.T, c=err_vsa, s=8, cmap="viridis", vmax=1.5)
    draw_room(ax, segments)
    ax.set_title("VSA joined-map error by test pose", fontsize=9)
    ax.set_xticks([]), ax.set_yticks([])
    for j in range(2, ncol):
        axes[1, j].axis("off")
    txt = "\n".join(
        f"{r['method']:<28} {r['bytes_per_robot']/1024:7.1f} KB/robot   "
        f"med {r['med_err']:.2f}u   R@0.5 {r['r_at_05']:.0%}   {r['exact_vs_joint']}"
        for r in rows)
    axes[1, 2].text(0, 0.7, "K = %d robots, shared frame assumed\n\n%s"
                    % (args.robots, txt), fontsize=8, family="monospace",
                    va="top", transform=axes[1, 2].transAxes)
    fig.suptitle("One geometric map from many: joining K single-session lidar maps "
                 "(le-marmotte phasor-VSA simulator)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    png = os.path.join(args.out_dir, f"merge_comparison_lidar_k{args.robots}.png")
    fig.savefig(png, dpi=140)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
