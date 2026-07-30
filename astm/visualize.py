"""Matplotlib visualiser for the VSA associative-memory cognitive map.

Mirrors how ``neuromorphs/VSACognitiveMapping`` shows its results: a set of
PNG panels saved to a ``plots/`` folder, plus an animated demo that sweeps
the queried frame along the trajectory and watches the localization field
track the robot (the upstream ``demo`` command renders the same idea to
``demo.mp4``; we use an animated GIF so no ffmpeg is needed).

It reuses the exact pipeline from ``demo_associative_memory.py`` so the
pictures and the console demo are the same computation.

Run:
    python vsa_cognitive_mapping/visualize.py                 # save all panels + gif to ./plots
    python vsa_cognitive_mapping/visualize.py --show          # also pop up interactive windows
    python vsa_cognitive_mapping/visualize.py --no-gif        # skip the (slower) animation
    python vsa_cognitive_mapping/visualize.py --hd-dim 4096 --n-frames 32 --frame 10
    python vsa_cognitive_mapping/visualize.py --out-dir plots/vsa_demo
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vsa_cognitive_mapping.vsa import Phasor, phasor_correlation_matrix  # noqa: E402
from vsa_cognitive_mapping.demo_associative_memory import (  # noqa: E402
    make_scene, encode_content, ContextEncoders, build_memory, cleanup,
    evaluate_position_recall,
)


# --------------------------------------------------------------------------
# Vectorised helpers (fast enough to animate on CPU)
# --------------------------------------------------------------------------

def build_ctx_grid(enc: ContextEncoders, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """Precompute ctx_pos(x, y) for every grid cell once: (ny, nx, d) complex."""
    d = enc.Bx.dim
    grid = np.empty((len(gy), len(gx), d), dtype=np.complex128)
    for j, y in enumerate(gy):
        for i, x in enumerate(gx):
            grid[j, i] = enc.ctx_pos(x, y).values
    return grid


def localization_field(memory: Phasor, probe: Phasor, ctx_grid: np.ndarray) -> np.ndarray:
    """Similarity field of (memory unbound by probe) against every grid ctx.

    probe is a *content* phasor ("where have I seen this?"). Returns (ny, nx)."""
    residual = (memory / probe).values                      # (d,)
    d = residual.shape[0]
    flat = ctx_grid.reshape(-1, d)                          # (ny*nx, d)
    sims = np.real(flat @ np.conj(residual)) / d            # Re<ctx, residual>/d
    return sims.reshape(ctx_grid.shape[:2])


# --------------------------------------------------------------------------
# Panels
# --------------------------------------------------------------------------

def panel_correlations(plt, content, positions, out):
    """Content self-similarity matrix + the trajectory it came from.

    The analogue of upstream ``plot_embedding_correlation`` — how distinct the
    per-frame phasor content vectors are from each other."""
    C = phasor_correlation_matrix(np.stack([c.values for c in content]))
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(10, 4.4), constrained_layout=True)

    im = a0.imshow(C, cmap="coolwarm", vmin=-1, vmax=1)
    a0.set_title("content self-similarity\n(phasor correlation matrix)", fontsize=11)
    a0.set_xlabel("frame"); a0.set_ylabel("frame")
    fig.colorbar(im, ax=a0, fraction=0.046, pad=0.04)

    n = len(positions)
    a1.plot(positions[:, 0], positions[:, 1], "-", color="0.7", lw=1, zorder=1)
    sc = a1.scatter(positions[:, 0], positions[:, 1], c=np.arange(n), cmap="viridis",
                    s=60, zorder=2, edgecolor="white", linewidth=0.6)
    a1.set_title("robot trajectory (coloured by frame)", fontsize=11)
    a1.set_xlabel("x (m)"); a1.set_ylabel("y (m)"); a1.set_aspect("equal", "box")
    fig.colorbar(sc, ax=a1, fraction=0.046, pad=0.04, label="frame")
    return _save(fig, out, "correlations.png")


def panel_localization(plt, memory, content, enc, positions, frame, gx, gy, ctx_grid, out):
    """Reverse query one frame's content -> where the map says it was seen."""
    field = localization_field(memory, content[frame], ctx_grid)
    tx, ty = positions[frame]
    jpk, ipk = np.unravel_index(int(np.argmax(field)), field.shape)

    fig, ax = plt.subplots(figsize=(6.6, 5.2), constrained_layout=True)
    lim = np.max(np.abs(field))
    im = ax.imshow(field, origin="lower", cmap="coolwarm", vmin=-lim, vmax=lim,
                   extent=[gx[0], gx[-1], gy[0], gy[-1]], aspect="equal")
    ax.plot(positions[:, 0], positions[:, 1], "-", color="0.35", lw=1, alpha=0.6)
    ax.scatter(positions[:, 0], positions[:, 1], s=14, facecolor="none",
               edgecolor="0.2", linewidth=0.8)
    ax.scatter([tx], [ty], s=180, marker="o", facecolor="none",
               edgecolor="black", linewidth=2.2, label="true location")
    ax.scatter([gx[ipk]], [gy[jpk]], s=90, marker="x", color="black",
               linewidth=2.2, label="recovered peak")
    ax.set_title(f"localization by content  (frame {frame})\n"
                 f"query: 'where did I see this?'", fontsize=11)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="similarity")
    return _save(fig, out, "localization.png")


def panel_kernel(plt, memory, content, enc, positions, frame, out, span=6.0, steps=121):
    """1-D slice of the localization field through the true y -> the FPE kernel."""
    tx, ty = positions[frame]
    residual = memory / content[frame]
    xs = np.linspace(tx - span, tx + span, steps)
    sims = np.array([residual.similarity(enc.ctx_pos(x, ty)) for x in xs])
    peak = xs[int(np.argmax(sims))]

    fig, ax = plt.subplots(figsize=(7.2, 4.0), constrained_layout=True)
    ax.axhline(0, color="0.8", lw=1)
    ax.fill_between(xs, 0, sims, color="#0d7d88", alpha=0.15)
    ax.plot(xs, sims, color="#0d7d88", lw=2)
    ax.axvline(tx, color="#b9772a", ls="--", lw=1.4, label=f"true x = {tx:.2f}")
    ax.scatter([peak], [sims.max()], color="#0d7d88", zorder=3,
               label=f"peak x = {peak:.2f}")
    ax.set_title(f"FPE similarity kernel  (frame {frame})", fontsize=11)
    ax.set_xlabel("queried x (m)"); ax.set_ylabel("similarity")
    ax.legend(fontsize=9)
    return _save(fig, out, "fpe_kernel.png")


def panel_capacity(plt, positions, embeddings, seed, out, dims=(128, 256, 512, 1024, 2048, 4096)):
    """Position exact-frame recall + mean error as hd_dim grows."""
    accs, errs = [], []
    for d in dims:
        c = encode_content(embeddings, hd_dim=d, seed=seed)
        e = ContextEncoders(hd_dim=d, seed=seed + 100, pos_length_scale=1.0, time_length_scale=4.0)
        r = evaluate_position_recall(positions, c, e)
        accs.append(r["exact_frame_acc"]); errs.append(abs(r["mean_pos_err_m"]))

    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    x = np.arange(len(dims))
    bars = ax.bar(x, accs, color="#0d7d88", alpha=0.85, width=0.6)
    ax.set_ylim(0, 1.08); ax.set_xticks(x); ax.set_xticklabels(dims)
    ax.set_xlabel("hd_dim"); ax.set_ylabel("exact-frame recall")
    ax.set_title("bundling capacity: recall vs hypervector width", fontsize=11)
    for b, a, er in zip(bars, accs, errs):
        ax.text(b.get_x() + b.get_width() / 2, a + 0.02, f"{a*100:.1f}%",
                ha="center", va="bottom", fontsize=9)
        ax.text(b.get_x() + b.get_width() / 2, 0.03,
                ("0 m" if er < 1e-6 else f"{er:.2f} m"),
                ha="center", va="bottom", fontsize=8, color="white")
    return _save(fig, out, "capacity_sweep.png")


def panel_evaluate(plt, memory, content, enc, positions, out):
    """Self-recall: query each frame by its own position, show recalled vs true."""
    errors, recalled = [], []
    for n, (x, y) in enumerate(positions):
        residual = memory / enc.ctx_pos(x, y)
        best = int(np.argmax(cleanup(residual, content)))
        recalled.append(best)
        errors.append(float(np.linalg.norm(positions[best] - positions[n])))
    errors = np.array(errors)

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    a0.plot(positions[:, 0], positions[:, 1], "-", color="0.8", lw=1)
    for n, best in enumerate(recalled):
        col = "#2f8f5b" if best == n else "#c0392b"
        a0.plot([positions[n, 0], positions[best, 0]],
                [positions[n, 1], positions[best, 1]], "-", color=col, lw=1.2, alpha=0.8)
    a0.scatter(positions[:, 0], positions[:, 1], s=40, color="#0d7d88",
               edgecolor="white", linewidth=0.6, zorder=3)
    a0.set_title("self-recall: true -> recalled position\n(green = exact, red = miss)", fontsize=11)
    a0.set_xlabel("x (m)"); a0.set_ylabel("y (m)"); a0.set_aspect("equal", "box")

    a1.bar(np.arange(len(errors)), errors, color="#0d7d88", alpha=0.85)
    a1.set_title(f"per-frame position error\nmean {errors.mean():.3f} m · "
                 f"exact {(np.array(recalled) == np.arange(len(recalled))).mean()*100:.1f}%",
                 fontsize=11)
    a1.set_xlabel("frame"); a1.set_ylabel("position error (m)")
    return _save(fig, out, "evaluate.png")


def make_demo_gif(plt, animation, memory, content, enc, positions, gx, gy, ctx_grid, out, fps=4):
    """Animated 'demo': step the queried frame along the walk and watch the
    localization field peak follow the robot -- the upstream demo.mp4 idea."""
    fields = [localization_field(memory, content[n], ctx_grid) for n in range(len(positions))]
    lim = max(np.max(np.abs(f)) for f in fields)

    fig, ax = plt.subplots(figsize=(6.6, 5.4), constrained_layout=True)
    im = ax.imshow(fields[0], origin="lower", cmap="coolwarm", vmin=-lim, vmax=lim,
                   extent=[gx[0], gx[-1], gy[0], gy[-1]], aspect="equal")
    ax.plot(positions[:, 0], positions[:, 1], "-", color="0.4", lw=1, alpha=0.5)
    ax.scatter(positions[:, 0], positions[:, 1], s=12, facecolor="none",
               edgecolor="0.25", linewidth=0.7)
    marker = ax.scatter([positions[0, 0]], [positions[0, 1]], s=200, marker="o",
                        facecolor="none", edgecolor="black", linewidth=2.5, zorder=5)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="similarity")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    title = ax.set_title("", fontsize=11)

    def update(n):
        im.set_data(fields[n])
        marker.set_offsets([[positions[n, 0], positions[n, 1]]])
        title.set_text(f"query by frame {n:>2}'s appearance  ->  map localizes the robot")
        return im, marker, title

    anim = animation.FuncAnimation(fig, update, frames=len(positions), blit=False)
    path = os.path.join(out, "demo.gif")
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    print(f"saved {path}")
    return path


def _save(fig, out, name):
    path = os.path.join(out, name)
    fig.savefig(path, dpi=130)
    print(f"saved {path}")
    return fig


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="plots", help="where PNGs / gif are written")
    ap.add_argument("--n-frames", type=int, default=24)
    ap.add_argument("--hd-dim", type=int, default=2048)
    ap.add_argument("--embed-dim", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frame", type=int, default=8, help="frame used for the single-frame panels")
    ap.add_argument("--grid", type=int, default=48, help="localization grid resolution (x)")
    ap.add_argument("--no-gif", action="store_true", help="skip the animation (faster)")
    ap.add_argument("--show", action="store_true", help="also open interactive windows")
    args = ap.parse_args()

    if not args.show:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    os.makedirs(args.out_dir, exist_ok=True)
    frame = args.frame % args.n_frames

    positions, times, embeddings = make_scene(args.n_frames, args.embed_dim, args.seed)
    content = encode_content(embeddings, hd_dim=args.hd_dim, seed=args.seed)
    enc = ContextEncoders(hd_dim=args.hd_dim, seed=args.seed + 100,
                          pos_length_scale=1.0, time_length_scale=4.0)
    M_pos = build_memory(content, [enc.ctx_pos(x, y) for x, y in positions])

    # localization grid + precomputed context phasors (shared by static + gif)
    gx = np.linspace(-4.0, 4.0, args.grid)
    gy = np.linspace(-3.0, 3.0, max(8, int(args.grid * 6 / 8)))
    print(f"building {len(gy)}x{len(gx)} context grid at hd_dim={args.hd_dim} ...")
    ctx_grid = build_ctx_grid(enc, gx, gy)

    panel_correlations(plt, content, positions, args.out_dir)
    panel_localization(plt, M_pos, content, enc, positions, frame, gx, gy, ctx_grid, args.out_dir)
    panel_kernel(plt, M_pos, content, enc, positions, frame, args.out_dir)
    panel_evaluate(plt, M_pos, content, enc, positions, args.out_dir)
    panel_capacity(plt, positions, embeddings, args.seed, args.out_dir)

    if not args.no_gif:
        print("rendering demo.gif (this is the slow part) ...")
        make_demo_gif(plt, animation, M_pos, content, enc, positions, gx, gy, ctx_grid, args.out_dir)

    print(f"\nall figures written to {os.path.abspath(args.out_dir)}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
