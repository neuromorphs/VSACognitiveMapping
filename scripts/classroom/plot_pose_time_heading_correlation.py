"""Independently bind the robot's position, heading, and time to basis
phasors, and plot each channel's own pairwise correlation structure.

    python scripts/classroom/plot_pose_time_heading_correlation.py

Loads D455 RGB timestamps/frame_idx (the anchor stream detections.csv and
embeddings.pt are indexed by -- see scripts/classroom/detect_and_embed_classroom.py)
and odometry_lio_sam (x, y, heading), nearest-timestamp aligned so there's
exactly one position/heading/time sample per D455 frame, matching
embeddings.pt's frame ordering for a later step that binds these context
codes to the YOLO content vectors (scripts/associative_memory.py's phase 2
does this for the JEPA dataset; this is the classroom-dataset equivalent of
its phase 2b encode_time/encode_position/encode_heading).

Each channel gets its own basis phasor(s) via src/vsa_cognitive_mapping/vsa.py:
- position (x, y): FPE against two bases, bound together per frame --
  (Bx**x) * (By**y) -- so nearby (x, y) points correlate.
- heading (yaw): FPE against a *circular-safe* base (Phasor(circular=True)),
  since yaw wraps at +/-pi and an ordinary random-phase base does not.
- time (frame index): FPE against a single base -- by construction this
  always just measures |frame_i - frame_j|, a "boring" reference to
  contrast against position (which can show off-diagonal revisit bands)
  and heading (which tracks facing direction, not location).

No binding *across* channels happens here -- each is encoded and correlated
independently, which is the point: to see what structure each axis
contributes on its own before they get bound together into one shared trace.

Writes outputs/classroom_detections/pose_time_heading_correlation.png: one
row of 3 pairwise-correlation heatmaps (phasor_correlation_matrix), position
on the left, heading in the middle, time on the right.
"""

import argparse
from pathlib import Path

import numpy as np
from datasets import load_dataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.colors as mcolors  # noqa: E402

from vsa_cognitive_mapping.vsa import Phasor, orthogonality_score, phasor_correlation_matrix

REPO = "lorinachey/spot-telluride-workshop-dataset"

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"

# Signed [-1, 1] correlation -> diverging, same validated pair
# scripts/associative_memory.py, scripts/encoder_sweep.py, and
# scripts/classroom/plot_embedding_correlation.py each define locally.
DIVERGING_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#2a78d6", "#f0efec", "#e34948"])


def quat_to_yaw(qx: np.ndarray, qy: np.ndarray, qz: np.ndarray, qw: np.ndarray) -> np.ndarray:
    """Yaw (rotation about z, radians) from unit quaternions."""
    return np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy ** 2 + qz ** 2))


def nearest_index(sorted_ts: np.ndarray, query_ts: int) -> int:
    """Index into ascending sorted_ts closest to query_ts."""
    i = np.searchsorted(sorted_ts, query_ts)
    if i == 0:
        return 0
    if i == len(sorted_ts):
        return len(sorted_ts) - 1
    before, after = sorted_ts[i - 1], sorted_ts[i]
    return int(i - 1 if query_ts - before <= after - query_ts else i)


def encode_time(row_idx: np.ndarray, hd_dim: int, seed: int, length_scale: float) -> np.ndarray:
    base = Phasor(dim=hd_dim, seed=seed)
    return np.stack([(base ** float(t / length_scale)).values for t in row_idx])


def encode_position(xy: np.ndarray, hd_dim: int, x_seed: int, y_seed: int, length_scale: float) -> np.ndarray:
    Bx = Phasor(dim=hd_dim, seed=x_seed)
    By = Phasor(dim=hd_dim, seed=y_seed)
    return np.stack([
        ((Bx ** float(x / length_scale)) * (By ** float(y / length_scale))).values
        for x, y in xy
    ])


def encode_heading(yaw: np.ndarray, hd_dim: int, seed: int, max_freq: int) -> np.ndarray:
    """FPE with `circular=True`, not an ordinary Phasor base -- a plain
    random-phase base isn't circular-safe (`base**angle` doesn't return to
    the same phasor at `angle + 2*pi`), see `Phasor.__init__`'s `circular`
    branch."""
    base = Phasor(dim=hd_dim, seed=seed, circular=True, max_freq=max_freq)
    return np.stack([(base ** float(a)).values for a in yaw])


def plot_channel_correlations(pos_corr: np.ndarray, heading_corr: np.ndarray, time_corr: np.ndarray,
                              out_path: Path) -> Path:
    # Each panel gets its own data-driven color scale (not a shared [-1, 1]):
    # unlike a raw-vs-projected fidelity comparison, these three aren't
    # meant to be compared cell-for-cell against each other -- each is an
    # independent structural view of a different channel, so a fixed shared
    # range would just wash out whichever channel has the tightest spread.
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), facecolor=SURFACE, constrained_layout=True)
    panels = (("position (x, y)", pos_corr), ("heading (yaw)", heading_corr), ("time (frame index)", time_corr))
    for ax, (title, corr) in zip(axes, panels):
        vmin, vmax = float(corr.min()), float(corr.max())
        im = ax.imshow(corr, cmap=DIVERGING_CMAP, vmin=vmin, vmax=vmax)
        ax.set_facecolor(SURFACE)
        ax.set_title(title, color=INK_PRIMARY, fontsize=11, pad=8)
        ax.set_xlabel("frame index", color=INK_MUTED, fontsize=9, labelpad=6)
        ax.set_ylabel("frame index", color=INK_MUTED, fontsize=9, labelpad=8)
        ax.tick_params(colors=INK_MUTED, labelsize=7)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("phasor similarity", color=INK_SECONDARY, fontsize=8)
        cbar.ax.tick_params(colors=INK_MUTED, labelsize=7)

    fig.suptitle("Independently bound context phasors -- pairwise self-correlation",
                color=INK_PRIMARY, fontsize=13)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default=REPO)
    parser.add_argument("--d455-rgb-config", default="rgb_d455")
    parser.add_argument("--odom-config", default="odometry_lio_sam")
    parser.add_argument("--out-path", default="outputs/classroom_detections/pose_time_heading_correlation.png")
    parser.add_argument("--limit", type=int, default=None, help="only use the first N D455 frames")
    parser.add_argument("--hd-dim", type=int, default=256)
    parser.add_argument("--time-seed", type=int, default=42)
    parser.add_argument("--x-seed", type=int, default=1)
    parser.add_argument("--y-seed", type=int, default=2)
    parser.add_argument("--yaw-seed", type=int, default=3)
    parser.add_argument("--yaw-max-freq", type=int, default=3,
                        help="circular-safe base frequency cap, see Phasor's _MAX_CIRCULAR_FREQ")
    parser.add_argument("--pos-length-scale", type=float, default=1.0,
                        help="FPE kernel width for position, in meters -- e.g. 1.0 means "
                             "roughly a 1m radius of high similarity around each point")
    parser.add_argument("--time-length-scale", type=float, default=50.0,
                        help="FPE kernel width for time, in frames -- by construction this "
                             "only sets how wide the diagonal band is, since frame-index "
                             "distance is the same regardless of position/heading")
    args = parser.parse_args()

    print(f"loading {args.repo} ({args.d455_rgb_config}, {args.odom_config})...")
    rgb = load_dataset(args.repo, args.d455_rgb_config, split="train").sort("timestamp_ns")
    odom = load_dataset(args.repo, args.odom_config, split="train").sort("timestamp_ns")

    n_frames = len(rgb) if args.limit is None else min(args.limit, len(rgb))
    rgb_ts = np.array(rgb["timestamp_ns"][:n_frames])
    # Row position, not the dataset's own frame_idx field: frame_idx is not
    # monotonic once sorted by timestamp_ns (it jumps, e.g. ...2477, 472,
    # 473... partway through this dataset) while timestamp_ns is -- so
    # row position is what actually tracks elapsed time in capture order.
    row_idx = np.arange(n_frames)

    odom_ts = np.array(odom["timestamp_ns"])
    odom_x, odom_y = np.array(odom["x"]), np.array(odom["y"])
    odom_yaw = quat_to_yaw(np.array(odom["qx"]), np.array(odom["qy"]),
                           np.array(odom["qz"]), np.array(odom["qw"]))
    odom_i = np.array([nearest_index(odom_ts, ts) for ts in rgb_ts])
    x, y, yaw = odom_x[odom_i], odom_y[odom_i], odom_yaw[odom_i]
    print(f"aligned {n_frames} D455 frames to nearest odometry samples")

    pos_codes = encode_position(np.stack([x, y], axis=1), args.hd_dim, args.x_seed, args.y_seed,
                                args.pos_length_scale)
    heading_codes = encode_heading(yaw, args.hd_dim, args.yaw_seed, args.yaw_max_freq)
    time_codes = encode_time(row_idx, args.hd_dim, args.time_seed, args.time_length_scale)

    pos_corr = phasor_correlation_matrix(pos_codes)
    heading_corr = phasor_correlation_matrix(heading_codes)
    time_corr = phasor_correlation_matrix(time_codes)

    for name, corr in (("position", pos_corr), ("heading", heading_corr), ("time", time_corr)):
        print(f"[{name}] orthogonality={orthogonality_score(corr):.3f}")

    out_path = Path(args.out_path)
    plot_channel_correlations(pos_corr, heading_corr, time_corr, out_path)
    print(f"pose/heading/time correlation plot written to {out_path}")


if __name__ == "__main__":
    main()
