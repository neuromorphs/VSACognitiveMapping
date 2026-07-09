"""Load the Spot Telluride classroom-walk dataset and save aligned
multi-camera / LiDAR / position-heading figures.

    python scripts/classroom/load_and_view_classroom.py --out-dir outputs/classroom_frames

One figure is written per D455 RGB frame — the highest-rate image stream
(~2478 frames over the ~80s recording). Every other stream (frontleft,
frontright, D455 depth, LiDAR, odometry) is matched in by nearest timestamp,
so sparser streams (LiDAR: ~21 scans, Spot cameras: ~743 frames each) repeat
across several consecutive output figures until their next sample arrives.

Each figure is a 2x4 grid: frontleft / frontright / D455 RGB / D455 depth on
top, top-down LiDAR (sensor frame) and world-frame position/heading split
across the bottom.
"""

import argparse
from pathlib import Path

import numpy as np
from datasets import load_dataset
from matplotlib.colors import LinearSegmentedColormap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = "lorinachey/spot-telluride-workshop-dataset"

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BLUE = "#2a78d6"
RED = "#e34948"

# Sequential single-hue ramp (light -> dark blue) for magnitude encodings
# (LiDAR height, depth range).
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281"])
SEQ_BLUE.set_bad(GRID)


def parse_pcd_xyz(pcd_bytes: bytes) -> np.ndarray:
    """ASCII .pcd bytes -> (N, 3) float32 array of the x y z fields."""
    lines = pcd_bytes.decode("ascii", errors="ignore").splitlines()
    data_line = next(l for l in lines if l.startswith("DATA"))
    if data_line.split()[-1] != "ascii":
        raise ValueError(f"only ASCII .pcd files are supported, got: {data_line!r}")
    start = lines.index(data_line) + 1
    rows = [line.split() for line in lines[start:] if line.strip()]
    return np.array(rows, dtype=np.float32)


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


def load_sorted(repo: str, config: str):
    return load_dataset(repo, config, split="train").sort("timestamp_ns")


def style_axes(ax, title: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK_PRIMARY, fontsize=11, pad=8)
    ax.tick_params(colors=INK_MUTED, labelsize=7)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def show_image(ax, image, title: str) -> None:
    ax.imshow(image.convert("RGB"))
    ax.set_title(title, color=INK_PRIMARY, fontsize=11, pad=8)
    ax.axis("off")


def show_depth(ax, fig, depth_image, title: str, max_range_m: float) -> None:
    """16-bit depth (millimeters, 0 = no return) -> colorized meters image."""
    depth_m = np.asarray(depth_image, dtype=np.float32) / 1000.0
    depth_m = np.ma.masked_where(depth_m <= 0, np.clip(depth_m, 0, max_range_m))
    im = ax.imshow(depth_m, cmap=SEQ_BLUE, vmin=0, vmax=max_range_m)
    ax.set_title(title, color=INK_PRIMARY, fontsize=11, pad=8)
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("depth (m)", color=INK_SECONDARY, fontsize=8)
    cbar.ax.tick_params(colors=INK_MUTED, labelsize=6)


def plot_frame(frontleft_img, frontright_img, d455_rgb_img, d455_depth_img,
               points_xyz: np.ndarray, odom_x: np.ndarray, odom_y: np.ndarray,
               odom_yaw: np.ndarray, cur_i: int, out_path: Path, lidar_max_points: int,
               lidar_max_range: float, depth_max_range: float) -> None:
    fig = plt.figure(figsize=(18, 9), facecolor=SURFACE, constrained_layout=True)
    gs = fig.add_gridspec(2, 4)

    show_image(fig.add_subplot(gs[0, 0]), frontleft_img, "Spot frontleft")
    show_image(fig.add_subplot(gs[0, 1]), frontright_img, "Spot frontright")
    show_image(fig.add_subplot(gs[0, 2]), d455_rgb_img, "D455 RGB")
    show_depth(fig.add_subplot(gs[0, 3]), fig, d455_depth_img, "D455 depth", depth_max_range)

    ax_lidar = fig.add_subplot(gs[1, 0:2])
    in_range = np.hypot(points_xyz[:, 0], points_xyz[:, 1]) < lidar_max_range
    points_xyz = points_xyz[in_range]
    if len(points_xyz) > lidar_max_points:
        idx = np.random.default_rng(0).choice(len(points_xyz), lidar_max_points, replace=False)
        points_xyz = points_xyz[idx]
    sc = ax_lidar.scatter(points_xyz[:, 0], points_xyz[:, 1], c=points_xyz[:, 2],
                          cmap=SEQ_BLUE, s=1.5, linewidths=0, rasterized=True)
    ax_lidar.set_aspect("equal")
    # Scans are published in the sensor/base frame (always centered near the
    # robot), unlike the world-frame odometry on the right, so label it as such.
    style_axes(ax_lidar, "LiDAR scan (sensor frame, top-down)")
    ax_lidar.set_xlabel("x (m, sensor frame)", color=INK_MUTED, fontsize=8)
    ax_lidar.set_ylabel("y (m, sensor frame)", color=INK_MUTED, fontsize=8)
    cbar = fig.colorbar(sc, ax=ax_lidar, fraction=0.046, pad=0.04)
    cbar.set_label("height z (m)", color=INK_SECONDARY, fontsize=8)
    cbar.ax.tick_params(colors=INK_MUTED, labelsize=6)

    ax_pos = fig.add_subplot(gs[1, 2:4])
    ax_pos.plot(odom_x, odom_y, color=BLUE, linewidth=2, label="path", zorder=2)
    ax_pos.scatter([odom_x[cur_i]], [odom_y[cur_i]], color=RED, s=60, zorder=3, label="current")
    # quiver (not annotate) so the arrow is a normal data artist: annotate
    # silently drops the whole arrow when its tip falls outside the
    # autoscaled view, which a long heading arrow easily does.
    arrow_len = 0.08 * max(np.ptp(odom_x), np.ptp(odom_y), 1e-3)
    ax_pos.quiver(odom_x[cur_i], odom_y[cur_i],
                 arrow_len * np.cos(odom_yaw[cur_i]), arrow_len * np.sin(odom_yaw[cur_i]),
                 color=RED, angles="xy", scale_units="xy", scale=1, width=0.01, zorder=4)
    ax_pos.set_aspect("equal")
    ax_pos.margins(0.15)
    style_axes(ax_pos, "Position & heading (world frame)")
    ax_pos.set_xlabel("x (m)", color=INK_MUTED, fontsize=8)
    ax_pos.set_ylabel("y (m)", color=INK_MUTED, fontsize=8)
    legend = ax_pos.legend(loc="best", fontsize=8, facecolor=SURFACE, edgecolor=GRID)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default=REPO)
    parser.add_argument("--frontleft-config", default="rgb_spot_frontleft")
    parser.add_argument("--frontright-config", default="rgb_spot_frontright")
    parser.add_argument("--d455-rgb-config", default="rgb_d455")
    parser.add_argument("--d455-depth-config", default="depth_d455")
    parser.add_argument("--pointcloud-config", default="pointclouds")
    parser.add_argument("--odom-config", default="odometry_lio_sam")
    parser.add_argument("--out-dir", default="outputs/classroom_frames")
    parser.add_argument("--limit", type=int, default=None,
                        help="only process the first N D455 RGB frames (the anchor stream)")
    parser.add_argument("--lidar-max-points", type=int, default=20000,
                        help="random subsample cap per LiDAR scan, for plotting speed")
    parser.add_argument("--lidar-max-range", type=float, default=20.0,
                        help="drop LiDAR points beyond this many meters from the sensor origin")
    parser.add_argument("--depth-max-range", type=float, default=8.0,
                        help="D455 depth colorbar ceiling, in meters")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading {args.repo} (frontleft, frontright, d455 rgb+depth, pointclouds, odometry)...")
    frontleft = load_sorted(args.repo, args.frontleft_config)
    frontright = load_sorted(args.repo, args.frontright_config)
    d455_rgb = load_sorted(args.repo, args.d455_rgb_config)
    d455_depth = load_sorted(args.repo, args.d455_depth_config)
    pcd = load_sorted(args.repo, args.pointcloud_config)
    odom = load_sorted(args.repo, args.odom_config)

    frontleft_ts = np.array(frontleft["timestamp_ns"])
    frontright_ts = np.array(frontright["timestamp_ns"])
    d455_depth_ts = np.array(d455_depth["timestamp_ns"])
    pcd_ts = np.array(pcd["timestamp_ns"])
    pcd_points = [parse_pcd_xyz(pcd[i]["pcd_bytes"]) for i in range(len(pcd))]  # only ~21, cheap to preload
    odom_ts = np.array(odom["timestamp_ns"])
    odom_x, odom_y = np.array(odom["x"]), np.array(odom["y"])
    odom_yaw = quat_to_yaw(np.array(odom["qx"]), np.array(odom["qy"]),
                           np.array(odom["qz"]), np.array(odom["qw"]))

    n_frames = len(d455_rgb) if args.limit is None else min(args.limit, len(d455_rgb))
    for i in range(n_frames):
        row = d455_rgb[i]
        ts = row["timestamp_ns"]
        fl_i = nearest_index(frontleft_ts, ts)
        fr_i = nearest_index(frontright_ts, ts)
        depth_i = nearest_index(d455_depth_ts, ts)
        pcd_i = nearest_index(pcd_ts, ts)
        odom_i = nearest_index(odom_ts, ts)

        out_path = out_dir / f"frame_{i:04d}_t{ts}.png"
        plot_frame(frontleft[fl_i]["image"], frontright[fr_i]["image"], row["image"],
                  d455_depth[depth_i]["depth"], pcd_points[pcd_i], odom_x, odom_y, odom_yaw,
                  odom_i, out_path, args.lidar_max_points, args.lidar_max_range,
                  args.depth_max_range)
        print(f"[{i + 1}/{n_frames}] wrote {out_path}")

    print(f"done: {n_frames} figures written to {out_dir}")


if __name__ == "__main__":
    main()
