"""Run YOLO over the D455 RGB stream of the Spot Telluride classroom-walk
dataset to get per-timestamp object detections and image embeddings.

    python scripts/classroom/detect_and_embed_classroom.py --out-dir outputs/classroom_detections

Writes two files to --out-dir:
- detections.csv: long format, one row per detected object
  (frame_idx, timestamp_ns, class_id, class_name, confidence, x1, y1, x2, y2
  in pixel coordinates; frames with zero detections contribute no rows).
- embeddings.pt: {"frame_idx": LongTensor, "timestamp_ns": LongTensor,
  "embedding": FloatTensor[N, D]} — the penultimate-layer feature vector
  per frame, D=256 for yolov8n. Same dict shape as eval.py's
  --save-embeddings output (see scripts/eval.py), for downstream use e.g.
  binding into an HD associative memory (scripts/associative_memory.py).

PNG decode and YOLO inference were both benchmarked as fast on this dataset
(~0.9ms/image decode, ~15ms/image detect, ~9ms/image embed on Apple
Silicon/MPS — under a minute total for all ~2478 D455 frames), so images are
read directly from the Hugging Face dataset's local cache with no
intermediate on-disk conversion.

Pass --visualize to also render observations.mp4 to --out-dir: D455 RGB with
detection boxes drawn from detections.csv, the robot's world-frame
position/heading from odometry, and the embedding self-correlation matrix
(from embeddings.pt) with a crosshair marking the current frame in time,
side by side, played back at the dataset's native capture rate. --visualize
can be combined with an existing run (detection is skipped, only the video
is (re)built). Detection/embeddings and the video are regenerated
independently: --force redoes detection+embeddings, --viz-force redoes
observations.mp4; by default each is skipped if its output already exists.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from ultralytics import YOLO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

REPO = "lorinachey/spot-telluride-workshop-dataset"

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BLUE = "#2a78d6"
RED = "#e34948"

# Detection boxes span ~40 COCO classes in a single frame -- far more than a
# legend can hold -- so each box carries its own text label and just needs a
# color that's stable per class and visually distinct from its neighbors,
# not a fixed categorical hue order.
CLASS_CMAP = plt.get_cmap("tab20")


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_detect_and_embed(model: YOLO, images: list[np.ndarray], device: str, conf: float):
    """One batch -> (per-image detection rows, per-image embedding vectors).

    Two forward passes (detect then embed): passing `embed=` to YOLO.predict
    short-circuits the forward pass to return raw features instead of
    running the detection head, so a single call can't produce both.
    """
    detections = model.predict(images, device=device, conf=conf, verbose=False)
    embeddings = model.embed(images, device=device, verbose=False)
    return detections, embeddings


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


def load_detections_by_frame(det_path: Path) -> dict[int, pd.DataFrame]:
    df = pd.read_csv(det_path)
    return {int(frame_idx): group for frame_idx, group in df.groupby("frame_idx")}


def draw_boxes(ax, prior_artists: list, frame_dets: pd.DataFrame | None) -> list:
    """Remove the previous frame's boxes/labels and draw this frame's, returning the new artists."""
    for artist in prior_artists:
        artist.remove()
    if frame_dets is None:
        return []
    new_artists = []
    for _, det in frame_dets.iterrows():
        color = CLASS_CMAP(int(det["class_id"]) % CLASS_CMAP.N)
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        box = Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=1.6)
        ax.add_patch(box)
        label = ax.text(x1, max(y1 - 4, 0), f'{det["class_name"]} {det["confidence"]:.2f}',
                        color="white", fontsize=7, va="bottom", ha="left",
                        bbox=dict(facecolor=color, edgecolor="none", pad=1.5, alpha=0.9))
        new_artists += [box, label]
    return new_artists


def build_visualization(rgb, det_path: Path, emb_path: Path, out_path: Path, repo: str,
                        odom_config: str, n_frames: int, fps: float | None) -> None:
    """Render observations.mp4: D455 RGB + detection boxes, world-frame
    position/heading, and the embedding self-correlation matrix with a
    crosshair marking the current frame, side by side -- one video frame per
    D455 RGB frame, at the dataset's native capture rate unless fps is given.
    """
    print(f"loading {odom_config} for visualization...")
    odom_ds = load_dataset(repo, odom_config, split="train").sort("timestamp_ns")
    odom_ts = np.array(odom_ds["timestamp_ns"])
    odom_x, odom_y = np.array(odom_ds["x"]), np.array(odom_ds["y"])
    odom_yaw = quat_to_yaw(np.array(odom_ds["qx"]), np.array(odom_ds["qy"]),
                           np.array(odom_ds["qz"]), np.array(odom_ds["qw"]))

    print(f"loading detections from {det_path}...")
    det_by_frame = load_detections_by_frame(det_path)

    print(f"loading embeddings from {emb_path}...")
    # Positional slice, not a frame_idx lookup: both detect_and_embed's write
    # loop and the sorted RGB dataset append/index in the same timestamp
    # order, so row i of embeddings.pt is frame i here too.
    embeddings = torch.load(emb_path)["embedding"].numpy()[:n_frames]
    corr = np.corrcoef(embeddings)
    corr_vmin, corr_vmax = corr.min(), corr.max()

    rgb_ts = np.array(rgb["timestamp_ns"][:n_frames])
    if fps is None:
        duration_s = (rgb_ts[-1] - rgb_ts[0]) / 1e9
        fps = (n_frames - 1) / duration_s if duration_s > 0 else 30.0
    print(f"rendering {n_frames} frames at {fps:.2f} fps -> {out_path}")

    fig = plt.figure(figsize=(16, 6), facecolor=SURFACE, constrained_layout=True)
    gs = fig.add_gridspec(1, 3)
    ax_rgb = fig.add_subplot(gs[0, 0])
    ax_pos = fig.add_subplot(gs[0, 1])
    ax_corr = fig.add_subplot(gs[0, 2])

    rgb_im = ax_rgb.imshow(np.asarray(rgb[0]["image"].convert("RGB")))
    ax_rgb.axis("off")
    ax_rgb.set_title("D455 RGB + detections", color=INK_PRIMARY, fontsize=11, pad=8)
    box_artists: list = []

    ax_pos.plot(odom_x, odom_y, color=BLUE, linewidth=2, label="path", zorder=2)
    cur_point = ax_pos.scatter([odom_x[0]], [odom_y[0]], color=RED, s=60, zorder=3, label="current")
    arrow_len = 0.08 * max(np.ptp(odom_x), np.ptp(odom_y), 1e-3)
    # quiver (not annotate) so the arrow is a normal data artist that can be
    # moved in place each frame via set_offsets/set_UVC.
    heading = ax_pos.quiver([odom_x[0]], [odom_y[0]], [arrow_len], [0.0],
                            color=RED, angles="xy", scale_units="xy", scale=1, width=0.01, zorder=4)
    ax_pos.set_aspect("equal")
    ax_pos.margins(0.15)
    ax_pos.set_facecolor(SURFACE)
    ax_pos.set_title("Position & heading (world frame)", color=INK_PRIMARY, fontsize=11, pad=8)
    ax_pos.tick_params(colors=INK_MUTED, labelsize=7)
    for spine in ax_pos.spines.values():
        spine.set_color(GRID)
    ax_pos.grid(color=GRID, linewidth=0.6)
    ax_pos.set_axisbelow(True)
    ax_pos.set_xlabel("x (m)", color=INK_MUTED, fontsize=8)
    ax_pos.set_ylabel("y (m)", color=INK_MUTED, fontsize=8)
    legend = ax_pos.legend(loc="best", fontsize=8, facecolor=SURFACE, edgecolor=GRID)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    corr_im = ax_corr.imshow(corr, cmap="viridis", vmin=corr_vmin, vmax=corr_vmax)
    ax_corr.set_facecolor(SURFACE)
    ax_corr.set_title("Embedding self-correlation", color=INK_PRIMARY, fontsize=11, pad=8)
    ax_corr.set_xlabel("frame index", color=INK_MUTED, fontsize=8)
    ax_corr.set_ylabel("frame index", color=INK_MUTED, fontsize=8)
    ax_corr.tick_params(colors=INK_MUTED, labelsize=7)
    for spine in ax_corr.spines.values():
        spine.set_color(GRID)
    cbar = fig.colorbar(corr_im, ax=ax_corr, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson correlation", color=INK_SECONDARY, fontsize=8)
    cbar.ax.tick_params(colors=INK_MUTED, labelsize=6)
    # Crosshair over the current frame's row/column -- its intersection is
    # "now"; a bright cell where it crosses an off-diagonal band means the
    # robot's current view correlates with a moment earlier in the walk
    # (e.g. a revisited spot in the room).
    corr_hline = ax_corr.axhline(0, color=RED, linewidth=1.2, zorder=5)
    corr_vline = ax_corr.axvline(0, color=RED, linewidth=1.2, zorder=5)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=fps, metadata={"title": "classroom RGBD observations"})
    with writer.saving(fig, str(out_path), dpi=120):
        for i in range(n_frames):
            row = rgb[i]
            ts = row["timestamp_ns"]

            rgb_im.set_data(np.asarray(row["image"].convert("RGB")))
            box_artists = draw_boxes(ax_rgb, box_artists, det_by_frame.get(int(row["frame_idx"])))

            odom_i = nearest_index(odom_ts, ts)
            cur_point.set_offsets([[odom_x[odom_i], odom_y[odom_i]]])
            heading.set_offsets([[odom_x[odom_i], odom_y[odom_i]]])
            heading.set_UVC(arrow_len * np.cos(odom_yaw[odom_i]), arrow_len * np.sin(odom_yaw[odom_i]))

            corr_hline.set_ydata([i, i])
            corr_vline.set_xdata([i, i])

            fig.suptitle(f"frame {i + 1}/{n_frames}   t={(ts - rgb_ts[0]) / 1e9:.2f}s",
                        color=INK_SECONDARY, fontsize=10)
            writer.grab_frame()

            if (i + 1) % 200 == 0 or i + 1 == n_frames:
                print(f"[{i + 1}/{n_frames}] rendered")

    plt.close(fig)
    print(f"visualization written to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default=REPO)
    parser.add_argument("--d455-rgb-config", default="rgb_d455")
    parser.add_argument("--out-dir", default="outputs/classroom_detections")
    parser.add_argument("--yolo-model", default="yolov8n.pt",
                        help="ultralytics checkpoint name or path to a custom .pt")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--conf", type=float, default=0.25, help="detection confidence threshold")
    parser.add_argument("--limit", type=int, default=None, help="only process the first N frames")
    parser.add_argument("--force", action="store_true",
                        help="regenerate even if detections.csv and embeddings.pt already exist")
    parser.add_argument("--visualize", action="store_true",
                        help="also render observations.mp4 (D455 RGB + detections, robot "
                             "position/heading, embedding self-correlation) to --out-dir")
    parser.add_argument("--viz-out", default=None,
                        help="output video path (default: <out-dir>/observations.mp4)")
    parser.add_argument("--viz-fps", type=float, default=None,
                        help="output video frame rate (default: matches the dataset's native "
                             "capture rate, i.e. real-time playback)")
    parser.add_argument("--viz-force", action="store_true",
                        help="rebuild observations.mp4 even if it already exists at the target path")
    parser.add_argument("--odom-config", default="odometry_lio_sam")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    det_path, emb_path = out_dir / "detections.csv", out_dir / "embeddings.pt"
    need_detect = args.force or not (det_path.exists() and emb_path.exists())
    if not need_detect:
        print(f"{det_path} and {emb_path} already exist, skipping detection (pass --force to regenerate)")

    viz_out = Path(args.viz_out) if args.viz_out else out_dir / "observations.mp4"
    need_visualize = args.visualize and (args.viz_force or not viz_out.exists())
    if args.visualize and not need_visualize:
        print(f"{viz_out} already exists, skipping visualization (pass --viz-force to regenerate)")

    rgb, n_frames = None, None
    if need_detect or need_visualize:
        print(f"loading {args.repo} ({args.d455_rgb_config})...")
        rgb = load_dataset(args.repo, args.d455_rgb_config, split="train").sort("timestamp_ns")
        n_frames = len(rgb) if args.limit is None else min(args.limit, len(rgb))

    if need_detect:
        device = resolve_device(args.device)
        print(f"loading {args.yolo_model} on device={device}...")
        model = YOLO(args.yolo_model)

        det_rows = []
        frame_idxs, timestamps, embeds = [], [], []
        for start in range(0, n_frames, args.batch_size):
            end = min(start + args.batch_size, n_frames)
            batch = rgb[start:end]
            images = [np.asarray(img) for img in batch["image"]]

            detections, embeddings = run_detect_and_embed(model, images, device, args.conf)
            for j, (result, embedding) in enumerate(zip(detections, embeddings)):
                frame_idx, ts = batch["frame_idx"][j], batch["timestamp_ns"][j]
                boxes = result.boxes
                for k in range(len(boxes)):
                    x1, y1, x2, y2 = boxes.xyxy[k].tolist()
                    cls_id = int(boxes.cls[k])
                    det_rows.append({"frame_idx": frame_idx, "timestamp_ns": ts, "class_id": cls_id,
                                     "class_name": model.names[cls_id], "confidence": float(boxes.conf[k]),
                                     "x1": x1, "y1": y1, "x2": x2, "y2": y2})
                frame_idxs.append(frame_idx)
                timestamps.append(ts)
                embeds.append(embedding.cpu())

            print(f"[{end}/{n_frames}] {len(det_rows)} detections so far")

        pd.DataFrame(det_rows).to_csv(det_path, index=False)
        print(f"detections written to {det_path} ({len(det_rows)} rows)")

        torch.save({"frame_idx": torch.tensor(frame_idxs, dtype=torch.long),
                   "timestamp_ns": torch.tensor(timestamps, dtype=torch.long),
                   "embedding": torch.stack(embeds)}, emb_path)
        print(f"embeddings written to {emb_path} ({len(embeds)} frames)")

    if need_visualize:
        build_visualization(rgb, det_path, emb_path, viz_out, args.repo, args.odom_config,
                            n_frames, args.viz_fps)


if __name__ == "__main__":
    main()
