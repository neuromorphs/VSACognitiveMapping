"""Run YOLO and/or DINOv2 over the Freiburg (ICL-NUIM "living room traj0")
RGB PNG sequence to get per-frame object detections and image embeddings --
the same shape of output as scripts/classroom/detect_and_embed_classroom.py,
but reading local files instead of a HuggingFace dataset.

    python scripts/freiburg/detect_and_embed_freiburg.py --out-dir outputs/freiburg_detections
    python scripts/freiburg/detect_and_embed_freiburg.py --out-dir outputs/freiburg_detections --embedding-model dino

data/frieburg/rgb/<frame>.png (frame = 0..1508, 640x480) is this dataset's
only input this script touches -- depth/ is a separate render and out of
scope here (RGB-only, per this comparison's goal). There's no real capture
clock (this is a synthetic/rendered sequence, unlike the classroom D455
recording), so timestamp_ns is synthesized at a nominal --fps (default 30,
the standard ICL-NUIM rate) rather than read from metadata; every downstream
duration/fps computation that consumes timestamp_ns (this script's
--visualize, and scripts/freiburg/freiburg_associative_memory.py's `demo`)
is therefore accurate relative to that assumed rate, not a measured one.

Writes up to two files to --out-dir:
- detections.csv: long format, one row per detected object (frame_idx,
  timestamp_ns, class_id, class_name, confidence, x1, y1, x2, y2 in pixel
  coordinates; frames with zero detections contribute no rows). Produced by
  YOLO only -- shared/backend-independent, so it's named the same regardless
  of --embedding-model. Controlled by --detect: 'auto' (default) runs it iff
  --embedding-model yolo (DINOv2 has no detection head); 'on'/'off' force it
  either way.
- embeddings.pt (or embeddings_dino.pt when --embedding-model dino): {
  "frame_idx": LongTensor, "timestamp_ns": LongTensor, "embedding":
  FloatTensor[N, D]} -- one feature vector per frame: the penultimate-layer
  YOLO feature (D=256 for yolov8n) or the frozen DINOv2 CLS embedding
  (D=384 for the default dinov2_vits14), selected by --embedding-model. Same
  dict shape as scripts/classroom/detect_and_embed_classroom.py's output,
  for downstream use e.g. binding into an HD associative memory
  (scripts/freiburg/freiburg_associative_memory.py).

Detection and embedding are independently cacheable: each is (re)computed
only if its own output file is missing or --force is passed, regardless of
whether the other one already exists.

Pass --visualize to also render observations.mp4 (or observations_dino.mp4)
to --out-dir: RGB with detection boxes drawn from detections.csv (if
present), the ground-truth camera trajectory/heading (from
data/frieburg/livingRoom0.gt.freiburg), and the embedding self-correlation
matrix (from embeddings.pt) with a crosshair marking the current frame in
time, side by side, played back at --fps. Frame 0 has no ground-truth pose
in this dataset (poses start at frame 1) and is skipped in the video only.
--visualize can be combined with an existing run (detection/embedding are
skipped, only the video is (re)built). --viz-force redoes the video; by
default it's skipped if it already exists.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from vsa_cognitive_mapping.data import IMAGENET_MEAN, IMAGENET_STD
from vsa_cognitive_mapping.encoder import load_backbone

RGB_DIR = "data/frieburg/rgb"
GT_PATH = "data/frieburg/livingRoom0.gt.freiburg"

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


def with_suffix_for_model(path: Path, embedding_model: str) -> Path:
    """Insert a _dino suffix before the extension when embedding_model is
    'dino', so both backends' default outputs coexist under the same
    --out-dir without clobbering each other; 'yolo' leaves the path
    unchanged (backward compatible with every existing default filename)."""
    return path if embedding_model == "yolo" else path.with_stem(path.stem + "_dino")


def list_frames(rgb_dir: Path) -> list[tuple[int, Path]]:
    """(frame_idx, path) pairs sorted numerically by filename stem (0.png,
    1.png, ..., 1508.png) -- a plain lexicographic sort would put 10.png
    before 2.png."""
    frames = [(int(p.stem), p) for p in rgb_dir.glob("*.png")]
    return sorted(frames, key=lambda t: t[0])


def load_yolo_model(yolo_model: str) -> YOLO:
    """Lazily import ultralytics -- only called when detection or a YOLO
    embedding pass actually runs, so a pure-DINO run (--embedding-model dino
    with detection off) never requires the `ultralytics` package."""
    from ultralytics import YOLO
    return YOLO(yolo_model)


def run_detect(model: YOLO, images: list[np.ndarray], device: str, conf: float):
    """One batch -> per-image detection Results."""
    return model.predict(images, device=device, conf=conf, verbose=False)


def run_embed_yolo(model: YOLO, images: list[np.ndarray], device: str) -> list[torch.Tensor]:
    """One batch -> per-image penultimate-layer feature vectors (256-dim
    for yolov8n) via passing `embed=` to YOLO.predict, which short-circuits
    the forward pass to return raw features instead of running the
    detection head -- this is why detection and embedding are two separate
    forward passes rather than one call producing both."""
    return model.embed(images, device=device, verbose=False)


DINO_PATCH_SIZE = 14  # all public dinov2_*14 hub entrypoints use patch size 14


def preprocess_dino_batch(images: list[np.ndarray], img_size: int) -> torch.Tensor:
    """List of HxWx3 uint8 RGB arrays -> float32 (B, 3, img_size, img_size)
    tensor, ImageNet-normalized -- same resize/normalize formula as
    vsa_cognitive_mapping.data.load_image, starting from an already-decoded
    array."""
    resized = [
        np.asarray(Image.fromarray(img).resize((img_size, img_size), Image.Resampling.BILINEAR),
                  dtype=np.float32) / 255.0
        for img in images
    ]
    batch = (np.stack(resized) - IMAGENET_MEAN) / IMAGENET_STD  # (B, H, W, 3)
    return torch.from_numpy(batch).permute(0, 3, 1, 2).contiguous()


@torch.no_grad()
def run_embed_dino(backbone: torch.nn.Module, images: list[np.ndarray], device: str,
                   img_size: int) -> list[torch.Tensor]:
    """One batch -> per-image (D,) CLS embeddings from the frozen DINOv2
    backbone. A raw backbone forward returns one batched (B, D) tensor
    (unlike YOLO.embed(), which already returns a list of per-image
    tensors) -- unbind(0) here normalizes that so the write loop downstream
    doesn't need to know which backend produced the embeddings."""
    batch = preprocess_dino_batch(images, img_size).to(device)
    return list(backbone(batch).cpu().unbind(0))


def load_gt_poses(gt_path: Path) -> dict[str, np.ndarray]:
    """data/frieburg/livingRoom0.gt.freiburg -> {frame_id, tx, ty, tz, qx,
    qy, qz, qw} arrays, one row per line (`frame_id tx ty tz qx qy qz qw`).
    Frame numbering starts at 1 (frame 0.png has no ground-truth pose)."""
    frame_id, tx, ty, tz, qx, qy, qz, qw = [], [], [], [], [], [], [], []
    with open(gt_path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            fid, x, y, z, a, b, c, d = parts
            frame_id.append(int(fid))
            tx.append(float(x)); ty.append(float(y)); tz.append(float(z))
            qx.append(float(a)); qy.append(float(b)); qz.append(float(c)); qw.append(float(d))
    return {
        "frame_id": np.array(frame_id, dtype=np.int64),
        "tx": np.array(tx), "ty": np.array(ty), "tz": np.array(tz),
        "qx": np.array(qx), "qy": np.array(qy), "qz": np.array(qz), "qw": np.array(qw),
    }


def camera_heading_xz(qx: np.ndarray, qy: np.ndarray, qz: np.ndarray, qw: np.ndarray) -> np.ndarray:
    """Angle (radians) of the camera's forward (local +z) axis projected
    onto the world x-z plane -- this dataset's ground-truth quaternion
    convention has y as the vertical axis (tx/tz range over ~0.5-1.9m,
    consistent with a room footprint; ty ranges over ~1.2m, consistent with
    camera height/tilt changes), so x-z is the floor plane here, playing
    the role the classroom dataset's world-frame (x, y) with z-up heading
    does. Derived from the quaternion's local-z-axis rotation-matrix column
    (2*(qx*qz+qw*qy), *, 1-2*(qx^2+qy^2)), atan2'd over its x/z components
    rather than an Euler yaw decomposition, so it's correct regardless of
    which axis-order convention the recording used."""
    forward_x = 2.0 * (qx * qz + qw * qy)
    forward_z = 1.0 - 2.0 * (qx ** 2 + qy ** 2)
    return np.arctan2(forward_x, forward_z)


def load_detections_by_frame(det_path: Path) -> dict[int, pd.DataFrame]:
    if not det_path.exists():
        print(f"{det_path} not found (detection was skipped) -- rendering without detection boxes")
        return {}
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


def build_visualization(frames: list[tuple[int, Path]], gt_path: Path, det_path: Path, emb_path: Path,
                        out_path: Path, fps: float) -> None:
    """Render observations.mp4: RGB + detection boxes, ground-truth
    trajectory/heading, and the embedding self-correlation matrix with a
    crosshair marking the current frame, side by side -- one video frame per
    posed RGB frame (frame 0 excluded, no ground-truth pose), at --fps.
    """
    print(f"loading ground-truth poses from {gt_path}...")
    gt = load_gt_poses(gt_path)
    pose_row_by_frame = {int(fid): i for i, fid in enumerate(gt["frame_id"])}
    heading = camera_heading_xz(gt["qx"], gt["qy"], gt["qz"], gt["qw"])

    posed_frames = [(fidx, path) for fidx, path in frames if fidx in pose_row_by_frame]
    n_frames = len(posed_frames)
    dropped = len(frames) - n_frames
    if dropped:
        print(f"{dropped} frame(s) with no ground-truth pose excluded from the video")

    print(f"loading detections from {det_path}...")
    det_by_frame = load_detections_by_frame(det_path)

    print(f"loading embeddings from {emb_path}...")
    emb_data = torch.load(emb_path)
    emb_frame_idx = emb_data["frame_idx"].numpy()
    emb_by_frame = {int(f): i for i, f in enumerate(emb_frame_idx)}
    embeddings = emb_data["embedding"].numpy()
    corr = np.corrcoef(embeddings)
    corr_vmin, corr_vmax = corr.min(), corr.max()
    corr_row_for = [emb_by_frame[fidx] for fidx, _ in posed_frames]

    print(f"rendering {n_frames} frames at {fps:.2f} fps -> {out_path}")

    fig = plt.figure(figsize=(16, 6), facecolor=SURFACE, constrained_layout=True)
    gs = fig.add_gridspec(1, 3)
    ax_rgb = fig.add_subplot(gs[0, 0])
    ax_pos = fig.add_subplot(gs[0, 1])
    ax_corr = fig.add_subplot(gs[0, 2])

    first_frame, first_path = posed_frames[0]
    rgb_im = ax_rgb.imshow(np.asarray(Image.open(first_path).convert("RGB")))
    ax_rgb.axis("off")
    ax_rgb.set_title("Freiburg RGB + detections", color=INK_PRIMARY, fontsize=11, pad=8)
    box_artists: list = []

    order = np.argsort(gt["frame_id"])
    traj_x, traj_z = gt["tx"][order], gt["tz"][order]
    ax_pos.plot(traj_x, traj_z, color=BLUE, linewidth=2, label="path", zorder=2)
    cur_point = ax_pos.scatter([traj_x[0]], [traj_z[0]], color=RED, s=60, zorder=3, label="current")
    arrow_len = 0.08 * max(np.ptp(traj_x), np.ptp(traj_z), 1e-3)
    heading_q = ax_pos.quiver([traj_x[0]], [traj_z[0]], [arrow_len], [0.0],
                             color=RED, angles="xy", scale_units="xy", scale=1, width=0.01, zorder=4)
    ax_pos.set_aspect("equal")
    ax_pos.margins(0.15)
    ax_pos.set_facecolor(SURFACE)
    ax_pos.set_title("Position & heading (ground truth, x-z plane)", color=INK_PRIMARY, fontsize=11, pad=8)
    ax_pos.tick_params(colors=INK_MUTED, labelsize=7)
    for spine in ax_pos.spines.values():
        spine.set_color(GRID)
    ax_pos.grid(color=GRID, linewidth=0.6)
    ax_pos.set_axisbelow(True)
    ax_pos.set_xlabel("x (m)", color=INK_MUTED, fontsize=8)
    ax_pos.set_ylabel("z (m)", color=INK_MUTED, fontsize=8)
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
    corr_hline = ax_corr.axhline(0, color=RED, linewidth=1.2, zorder=5)
    corr_vline = ax_corr.axvline(0, color=RED, linewidth=1.2, zorder=5)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=fps, metadata={"title": "Freiburg RGB observations"})
    with writer.saving(fig, str(out_path), dpi=120):
        for i, (fidx, path) in enumerate(posed_frames):
            rgb_im.set_data(np.asarray(Image.open(path).convert("RGB")))
            box_artists = draw_boxes(ax_rgb, box_artists, det_by_frame.get(fidx))

            pose_i = pose_row_by_frame[fidx]
            cur_point.set_offsets([[gt["tx"][pose_i], gt["tz"][pose_i]]])
            heading_q.set_offsets([[gt["tx"][pose_i], gt["tz"][pose_i]]])
            heading_q.set_UVC(arrow_len * np.cos(heading[pose_i]), arrow_len * np.sin(heading[pose_i]))

            corr_row = corr_row_for[i]
            corr_hline.set_ydata([corr_row, corr_row])
            corr_vline.set_xdata([corr_row, corr_row])

            fig.suptitle(f"frame {i + 1}/{n_frames}   t={i / fps:.2f}s", color=INK_SECONDARY, fontsize=10)
            writer.grab_frame()

            if (i + 1) % 200 == 0 or i + 1 == n_frames:
                print(f"[{i + 1}/{n_frames}] rendered")

    plt.close(fig)
    print(f"visualization written to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rgb-dir", default=RGB_DIR)
    parser.add_argument("--gt-path", default=GT_PATH, help="used only by --visualize")
    parser.add_argument("--out-dir", default="outputs/freiburg_detections")
    parser.add_argument("--embedding-model", choices=["yolo", "dino"], default="yolo",
                        help="backend for embeddings.pt: 'yolo' (YOLOv8n penultimate-layer "
                             "features, 256-dim) or 'dino' (frozen DINOv2 CLS embedding, "
                             "384-dim for the default dinov2_vits14). Default 'yolo' keeps "
                             "every output filename identical to before this flag existed; "
                             "'dino' appends a _dino suffix to embeddings.pt/observations.mp4")
    parser.add_argument("--detect", choices=["auto", "on", "off"], default="auto",
                        help="whether to run YOLO object detection (detections.csv): 'auto' = "
                             "on when --embedding-model yolo, off when --embedding-model dino "
                             "(DINOv2 has no detection head); 'on'/'off' force it either way")
    parser.add_argument("--yolo-model", default="yolov8n.pt",
                        help="ultralytics checkpoint name or path to a custom .pt (used for "
                             "detection and/or when --embedding-model yolo)")
    parser.add_argument("--dino-model", default="dinov2_vits14",
                        help="torch.hub facebookresearch/dinov2 entrypoint, used when "
                             "--embedding-model dino. Downloads on first use (requires network "
                             "access; cached under ~/.cache/torch/hub, override via $TORCH_HOME)")
    parser.add_argument("--dino-img-size", type=int, default=224,
                        help="resize frames to this square size before the DINOv2 forward pass; "
                             "must be a multiple of the patch size (14)")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--conf", type=float, default=0.25,
                        help="detection confidence threshold (only used when detection runs)")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="nominal capture rate used to synthesize timestamp_ns (this "
                             "dataset has no real capture clock) and, by default, --viz-fps")
    parser.add_argument("--limit", type=int, default=None, help="only process the first N frames")
    parser.add_argument("--force", action="store_true",
                        help="regenerate the applicable outputs even if they already exist")
    parser.add_argument("--visualize", action="store_true",
                        help="also render observations.mp4 (RGB + detections, ground-truth "
                             "trajectory/heading, embedding self-correlation) to --out-dir")
    parser.add_argument("--viz-out", default=None,
                        help="output video path (default: <out-dir>/observations.mp4, or "
                             "observations_dino.mp4 if --embedding-model dino)")
    parser.add_argument("--viz-fps", type=float, default=None, help="default: --fps")
    parser.add_argument("--viz-force", action="store_true",
                        help="rebuild observations.mp4 even if it already exists at the target path")
    args = parser.parse_args()

    if args.embedding_model == "dino" and args.dino_img_size % DINO_PATCH_SIZE != 0:
        parser.error(f"--dino-img-size {args.dino_img_size} must be a multiple of "
                     f"{DINO_PATCH_SIZE} (DINOv2's patch size)")

    do_detect = (args.embedding_model == "yolo") if args.detect == "auto" else (args.detect == "on")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    det_path = out_dir / "detections.csv"  # shared/backend-independent, never suffixed
    emb_path = with_suffix_for_model(out_dir / "embeddings.pt", args.embedding_model)

    need_detect = do_detect and (args.force or not det_path.exists())
    if do_detect and not need_detect:
        print(f"{det_path} already exists, skipping detection (pass --force to regenerate)")
    need_embed = args.force or not emb_path.exists()
    if not need_embed:
        print(f"{emb_path} already exists, skipping embedding (pass --force to regenerate)")

    viz_out = Path(args.viz_out) if args.viz_out else with_suffix_for_model(
        out_dir / "observations.mp4", args.embedding_model)
    need_visualize = args.visualize and (args.viz_force or not viz_out.exists())
    if args.visualize and not need_visualize:
        print(f"{viz_out} already exists, skipping visualization (pass --viz-force to regenerate)")

    need_process = need_detect or need_embed
    frames: list[tuple[int, Path]] = []
    if need_process or need_visualize:
        print(f"listing frames in {args.rgb_dir}...")
        frames = list_frames(Path(args.rgb_dir))
        if args.limit is not None:
            frames = frames[:args.limit]
        print(f"found {len(frames)} frames")

    if need_process:
        device = resolve_device(args.device)

        model = None
        if need_detect or (need_embed and args.embedding_model == "yolo"):
            print(f"loading {args.yolo_model} on device={device}...")
            model = load_yolo_model(args.yolo_model)

        backbone = None
        if need_embed and args.embedding_model == "dino":
            print(f"loading {args.dino_model} on device={device} (downloads on first use)...")
            backbone = load_backbone(args.dino_model).to(device)

        det_rows = []
        frame_idxs, timestamps, embeds = [], [], []
        n_frames = len(frames)
        for start in range(0, n_frames, args.batch_size):
            end = min(start + args.batch_size, n_frames)
            batch = frames[start:end]
            images = [np.asarray(Image.open(path).convert("RGB")) for _, path in batch]

            if need_detect:
                detections = run_detect(model, images, device, args.conf)
                for j, result in enumerate(detections):
                    frame_idx = batch[j][0]
                    ts = round(frame_idx / args.fps * 1e9)
                    boxes = result.boxes
                    for k in range(len(boxes)):
                        x1, y1, x2, y2 = boxes.xyxy[k].tolist()
                        cls_id = int(boxes.cls[k])
                        det_rows.append({"frame_idx": frame_idx, "timestamp_ns": ts, "class_id": cls_id,
                                         "class_name": model.names[cls_id], "confidence": float(boxes.conf[k]),
                                         "x1": x1, "y1": y1, "x2": x2, "y2": y2})

            if need_embed:
                embeddings = (run_embed_yolo(model, images, device) if args.embedding_model == "yolo"
                             else run_embed_dino(backbone, images, device, args.dino_img_size))
                for j, embedding in enumerate(embeddings):
                    frame_idx = batch[j][0]
                    frame_idxs.append(frame_idx)
                    timestamps.append(round(frame_idx / args.fps * 1e9))
                    embeds.append(embedding.cpu())

            status = []
            if need_detect:
                status.append(f"{len(det_rows)} detections so far")
            if need_embed:
                status.append(f"{len(embeds)} embeddings so far")
            print(f"[{end}/{n_frames}] " + ", ".join(status))

        if need_detect:
            pd.DataFrame(det_rows).to_csv(det_path, index=False)
            print(f"detections written to {det_path} ({len(det_rows)} rows)")

        if need_embed:
            torch.save({"frame_idx": torch.tensor(frame_idxs, dtype=torch.long),
                       "timestamp_ns": torch.tensor(timestamps, dtype=torch.long),
                       "embedding": torch.stack(embeds)}, emb_path)
            print(f"embeddings written to {emb_path} ({len(embeds)} frames)")

    if need_visualize:
        viz_fps = args.viz_fps if args.viz_fps is not None else args.fps
        build_visualization(frames, Path(args.gt_path), det_path, emb_path, viz_out, viz_fps)


if __name__ == "__main__":
    main()
