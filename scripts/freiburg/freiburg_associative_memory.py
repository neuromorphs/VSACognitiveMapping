"""Bind Freiburg (ICL-NUIM "living room traj0") content to
position/heading/time context, bundle into three separate associative
memories, and recall "what did I see at this position/heading/time?" by
unbind + cleanup lookup -- the Freiburg counterpart of
scripts/classroom/classroom_associative_memory.py, binding against
data/frieburg/livingRoom0.gt.freiburg's ground-truth camera poses instead of
Spot's odometry.

    python scripts/freiburg/freiburg_associative_memory.py build --subset all
    python scripts/freiburg/freiburg_associative_memory.py build --subset uncertain
    python scripts/freiburg/freiburg_associative_memory.py query --memory outputs/freiburg_detections/associative_memory_all.pt --query-x 0.05 --query-y -1.5
    python scripts/freiburg/freiburg_associative_memory.py evaluate --memory outputs/freiburg_detections/associative_memory_all.pt

For each frame i, content_i = random_project_to_phasor(embedding_i) (see
scripts/freiburg/detect_and_embed_freiburg.py for the raw embeddings) and
context_i = one of three independent phasor codes -- position (Bx**x * By**y),
heading (a circular-safe base**yaw), or time (Bt**row_idx). "Position" here
is (tx, tz) from the ground-truth pose file (this dataset's quaternion
convention has y as the vertical axis -- see camera_heading_xz below -- so
x-z is the floor plane, the analogue of the classroom's world-frame (x, y)
with z-up) and "heading" is the camera's forward direction projected onto
that plane. Frame 0 has no ground-truth pose in this dataset and is dropped
before binding (see load_frame_data). Same three-separate-memories design,
recall, `evaluate`, `--subset stride` + `demo` "have I seen this before?"
video as the classroom script -- see its docstring and
docs/classroom_associative_memory_math.md for the full pipeline writeup;
this module only re-derives the parts that depend on how position/heading
context and RGB frames are obtained.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.colors as mcolors  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from vsa_cognitive_mapping.vsa import (
    Phasor,
    cosine_self_correlation,
    pca_components,
    phasor_correlation_matrix,
    phasor_cross_correlation,
    random_project_to_phasor,
)

RGB_DIR = "data/frieburg/rgb"
GT_PATH = "data/frieburg/livingRoom0.gt.freiburg"

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BLUE = "#2a78d6"
RED = "#e34948"
CLASS_CMAP = plt.get_cmap("tab20")

# Signed [-1, 1] correlation -> diverging, same validated pair
# scripts/associative_memory.py, scripts/encoder_sweep.py, and the
# scripts/classroom/ scripts each define locally.
DIVERGING_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#2a78d6", "#f0efec", "#e34948"])

AXES = ("position", "heading", "time")


# ---------------------------------------------------------------------------
# Helpers duplicated from scripts/freiburg/detect_and_embed_freiburg.py and
# scripts/classroom (repo convention: each script stays self-contained
# rather than importing another script).
# ---------------------------------------------------------------------------

def camera_heading_xz(qx: np.ndarray, qy: np.ndarray, qz: np.ndarray, qw: np.ndarray) -> np.ndarray:
    """Angle (radians) of the camera's forward (local +z) axis projected
    onto the world x-z plane -- see
    scripts/freiburg/detect_and_embed_freiburg.py's copy of this function
    for the derivation and why x-z (not x-y) is this dataset's floor plane."""
    forward_x = 2.0 * (qx * qz + qw * qy)
    forward_z = 1.0 - 2.0 * (qx ** 2 + qy ** 2)
    return np.arctan2(forward_x, forward_z)


def load_gt_poses(gt_path: Path) -> dict[str, np.ndarray]:
    """data/frieburg/livingRoom0.gt.freiburg -> {frame_id, tx, ty, tz, qx,
    qy, qz, qw} arrays. Frame numbering starts at 1 (frame 0.png has no
    ground-truth pose)."""
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


def load_rgb_image(rgb_dir: Path, frame_idx: int) -> Image.Image:
    return Image.open(rgb_dir / f"{frame_idx}.png").convert("RGB")


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
    base = Phasor(dim=hd_dim, seed=seed, circular=True, max_freq=max_freq)
    return np.stack([(base ** float(a)).values for a in yaw])


def heading_sweep_codes(hd_dim: int, seed: int, max_freq: int, n_angles: int) -> tuple[np.ndarray, np.ndarray]:
    angles = np.linspace(-np.pi, np.pi, n_angles, endpoint=False)
    return angles, encode_heading(angles, hd_dim, seed, max_freq)


def position_grid_codes(x_range: tuple[float, float], y_range: tuple[float, float], resolution: int,
                        hd_dim: int, x_seed: int, y_seed: int, length_scale: float):
    grid_x = np.linspace(*x_range, resolution)
    grid_y = np.linspace(*y_range, resolution)
    gx, gy = np.meshgrid(grid_x, grid_y)
    xy = np.stack([gx.ravel(), gy.ravel()], axis=1)
    codes = encode_position(xy, hd_dim, x_seed, y_seed, length_scale)
    return grid_x, grid_y, codes


def adjacent_frame_uncertainty(corr: np.ndarray) -> np.ndarray:
    n = corr.shape[0]
    uncertainty = np.zeros(n)
    uncertainty[1:] = 1.0 - corr[np.arange(1, n), np.arange(0, n - 1)]
    return uncertainty


def global_frame_uncertainty(corr: np.ndarray) -> np.ndarray:
    n = corr.shape[0]
    row_sum = corr.sum(axis=1) - np.diag(corr)
    return 1.0 - row_sum / (n - 1)


def with_suffix_for_model(path: Path, embedding_model: str) -> Path:
    return path if embedding_model == "yolo" else path.with_stem(path.stem + "_dino")


def load_detections_by_frame(det_path: Path) -> dict[int, pd.DataFrame]:
    if not det_path.exists():
        print(f"{det_path} not found (detection was skipped) -- proceeding without detection boxes")
        return {}
    df = pd.read_csv(det_path)
    return {int(frame_idx): group for frame_idx, group in df.groupby("frame_idx")}


def draw_boxes(ax, prior_artists: list, frame_dets: pd.DataFrame | None) -> list:
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


# ---------------------------------------------------------------------------
# Shared data loading
# ---------------------------------------------------------------------------

def load_frame_data(embeddings_path: str | Path, gt_path: str | Path, limit: int | None) -> dict:
    print(f"loading {embeddings_path}...")
    data = torch.load(embeddings_path)
    embeddings_t = data["embedding"]
    frame_idx_all = data["frame_idx"].numpy()
    timestamp_all = data["timestamp_ns"].numpy()
    if limit is not None:
        embeddings_t = embeddings_t[:limit]
        frame_idx_all = frame_idx_all[:limit]
        timestamp_all = timestamp_all[:limit]

    print(f"loading {gt_path}...")
    gt = load_gt_poses(Path(gt_path))
    pose_row_by_frame = {int(fid): i for i, fid in enumerate(gt["frame_id"])}

    # Frame 0 has no ground-truth pose in this dataset (poses start at frame
    # 1) -- drop any embedded frame that can't be bound to a position/
    # heading, rather than inventing one.
    has_pose = np.array([int(f) in pose_row_by_frame for f in frame_idx_all])
    n_dropped = int((~has_pose).sum())
    if n_dropped:
        print(f"{n_dropped} frame(s) have no ground-truth pose in {gt_path} -- "
              "excluded from position/heading/time binding")

    frame_idx = frame_idx_all[has_pose]
    embeddings_t = embeddings_t[torch.from_numpy(has_pose)]
    timestamp_ns = timestamp_all[has_pose]
    pose_rows = np.array([pose_row_by_frame[int(f)] for f in frame_idx])

    x = gt["tx"][pose_rows]
    y = gt["tz"][pose_rows]  # this dataset's horizontal floor-plane axis (see camera_heading_xz)
    yaw = camera_heading_xz(gt["qx"][pose_rows], gt["qy"][pose_rows], gt["qz"][pose_rows], gt["qw"][pose_rows])
    row_idx = np.arange(len(frame_idx))

    order = np.argsort(gt["frame_id"])
    return {
        "embeddings_t": embeddings_t,
        "row_idx": row_idx,
        "x": x, "y": y, "yaw": yaw,
        "timestamp_ns": timestamp_ns,
        "dataset_frame_idx": frame_idx,
        "n_frames": len(frame_idx),
        "odom_x": gt["tx"][order], "odom_y": gt["tz"][order],  # full ground-truth trajectory, for plotting
    }


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def detect_stationary_trim(x: np.ndarray, y: np.ndarray, anchor_frames: int,
                           distance_threshold: float) -> tuple[int, int]:
    start_anchor = (np.median(x[:anchor_frames]), np.median(y[:anchor_frames]))
    end_anchor = (np.median(x[-anchor_frames:]), np.median(y[-anchor_frames:]))
    far_from_start = np.hypot(x - start_anchor[0], y - start_anchor[1]) > distance_threshold
    far_from_end = np.hypot(x - end_anchor[0], y - end_anchor[1]) > distance_threshold
    trim_start = int(np.argmax(far_from_start)) if far_from_start.any() else 0
    trim_end = int(np.argmax(far_from_end[::-1])) if far_from_end.any() else 0
    return trim_start, trim_end


def select_subset(embeddings: np.ndarray, uncertainty_type: str, quantile: float) -> np.ndarray:
    corr = cosine_self_correlation(embeddings)
    uncertainty = adjacent_frame_uncertainty(corr) if uncertainty_type == "local" else global_frame_uncertainty(corr)
    threshold = np.quantile(uncertainty, quantile)
    return np.where(uncertainty >= threshold)[0]


def bundle_memory(content: np.ndarray, context: np.ndarray, subset_idx: np.ndarray) -> np.ndarray:
    traces = content[subset_idx] * context[subset_idx]
    return traces.mean(axis=0)


def cmd_build(args: argparse.Namespace) -> None:
    embeddings_path = Path(args.embeddings) if args.embeddings else with_suffix_for_model(
        Path("outputs/freiburg_detections/embeddings.pt"), args.embedding_model)
    frames = load_frame_data(embeddings_path, args.gt_path, args.limit)
    n_frames = frames["n_frames"]

    content_input = frames["embeddings_t"]
    pca_components_used = None
    if args.pca_whiten:
        raw = content_input.numpy()
        pca_components_used = args.pca_components or raw.shape[1]
        whitened, explained_variance_ratio = pca_components(raw, pca_components_used)
        kept_variance = explained_variance_ratio[:pca_components_used].sum()
        print(f"PCA-whitened {raw.shape[1]}-dim embeddings to {pca_components_used} unit-variance "
              f"components ({kept_variance:.1%} of variance retained) before projecting to phasor content")
        content_input = torch.from_numpy(whitened.astype(np.float32))

    print(f"projecting {n_frames} embeddings to {args.hd_dim}-dim phasor content...")
    content, W = random_project_to_phasor(content_input, d=args.hd_dim, seed=args.content_seed)
    content = content.numpy()

    pos_codes = encode_position(np.stack([frames["x"], frames["y"]], axis=1), args.hd_dim,
                                args.x_seed, args.y_seed, args.pos_length_scale)
    heading_codes = encode_heading(frames["yaw"], args.hd_dim, args.yaw_seed, args.yaw_max_freq)
    time_codes = encode_time(frames["row_idx"], args.hd_dim, args.time_seed, args.time_length_scale)

    if args.trim_stationary:
        trim_start, trim_end = detect_stationary_trim(frames["x"], frames["y"], args.trim_anchor_frames,
                                                       args.trim_distance_threshold)
    else:
        trim_start, trim_end = 0, 0
    if trim_start or trim_end:
        print(f"--trim-stationary: excluding {trim_start} stationary frames from the start and "
              f"{trim_end} from the end ({n_frames - trim_start - trim_end}/{n_frames} frames "
              "remain in consideration)")
    valid_range = np.arange(trim_start, n_frames - trim_end)

    if args.subset == "all":
        subset_idx = valid_range
    elif args.subset == "stride":
        subset_idx = valid_range[::args.stride]
    else:
        raw_embeddings = frames["embeddings_t"].numpy()
        local_idx = select_subset(raw_embeddings[valid_range], args.uncertainty_type, args.uncertainty_quantile)
        subset_idx = valid_range[local_idx]
    print(f"bundling {len(subset_idx)}/{n_frames} frames ({args.subset}) into each memory")

    memory_position = bundle_memory(content, pos_codes, subset_idx)
    memory_heading = bundle_memory(content, heading_codes, subset_idx)
    memory_time = bundle_memory(content, time_codes, subset_idx)

    result = {
        "subset": args.subset,
        "embedding_model": args.embedding_model,
        "dino_model": args.dino_model if args.embedding_model == "dino" else None,
        "trim_start": trim_start,
        "trim_end": trim_end,
        "hd_dim": args.hd_dim,
        "content_seed": args.content_seed,
        "pca_whiten": args.pca_whiten,
        "pca_components": pca_components_used,
        "W": W,
        "memory_position": torch.from_numpy(memory_position),
        "memory_heading": torch.from_numpy(memory_heading),
        "memory_time": torch.from_numpy(memory_time),
        "bases": {
            "x_seed": args.x_seed, "y_seed": args.y_seed, "pos_length_scale": args.pos_length_scale,
            "yaw_seed": args.yaw_seed, "yaw_max_freq": args.yaw_max_freq,
            "time_seed": args.time_seed, "time_length_scale": args.time_length_scale,
        },
        "codebook": {
            "content": torch.from_numpy(content[subset_idx]),
            "embedding": content_input[subset_idx].clone(),
            "row_idx": torch.from_numpy(subset_idx),
            "frame_idx": torch.from_numpy(frames["dataset_frame_idx"][subset_idx]),
            "x": torch.from_numpy(frames["x"][subset_idx]),
            "y": torch.from_numpy(frames["y"][subset_idx]),
            "yaw": torch.from_numpy(frames["yaw"][subset_idx]),
            "timestamp_ns": torch.from_numpy(frames["timestamp_ns"][subset_idx]),
        },
    }

    out_path = Path(args.out) if args.out else with_suffix_for_model(
        Path(f"outputs/freiburg_detections/associative_memory_{args.subset}.pt"), args.embedding_model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, out_path)
    print(f"memory written to {out_path}")


# ---------------------------------------------------------------------------
# query
# ---------------------------------------------------------------------------

def unbind_and_score(memory: np.ndarray, query_context: np.ndarray, codebook_content: np.ndarray) -> np.ndarray:
    residual = memory / query_context
    return phasor_cross_correlation(residual[None, :], codebook_content)[0]


def cmd_query(args: argparse.Namespace) -> None:
    if args.query_time is None and args.query_yaw is None and (args.query_x is None or args.query_y is None):
        raise SystemExit("query needs at least one of: --query-x AND --query-y, --query-yaw, --query-time")

    saved = torch.load(args.memory)
    hd_dim, bases = saved["hd_dim"], saved["bases"]
    codebook = saved["codebook"]
    codebook_content = codebook["content"].numpy()
    mem_embedding_model = saved.get("embedding_model", "yolo")
    mem_dino_model = saved.get("dino_model")
    provenance = f"embedding_model={mem_embedding_model}" + (f" ({mem_dino_model})" if mem_dino_model else "")
    print(f"loaded {args.memory}: subset={saved['subset']}, {provenance}, "
          f"{codebook_content.shape[0]} stored frames")

    det_by_frame = load_detections_by_frame(Path(args.detections))

    axis_sims: dict[str, np.ndarray] = {}
    if args.query_x is not None and args.query_y is not None:
        ctx = encode_position(np.array([[args.query_x, args.query_y]]), hd_dim,
                              bases["x_seed"], bases["y_seed"], bases["pos_length_scale"])[0]
        axis_sims["position"] = unbind_and_score(saved["memory_position"].numpy(), ctx, codebook_content)
    if args.query_yaw is not None:
        ctx = encode_heading(np.array([args.query_yaw]), hd_dim, bases["yaw_seed"], bases["yaw_max_freq"])[0]
        axis_sims["heading"] = unbind_and_score(saved["memory_heading"].numpy(), ctx, codebook_content)
    if args.query_time is not None:
        ctx = encode_time(np.array([args.query_time]), hd_dim, bases["time_seed"], bases["time_length_scale"])[0]
        axis_sims["time"] = unbind_and_score(saved["memory_time"].numpy(), ctx, codebook_content)

    def report(label: str, sims: np.ndarray) -> int:
        order = np.argsort(sims)[::-1][:args.top_k]
        print(f"\n[{label}] top-{args.top_k}:")
        for rank, idx in enumerate(order, 1):
            fidx = int(codebook["frame_idx"][idx])
            dets = det_by_frame.get(fidx)
            classes = ", ".join(f'{r["class_name"]}({r["confidence"]:.2f})' for _, r in dets.iterrows()) if dets is not None else "(no detections)"
            x, y, yaw = float(codebook["x"][idx]), float(codebook["y"][idx]), float(codebook["yaw"][idx])
            print(f"  {rank}. frame_idx={fidx} sim={sims[idx]:.3f} pos=({x:.2f}, {y:.2f}) yaw={yaw:.2f} -- {classes}")
        return int(order[0])

    top1_per_axis = {axis: report(axis, sims) for axis, sims in axis_sims.items()}

    if len(axis_sims) > 1:
        combined = sum(axis_sims.values())
        best_idx = report("combined", combined)
    else:
        best_idx = next(iter(top1_per_axis.values()))

    out_path = Path(args.out_path) if args.out_path else with_suffix_for_model(
        Path("outputs/freiburg_detections/query_result.png"), args.embedding_model)
    plot_query_result(args, saved, codebook, det_by_frame, axis_sims, best_idx, out_path)


def plot_query_result(args: argparse.Namespace, saved: dict, codebook: dict,
                      det_by_frame: dict, axis_sims: dict, best_idx: int, out_path: Path) -> None:
    fidx = int(codebook["frame_idx"][best_idx])
    print(f"loading frame {fidx} from {args.rgb_dir} to fetch the recalled frame's image...")
    image = load_rgb_image(Path(args.rgb_dir), fidx)

    gt = load_gt_poses(Path(args.gt_path))
    order = np.argsort(gt["frame_id"])
    traj_x, traj_y = gt["tx"][order], gt["tz"][order]

    fig, (ax_pos, ax_rgb) = plt.subplots(1, 2, figsize=(14, 6), facecolor=SURFACE, constrained_layout=True)

    ax_pos.plot(traj_x, traj_y, color=BLUE, linewidth=1.5, alpha=0.6, zorder=1, label="trajectory")
    rx, ry = float(codebook["x"][best_idx]), float(codebook["y"][best_idx])
    ax_pos.scatter([rx], [ry], color=RED, s=80, zorder=3, label="recalled frame")
    if "position" in axis_sims:
        ax_pos.scatter([args.query_x], [args.query_y], color=INK_PRIMARY, marker="x", s=100, zorder=4, label="query")
        ax_pos.plot([args.query_x, rx], [args.query_y, ry], color=INK_MUTED, linewidth=1, linestyle=":", zorder=2)
    ax_pos.set_aspect("equal")
    ax_pos.set_facecolor(SURFACE)
    ax_pos.set_title("Recalled frame's position", color=INK_PRIMARY, fontsize=12, pad=8)
    ax_pos.set_xlabel("x (m)", color=INK_MUTED, fontsize=9)
    ax_pos.set_ylabel("z (m)", color=INK_MUTED, fontsize=9)
    ax_pos.tick_params(colors=INK_MUTED, labelsize=8)
    for spine in ax_pos.spines.values():
        spine.set_color(GRID)
    ax_pos.grid(color=GRID, linewidth=0.6)
    legend = ax_pos.legend(loc="best", fontsize=8, facecolor=SURFACE, edgecolor=GRID)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    ax_rgb.imshow(np.asarray(image))
    ax_rgb.axis("off")
    ax_rgb.set_title(f"Recalled: frame_idx={fidx}", color=INK_PRIMARY, fontsize=12, pad=8)
    draw_boxes(ax_rgb, [], det_by_frame.get(fidx))

    fig.suptitle(f"\"What did I see?\" -- memory subset={saved['subset']}", color=INK_PRIMARY, fontsize=13)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"\nquery plot written to {out_path}")


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

def circular_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(a - b), np.cos(a - b))


def l1_l2(errors: np.ndarray) -> tuple[float, float]:
    return float(np.mean(np.abs(errors))), float(np.sqrt(np.mean(errors ** 2)))


def physical_error(axis: str, codebook: dict, predicted_idx: np.ndarray) -> np.ndarray:
    if axis == "position":
        x, y = codebook["x"].numpy(), codebook["y"].numpy()
        return np.hypot(x - x[predicted_idx], y - y[predicted_idx])
    if axis == "heading":
        yaw = codebook["yaw"].numpy()
        return np.abs(circular_diff(yaw, yaw[predicted_idx]))
    row_idx = codebook["row_idx"].numpy()
    return np.abs(row_idx - row_idx[predicted_idx]).astype(float)


AXIS_UNITS = {"position": "m", "heading": "rad", "time": "frames"}


def cmd_evaluate(args: argparse.Namespace) -> None:
    saved = torch.load(args.memory)
    hd_dim, bases = saved["hd_dim"], saved["bases"]
    codebook = saved["codebook"]
    codebook_content = codebook["content"].numpy()
    n = codebook_content.shape[0]
    mem_embedding_model = saved.get("embedding_model", "yolo")
    mem_dino_model = saved.get("dino_model")
    provenance = f"embedding_model={mem_embedding_model}" + (f" ({mem_dino_model})" if mem_dino_model else "")
    print(f"evaluating {args.memory}: subset={saved['subset']}, {provenance}, n={n} stored frames")

    embeddings_corr = cosine_self_correlation(codebook["embedding"].numpy())

    pos_codes = encode_position(np.stack([codebook["x"].numpy(), codebook["y"].numpy()], axis=1),
                                hd_dim, bases["x_seed"], bases["y_seed"], bases["pos_length_scale"])
    heading_codes = encode_heading(codebook["yaw"].numpy(), hd_dim, bases["yaw_seed"], bases["yaw_max_freq"])
    time_codes = encode_time(codebook["row_idx"].numpy(), hd_dim, bases["time_seed"], bases["time_length_scale"])

    errors = {}
    per_axis_stages = {}
    for axis, memory_key, context_codes in (("position", "memory_position", pos_codes),
                                            ("heading", "memory_heading", heading_codes),
                                            ("time", "memory_time", time_codes)):
        memory = saved[memory_key].numpy()

        bound = codebook_content * context_codes
        context_corr = phasor_correlation_matrix(context_codes)
        bound_corr = phasor_correlation_matrix(bound)
        memory_vs_bound = phasor_cross_correlation(memory[None, :], bound)[0]
        per_axis_stages[axis] = (context_corr, bound_corr, memory_vs_bound)

        residuals = memory[None, :] / context_codes
        sims = phasor_cross_correlation(residuals, codebook_content)
        predicted_idx = np.argmax(sims, axis=1)
        item_errors = physical_error(axis, codebook, predicted_idx)
        l1, l2 = l1_l2(item_errors)
        errors[axis] = (l1, l2)

        unit = AXIS_UNITS[axis]
        print(f"[{axis}] L1={l1:.3f}{unit}  L2={l2:.3f}{unit}")

    out_path = Path(args.out_path) if args.out_path else with_suffix_for_model(
        Path("outputs/freiburg_detections/evaluate_result.png"), args.embedding_model)
    plot_pipeline_diagnostic(embeddings_corr, per_axis_stages, errors, saved["subset"], out_path)


def _plot_corr_cell(ax, corr: np.ndarray, cmap, title: str) -> None:
    vmin, vmax = float(corr.min()), float(corr.max())
    im = ax.imshow(corr, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK_PRIMARY, fontsize=10, pad=6)
    ax.tick_params(colors=INK_MUTED, labelsize=6)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors=INK_MUTED, labelsize=6)


def plot_pipeline_diagnostic(embeddings_corr: np.ndarray, per_axis_stages: dict,
                             errors: dict, subset: str, out_path: Path) -> Path:
    n = embeddings_corr.shape[0]
    fig = plt.figure(figsize=(21, 5.5 * len(AXES)), facecolor=SURFACE, constrained_layout=True)
    gs = fig.add_gridspec(len(AXES), 5, width_ratios=(1, 1, 1, 1, 0.8))

    for row, axis in enumerate(AXES):
        context_corr, bound_corr, memory_vs_bound = per_axis_stages[axis]

        _plot_corr_cell(fig.add_subplot(gs[row, 0]), embeddings_corr, "viridis",
                        f"[{axis}] embeddings self-corr")
        _plot_corr_cell(fig.add_subplot(gs[row, 1]), context_corr, DIVERGING_CMAP,
                        f"[{axis}] context self-corr")
        _plot_corr_cell(fig.add_subplot(gs[row, 2]), bound_corr, DIVERGING_CMAP,
                        f"[{axis}] bound (content*context) self-corr")

        ax_mem = fig.add_subplot(gs[row, 3])
        ax_mem.plot(np.arange(n), memory_vs_bound, color=BLUE, linewidth=1)
        ax_mem.set_facecolor(SURFACE)
        ax_mem.set_title(f"[{axis}] memory vs. each bound vector", color=INK_PRIMARY, fontsize=10, pad=6)
        ax_mem.set_xlabel("codebook index", color=INK_MUTED, fontsize=8)
        ax_mem.set_ylabel("similarity", color=INK_MUTED, fontsize=8)
        ax_mem.tick_params(colors=INK_MUTED, labelsize=7)
        for spine in ax_mem.spines.values():
            spine.set_color(GRID)
        ax_mem.grid(color=GRID, linewidth=0.6)
        ax_mem.set_axisbelow(True)

        ax_err = fig.add_subplot(gs[row, 4])
        l1, l2 = errors[axis]
        ax_err.bar([0, 1], [l1, l2], color=[BLUE, RED])
        ax_err.set_facecolor(SURFACE)
        ax_err.set_xticks([0, 1])
        ax_err.set_xticklabels(["L1", "L2"], color=INK_MUTED, fontsize=8)
        ax_err.set_ylabel(AXIS_UNITS[axis], color=INK_MUTED, fontsize=8)
        ax_err.tick_params(colors=INK_MUTED, labelsize=7)
        for spine in ax_err.spines.values():
            spine.set_color(GRID)
        ax_err.grid(color=GRID, linewidth=0.6, axis="y")
        ax_err.set_axisbelow(True)
        ax_err.set_title(f"[{axis}] top-1 error", color=INK_PRIMARY, fontsize=10, pad=6)

    fig.suptitle(f"Associative memory pipeline -- subset={subset}", color=INK_PRIMARY, fontsize=14)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"pipeline diagnostic written to {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# demo -- "have I seen this before?" over held-out (val) frames
# ---------------------------------------------------------------------------

def cmd_demo(args: argparse.Namespace) -> None:
    saved = torch.load(args.memory)
    hd_dim, bases = saved["hd_dim"], saved["bases"]
    codebook = saved["codebook"]
    train_row_idx = codebook["row_idx"].numpy()
    mem_embedding_model = saved.get("embedding_model", "yolo")
    mem_dino_model = saved.get("dino_model")
    provenance = f"embedding_model={mem_embedding_model}" + (f" ({mem_dino_model})" if mem_dino_model else "")
    print(f"loaded {args.memory}: subset={saved['subset']}, {provenance}, {len(train_row_idx)} train frames")

    embeddings_path = Path(args.embeddings) if args.embeddings else with_suffix_for_model(
        Path("outputs/freiburg_detections/embeddings.pt"), args.embedding_model)
    frames = load_frame_data(embeddings_path, args.gt_path, None)
    n_frames = frames["n_frames"]
    trim_start, trim_end = saved.get("trim_start", 0), saved.get("trim_end", 0)
    valid_range = np.arange(trim_start, n_frames - trim_end)
    val_row_idx = np.setdiff1d(valid_range, train_row_idx)
    if args.limit is not None:
        val_row_idx = val_row_idx[:args.limit]
    if trim_start or trim_end:
        print(f"excluding the build's trimmed stationary frames (first {trim_start}, last "
              f"{trim_end}) from the held-out set too")
    print(f"{len(val_row_idx)} held-out (val) frames to query, out of {n_frames} total")

    content_input = frames["embeddings_t"]
    if saved["pca_whiten"]:
        whitened, _ = pca_components(content_input.numpy(), saved["pca_components"])
        content_input = torch.from_numpy(whitened.astype(np.float32))
    content, _ = random_project_to_phasor(content_input, d=hd_dim, W=saved["W"])
    content = content.numpy()

    train_time_codes = encode_time(codebook["row_idx"].numpy(), hd_dim, bases["time_seed"], bases["time_length_scale"])

    pad = 1.0
    x_range = (float(frames["odom_x"].min() - pad), float(frames["odom_x"].max() + pad))
    y_range = (float(frames["odom_y"].min() - pad), float(frames["odom_y"].max() + pad))
    grid_x, grid_y, grid_codes = position_grid_codes(x_range, y_range, args.grid_resolution, hd_dim,
                                                      bases["x_seed"], bases["y_seed"], bases["pos_length_scale"])
    angles, angle_codes = heading_sweep_codes(hd_dim, bases["yaw_seed"], bases["yaw_max_freq"], args.n_angles)
    heat_shape = (args.grid_resolution, args.grid_resolution)

    memory_position = saved["memory_position"].numpy()
    memory_heading = saved["memory_heading"].numpy()
    memory_time = saved["memory_time"].numpy()

    det_by_frame = load_detections_by_frame(Path(args.detections))
    rgb_dir = Path(args.rgb_dir)

    val_ts = frames["timestamp_ns"][val_row_idx]
    if args.fps is None:
        duration_s = (val_ts[-1] - val_ts[0]) / 1e9 if len(val_ts) > 1 else 1.0
        fps = (len(val_row_idx) - 1) / duration_s if duration_s > 0 else 10.0
    else:
        fps = args.fps
    out_path = Path(args.out_path) if args.out_path else with_suffix_for_model(
        Path("outputs/freiburg_detections/demo.mp4"), args.embedding_model)
    print(f"rendering {len(val_row_idx)} held-out frames at {fps:.2f} fps -> {out_path}")

    fig = plt.figure(figsize=(20, 6.5), facecolor=SURFACE, constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=(1, 1.2, 1))
    ax_rgb = fig.add_subplot(gs[0, 0])
    ax_heat = fig.add_subplot(gs[0, 1])
    ax_polar = fig.add_subplot(gs[0, 2], projection="polar")

    first_frame_num = int(frames["dataset_frame_idx"][val_row_idx[0]])
    rgb_im = ax_rgb.imshow(np.asarray(load_rgb_image(rgb_dir, first_frame_num)))
    ax_rgb.axis("off")
    ax_rgb.set_title("current view (held out)", color=INK_PRIMARY, fontsize=11, pad=8)
    box_artists: list = []

    heat_im = ax_heat.imshow(np.zeros(heat_shape), origin="lower", cmap="viridis",
                             extent=(*x_range, *y_range), aspect="equal")
    ax_heat.plot(frames["odom_x"], frames["odom_y"], color=INK_MUTED, linewidth=1, alpha=0.5, zorder=2)
    gt_point = ax_heat.scatter([], [], color=INK_PRIMARY, marker="x", s=90, zorder=4, label="ground truth")
    est_point = ax_heat.scatter([], [], color=RED, marker="o", s=50, zorder=4, label="estimated (peak)")
    ax_heat.set_facecolor(SURFACE)
    ax_heat.set_title("recalled position (similarity heatmap)", color=INK_PRIMARY, fontsize=11, pad=8)
    ax_heat.set_xlabel("x (m)", color=INK_MUTED, fontsize=8)
    ax_heat.set_ylabel("z (m)", color=INK_MUTED, fontsize=8)
    ax_heat.tick_params(colors=INK_MUTED, labelsize=7)
    legend = ax_heat.legend(loc="upper right", fontsize=7, facecolor=SURFACE, edgecolor=GRID)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)
    cbar = fig.colorbar(heat_im, ax=ax_heat, fraction=0.046, pad=0.04)
    cbar.set_label("similarity", color=INK_SECONDARY, fontsize=8)
    cbar.ax.tick_params(colors=INK_MUTED, labelsize=6)

    (polar_line,) = ax_polar.plot(angles, np.zeros_like(angles), color=BLUE, linewidth=1.5, zorder=2)
    (gt_heading_line,) = ax_polar.plot([0, 0], [0, 1], color=INK_PRIMARY, linewidth=2, linestyle="--",
                                       zorder=3, label="ground truth")
    est_heading_point = ax_polar.scatter([0], [0], color=RED, s=50, zorder=4, label="estimated (peak)")
    ax_polar.set_facecolor(SURFACE)
    ax_polar.set_title("recalled heading (similarity vs. angle)", color=INK_PRIMARY, fontsize=11, pad=14)
    ax_polar.tick_params(colors=INK_MUTED, labelsize=7)
    legend2 = ax_polar.legend(loc="upper right", fontsize=7, facecolor=SURFACE, edgecolor=GRID)
    for text in legend2.get_texts():
        text.set_color(INK_SECONDARY)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=fps, metadata={"title": "Freiburg associative memory recall demo"})
    with writer.saving(fig, str(out_path), dpi=120):
        for i, row_i in enumerate(val_row_idx):
            query_content = content[row_i]

            residual_pos = memory_position / query_content
            heat_sims = phasor_cross_correlation(residual_pos[None, :], grid_codes)[0].reshape(heat_shape)
            heat_im.set_data(heat_sims)
            heat_im.set_clim(float(heat_sims.min()), float(heat_sims.max()))
            peak_row, peak_col = np.unravel_index(np.argmax(heat_sims), heat_shape)
            gt_point.set_offsets([[frames["x"][row_i], frames["y"][row_i]]])
            est_point.set_offsets([[grid_x[peak_col], grid_y[peak_row]]])

            residual_heading = memory_heading / query_content
            heading_sims = phasor_cross_correlation(residual_heading[None, :], angle_codes)[0]
            shifted = heading_sims - heading_sims.min()
            polar_line.set_data(angles, shifted)
            r_max = float(shifted.max()) * 1.05 if shifted.max() > 0 else 1.0
            ax_polar.set_ylim(0, r_max)
            gt_yaw = float(frames["yaw"][row_i])
            gt_heading_line.set_data([gt_yaw, gt_yaw], [0, r_max])
            est_angle = angles[np.argmax(shifted)]
            est_heading_point.set_offsets([[est_angle, shifted.max()]])

            residual_time = memory_time / query_content
            time_sims = phasor_cross_correlation(residual_time[None, :], train_time_codes)[0]
            best_train_i = int(np.argmax(time_sims))
            recalled_frame_idx = int(codebook["frame_idx"][best_train_i])
            recalled_row = int(codebook["row_idx"][best_train_i])
            dt_s = abs(int(frames["timestamp_ns"][row_i]) - int(frames["timestamp_ns"][recalled_row])) / 1e9

            frame_num = int(frames["dataset_frame_idx"][row_i])
            rgb_im.set_data(np.asarray(load_rgb_image(rgb_dir, frame_num)))
            box_artists = draw_boxes(ax_rgb, box_artists, det_by_frame.get(frame_num))

            fig.suptitle(f"held-out frame {i + 1}/{len(val_row_idx)} (row {row_i})  --  "
                        f"recalled time: ~frame {recalled_frame_idx}, {dt_s:.1f}s away",
                        color=INK_PRIMARY, fontsize=12)
            writer.grab_frame()

            if (i + 1) % 100 == 0 or i + 1 == len(val_row_idx):
                print(f"[{i + 1}/{len(val_row_idx)}] rendered")

    plt.close(fig)
    print(f"demo video written to {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="bind content to context and bundle into 3 memories")
    build.add_argument("--gt-path", default=GT_PATH)
    build.add_argument("--embedding-model", choices=["yolo", "dino"], default="yolo",
                       help="embedding backend --embeddings was produced with; only changes "
                            "the default --embeddings input path and default --out output path "
                            "(adds a _dino suffix), plus provenance recorded in the saved "
                            "memory .pt -- this script is otherwise embedding-dimension-agnostic")
    build.add_argument("--dino-model", default="dinov2_vits14",
                       help="informational only: recorded as provenance in the saved memory "
                            ".pt when --embedding-model dino")
    build.add_argument("--embeddings", default=None,
                       help="default: outputs/freiburg_detections/embeddings.pt, or "
                            "embeddings_dino.pt if --embedding-model dino")
    build.add_argument("--limit", type=int, default=None, help="only use the first N frames")
    build.add_argument("--out", default=None,
                       help="default: outputs/freiburg_detections/associative_memory_<subset>.pt, "
                            "or ..._<subset>_dino.pt if --embedding-model dino")
    build.add_argument("--subset", choices=["all", "uncertain", "stride"], default="all",
                       help="'all': bundle every frame. 'uncertain': bundle only the top "
                            "--uncertainty-quantile fraction by embedding uncertainty. "
                            "'stride': bundle every --stride-th frame (a train split), "
                            "leaving the rest held out for the `demo` command to query")
    build.add_argument("--stride", type=int, default=3,
                       help="train-set spacing for --subset stride, e.g. 3 keeps every third "
                            "frame and holds out the other two thirds")
    build.add_argument("--uncertainty-type", choices=["local", "global"], default="global",
                       help="local: big frame-to-frame jumps. global: atypical relative to "
                            "the whole video (used only with --subset uncertain)")
    build.add_argument("--uncertainty-quantile", type=float, default=0.8,
                       help="keep frames at or above this quantile of uncertainty, e.g. 0.8 "
                            "keeps the top 20%% (used only with --subset uncertain)")
    build.add_argument("--trim-stationary", action="store_true",
                       help="exclude a stationary prefix/suffix from whichever --subset gets "
                            "bundled, detected from ground-truth displacement relative to a "
                            "start/end anchor position. Recorded in the saved memory .pt so "
                            "`demo` reuses the same trim automatically")
    build.add_argument("--trim-anchor-frames", type=int, default=15,
                       help="number of frames at each end used to compute the stationary anchor "
                            "position (median x,y) -- used only with --trim-stationary")
    build.add_argument("--trim-distance-threshold", type=float, default=0.1,
                       help="meters from the start/end anchor position beyond which a frame "
                            "counts as 'moving' rather than stationary -- used only with "
                            "--trim-stationary")
    build.add_argument("--pca-whiten", action="store_true",
                       help="PCA-whiten the raw embeddings (center, decorrelate, unit-variance "
                            "per component) before projecting to phasor content")
    build.add_argument("--pca-components", type=int, default=None,
                       help="components to keep when --pca-whiten is set; default keeps all "
                            "(full-rank whitening, no dimensionality reduction)")
    build.add_argument("--hd-dim", type=int, default=2048)
    build.add_argument("--content-seed", type=int, default=0)
    build.add_argument("--x-seed", type=int, default=1)
    build.add_argument("--y-seed", type=int, default=2)
    build.add_argument("--yaw-seed", type=int, default=3)
    build.add_argument("--yaw-max-freq", type=int, default=3)
    build.add_argument("--time-seed", type=int, default=42)
    build.add_argument("--pos-length-scale", type=float, default=1.0)
    build.add_argument("--time-length-scale", type=float, default=50.0)

    query = sub.add_parser("query", help="recall what was seen at a position/heading/time")
    query.add_argument("--memory", required=True, help="path to a .pt written by `build`")
    query.add_argument("--embedding-model", choices=["yolo", "dino"], default="yolo",
                       help="only used to pick the default --out-path suffix (_dino); the "
                            "actual backend is read back from --memory's saved provenance and "
                            "printed")
    query.add_argument("--rgb-dir", default=RGB_DIR)
    query.add_argument("--gt-path", default=GT_PATH)
    query.add_argument("--detections", default="outputs/freiburg_detections/detections.csv")
    query.add_argument("--query-x", type=float, default=None)
    query.add_argument("--query-y", type=float, default=None, help="ground-truth tz (this dataset's z axis)")
    query.add_argument("--query-yaw", type=float, default=None, help="radians")
    query.add_argument("--query-time", type=float, default=None, help="row index into the posed frame sequence")
    query.add_argument("--top-k", type=int, default=3)
    query.add_argument("--out-path", default=None,
                       help="default: outputs/freiburg_detections/query_result.png, or "
                            "..._dino.png if --embedding-model dino")

    evaluate = sub.add_parser("evaluate", help="self-recall physical error: query each stored frame's own context")
    evaluate.add_argument("--memory", required=True, help="path to a .pt written by `build`")
    evaluate.add_argument("--embedding-model", choices=["yolo", "dino"], default="yolo",
                          help="only used to pick the default --out-path suffix (_dino); the "
                               "actual backend is read back from --memory's saved provenance "
                               "and printed")
    evaluate.add_argument("--out-path", default=None,
                          help="default: outputs/freiburg_detections/evaluate_result.png, or "
                               "..._dino.png if --embedding-model dino")

    demo = sub.add_parser("demo", help="video: 'have I seen this before?' over held-out (val) frames")
    demo.add_argument("--memory", required=True,
                      help="path to a .pt written by `build --subset stride` (or any build -- "
                           "held-out frames are just whichever weren't bundled in)")
    demo.add_argument("--rgb-dir", default=RGB_DIR)
    demo.add_argument("--gt-path", default=GT_PATH)
    demo.add_argument("--embedding-model", choices=["yolo", "dino"], default="yolo",
                      help="embedding backend --embeddings was produced with; only changes the "
                           "default --embeddings input path and default --out-path (adds a "
                           "_dino suffix) -- the memory's own saved provenance is read back "
                           "from --memory and printed for reference")
    demo.add_argument("--embeddings", default=None,
                      help="default: outputs/freiburg_detections/embeddings.pt, or "
                           "embeddings_dino.pt if --embedding-model dino")
    demo.add_argument("--detections", default="outputs/freiburg_detections/detections.csv")
    demo.add_argument("--grid-resolution", type=int, default=60,
                      help="position heatmap grid size (grid-resolution x grid-resolution)")
    demo.add_argument("--n-angles", type=int, default=180, help="heading polar plot angle samples")
    demo.add_argument("--fps", type=float, default=None,
                      help="output video frame rate (default: matches the held-out frames' own "
                           "synthetic capture rate, see detect_and_embed_freiburg.py's --fps)")
    demo.add_argument("--limit", type=int, default=None, help="only render the first N held-out frames")
    demo.add_argument("--out-path", default=None,
                      help="default: outputs/freiburg_detections/demo.mp4, or ..._dino.mp4 if "
                           "--embedding-model dino")

    args = parser.parse_args()
    if args.command == "build":
        cmd_build(args)
    elif args.command == "query":
        cmd_query(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "demo":
        cmd_demo(args)


if __name__ == "__main__":
    main()
