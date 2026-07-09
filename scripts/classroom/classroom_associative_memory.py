"""Bind YOLO content to position/heading/time context, bundle into three
separate associative memories, and recall "what did I see at this
position/heading/time?" by unbind + cleanup lookup.

    python scripts/classroom/classroom_associative_memory.py build --subset all
    python scripts/classroom/classroom_associative_memory.py build --subset uncertain
    python scripts/classroom/classroom_associative_memory.py query --memory outputs/classroom_detections/associative_memory_all.pt --query-x -2.0 --query-y 3.0
    python scripts/classroom/classroom_associative_memory.py evaluate --memory outputs/classroom_detections/associative_memory_all.pt

For each frame i, content_i = random_project_to_phasor(embedding_i) (see
scripts/classroom/detect_and_embed_classroom.py for the raw embeddings) and
context_i = one of three independent phasor codes -- position (Bx**x * By**y),
heading (a circular-safe base**yaw), or time (Bt**row_idx) -- computed with
the exact same encoders as scripts/classroom/plot_pose_time_heading_correlation.py.
Three SEPARATE memories are built, one per axis:

    memory_<axis> = bundle(content_1 * context_1, ..., content_M * context_M)

kept apart rather than bound into one combined quad-binding, so each only
has to discriminate along its own axis. Recall (`query`) unbinds a memory by
a candidate context to get a noisy residual, then runs cleanup
(vsa.py's best_matches) against a codebook of the content vectors that were
actually bundled in, restricted to the frames used for that memory's build.

`build --subset all` bundles every frame -- general recall, and a real test
of whether ~2,478 distinct items overwhelm an hd-dim=256 memory (bundling
capacity is roughly proportional to dimensionality, so this is expected to
be noisy). `build --subset uncertain` bundles only the top fraction of
frames by embedding uncertainty (scripts/classroom/plot_embedding_uncertainty.py's
adjacent_frame_uncertainty/global_frame_uncertainty over the raw-embedding
cosine self-correlation) -- fewer, more distinctive items bundled together
should mean less crosstalk. `evaluate` measures this directly: for each
memory, query every stored frame's own exact context and measure how far
(L1 mean / L2 root-mean-square, in physical units -- meters, radians,
frames) the top-1 recalled frame's true context is from the query -- exact
top-1/top-k match rate turned out to be too harsh a measure on its own
(near-duplicate frames make an "almost right" recall look identical to a
wildly wrong one), so physical error is what's reported and compared across
`all` vs. `uncertain` builds.

`build --subset stride --stride 3` bundles every third frame -- a train
split -- leaving the other two thirds held out. `demo` runs the *reverse*
query direction over those held-out frames: instead of context -> content
("what did I see at this position?"), it's content -> context ("where/when
have I seen something like this before?") -- unbind each memory by the
*current* frame's own content to get a residual, then instead of collapsing
to a single best-guess point, evaluate that residual's similarity
continuously over a dense grid of candidate positions (a heatmap) and a
dense sweep of candidate headings (a polar plot), so genuine ambiguity
(two similar-looking spots) is visible rather than hidden behind an
arbitrary argmax. Time recall stays a discrete lookup against the train
codebook (recalling a specific past frame, not a continuous quantity).
Renders one video frame per held-out frame, in chronological order --
Spot moving through the room while the memory answers "have I been here
before?" the whole way:

    python scripts/classroom/classroom_associative_memory.py build --subset stride --stride 3
    python scripts/classroom/classroom_associative_memory.py demo --memory outputs/classroom_detections/associative_memory_stride.pt
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.colors as mcolors  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from vsa_cognitive_mapping.vsa import (
    Phasor,
    best_matches,
    cosine_self_correlation,
    pca_components,
    phasor_correlation_matrix,
    phasor_cross_correlation,
    random_project_to_phasor,
)

REPO = "lorinachey/spot-telluride-workshop-dataset"

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BLUE = "#2a78d6"
RED = "#e34948"
CLASS_CMAP = plt.get_cmap("tab20")

# Signed [-1, 1] correlation -> diverging, same validated pair
# scripts/associative_memory.py, scripts/encoder_sweep.py, and the other
# scripts/classroom/ scripts each define locally.
DIVERGING_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#2a78d6", "#f0efec", "#e34948"])

AXES = ("position", "heading", "time")


# ---------------------------------------------------------------------------
# Helpers duplicated from sibling classroom scripts (repo convention: each
# script stays self-contained rather than importing another script).
# ---------------------------------------------------------------------------

def quat_to_yaw(qx: np.ndarray, qy: np.ndarray, qz: np.ndarray, qw: np.ndarray) -> np.ndarray:
    return np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy ** 2 + qz ** 2))


def nearest_index(sorted_ts: np.ndarray, query_ts: int) -> int:
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
    base = Phasor(dim=hd_dim, seed=seed, circular=True, max_freq=max_freq)
    return np.stack([(base ** float(a)).values for a in yaw])


def heading_sweep_codes(hd_dim: int, seed: int, max_freq: int, n_angles: int) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the heading base at a dense sweep of candidate angles ->
    (angles, codes), for a continuous similarity-vs-heading curve rather
    than only the handful of angles that happen to appear in the codebook.
    Same encoder encode_heading uses per frame, just swept densely."""
    angles = np.linspace(-np.pi, np.pi, n_angles, endpoint=False)
    return angles, encode_heading(angles, hd_dim, seed, max_freq)


def position_grid_codes(x_range: tuple[float, float], y_range: tuple[float, float], resolution: int,
                        hd_dim: int, x_seed: int, y_seed: int, length_scale: float):
    """Evaluate the position code on a dense (resolution x resolution) grid
    over the given extent -> (grid_x, grid_y, codes), for a continuous
    similarity heatmap rather than only the discrete codebook positions.
    Same encoder encode_position uses per frame, just swept over a grid."""
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


def load_detections_by_frame(det_path: Path) -> dict[int, pd.DataFrame]:
    df = pd.read_csv(det_path)
    return {int(frame_idx): group for frame_idx, group in df.groupby("frame_idx")}


def draw_boxes(ax, prior_artists: list, frame_dets: pd.DataFrame | None) -> list:
    """Remove the previous frame's boxes/labels and draw this frame's,
    returning the new artists -- safe to call once for a static plot
    (prior_artists=[]) or every frame of a video (pass back the return
    value each time, matching detect_and_embed_classroom.py's draw_boxes)."""
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

def load_frame_data(repo: str, d455_rgb_config: str, odom_config: str, embeddings_path: str,
                    limit: int | None) -> dict:
    print(f"loading {embeddings_path}...")
    data = torch.load(embeddings_path)
    embeddings_t = data["embedding"]
    dataset_frame_idx = data["frame_idx"].numpy()
    n_frames = len(embeddings_t) if limit is None else min(limit, len(embeddings_t))
    embeddings_t = embeddings_t[:n_frames]
    dataset_frame_idx = dataset_frame_idx[:n_frames]

    print(f"loading {repo} ({d455_rgb_config}, {odom_config})...")
    rgb = load_dataset(repo, d455_rgb_config, split="train").sort("timestamp_ns")
    odom = load_dataset(repo, odom_config, split="train").sort("timestamp_ns")

    rgb_ts = np.array(rgb["timestamp_ns"][:n_frames])
    # Row position, not the dataset's own frame_idx field -- see the fix in
    # plot_pose_time_heading_correlation.py: frame_idx isn't monotonic once
    # sorted by timestamp_ns, timestamp_ns is.
    row_idx = np.arange(n_frames)

    odom_ts = np.array(odom["timestamp_ns"])
    odom_x, odom_y = np.array(odom["x"]), np.array(odom["y"])
    odom_yaw = quat_to_yaw(np.array(odom["qx"]), np.array(odom["qy"]),
                           np.array(odom["qz"]), np.array(odom["qw"]))
    odom_i = np.array([nearest_index(odom_ts, ts) for ts in rgb_ts])

    return {
        "embeddings_t": embeddings_t,
        "row_idx": row_idx,
        "x": odom_x[odom_i], "y": odom_y[odom_i], "yaw": odom_yaw[odom_i],
        "timestamp_ns": rgb_ts,
        "dataset_frame_idx": dataset_frame_idx,
        "n_frames": n_frames,
        "odom_x": odom_x, "odom_y": odom_y,  # full trajectory, for plotting
    }


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def select_subset(embeddings: np.ndarray, uncertainty_type: str, quantile: float) -> np.ndarray:
    """Row indices of the top (1 - quantile) fraction of frames by embedding
    uncertainty over the raw-embedding cosine self-correlation matrix."""
    corr = cosine_self_correlation(embeddings)
    uncertainty = adjacent_frame_uncertainty(corr) if uncertainty_type == "local" else global_frame_uncertainty(corr)
    threshold = np.quantile(uncertainty, quantile)
    return np.where(uncertainty >= threshold)[0]


def bundle_memory(content: np.ndarray, context: np.ndarray, subset_idx: np.ndarray) -> np.ndarray:
    """content, context: (N, hd_dim) complex, aligned by row. Bind each
    subset row (elementwise complex multiply) then bundle (mean) -> (hd_dim,)."""
    traces = content[subset_idx] * context[subset_idx]
    return traces.mean(axis=0)


def cmd_build(args: argparse.Namespace) -> None:
    frames = load_frame_data(args.repo, args.d455_rgb_config, args.odom_config, args.embeddings, args.limit)
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

    if args.subset == "all":
        subset_idx = np.arange(n_frames)
    elif args.subset == "stride":
        # Train/val split: every args.stride-th frame is "seen" (bundled into
        # the memory); the rest are held out, for scripts/classroom's `demo`
        # command to query with content the memory was never directly given.
        subset_idx = np.arange(0, n_frames, args.stride)
    else:
        raw_embeddings = frames["embeddings_t"].numpy()
        subset_idx = select_subset(raw_embeddings, args.uncertainty_type, args.uncertainty_quantile)
    print(f"bundling {len(subset_idx)}/{n_frames} frames ({args.subset}) into each memory")

    memory_position = bundle_memory(content, pos_codes, subset_idx)
    memory_heading = bundle_memory(content, heading_codes, subset_idx)
    memory_time = bundle_memory(content, time_codes, subset_idx)

    result = {
        "subset": args.subset,
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

    out_path = Path(args.out) if args.out else Path(f"outputs/classroom_detections/associative_memory_{args.subset}.pt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, out_path)
    print(f"memory written to {out_path}")


# ---------------------------------------------------------------------------
# query
# ---------------------------------------------------------------------------

def unbind_and_score(memory: np.ndarray, query_context: np.ndarray, codebook_content: np.ndarray) -> np.ndarray:
    """Unbind memory by query_context, score the residual against every
    codebook entry -- (len(codebook),) similarities, in codebook order."""
    residual = memory / query_context
    return phasor_cross_correlation(residual[None, :], codebook_content)[0]


def cmd_query(args: argparse.Namespace) -> None:
    if args.query_time is None and args.query_yaw is None and (args.query_x is None or args.query_y is None):
        raise SystemExit("query needs at least one of: --query-x AND --query-y, --query-yaw, --query-time")

    saved = torch.load(args.memory)
    hd_dim, bases = saved["hd_dim"], saved["bases"]
    codebook = saved["codebook"]
    codebook_content = codebook["content"].numpy()
    print(f"loaded {args.memory}: subset={saved['subset']}, {codebook_content.shape[0]} stored frames")

    det_by_frame = load_detections_by_frame(Path(args.detections)) if Path(args.detections).exists() else {}

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

    plot_query_result(args, saved, codebook, det_by_frame, axis_sims, best_idx)


def plot_query_result(args: argparse.Namespace, saved: dict, codebook: dict,
                      det_by_frame: dict, axis_sims: dict, best_idx: int) -> None:
    print(f"loading {args.d455_rgb_config} to fetch the recalled frame's image...")
    rgb = load_dataset(args.repo, args.d455_rgb_config, split="train").sort("timestamp_ns")
    row = int(codebook["row_idx"][best_idx])
    image = rgb[row]["image"].convert("RGB")

    odom = load_dataset(args.repo, args.odom_config, split="train").sort("timestamp_ns")
    odom_x, odom_y = np.array(odom["x"]), np.array(odom["y"])

    fig, (ax_pos, ax_rgb) = plt.subplots(1, 2, figsize=(14, 6), facecolor=SURFACE, constrained_layout=True)

    ax_pos.plot(odom_x, odom_y, color=BLUE, linewidth=1.5, alpha=0.6, zorder=1, label="trajectory")
    rx, ry = float(codebook["x"][best_idx]), float(codebook["y"][best_idx])
    ax_pos.scatter([rx], [ry], color=RED, s=80, zorder=3, label="recalled frame")
    if "position" in axis_sims:
        ax_pos.scatter([args.query_x], [args.query_y], color=INK_PRIMARY, marker="x", s=100, zorder=4, label="query")
        ax_pos.plot([args.query_x, rx], [args.query_y, ry], color=INK_MUTED, linewidth=1, linestyle=":", zorder=2)
    ax_pos.set_aspect("equal")
    ax_pos.set_facecolor(SURFACE)
    ax_pos.set_title("Recalled frame's position", color=INK_PRIMARY, fontsize=12, pad=8)
    ax_pos.set_xlabel("x (m)", color=INK_MUTED, fontsize=9)
    ax_pos.set_ylabel("y (m)", color=INK_MUTED, fontsize=9)
    ax_pos.tick_params(colors=INK_MUTED, labelsize=8)
    for spine in ax_pos.spines.values():
        spine.set_color(GRID)
    ax_pos.grid(color=GRID, linewidth=0.6)
    legend = ax_pos.legend(loc="best", fontsize=8, facecolor=SURFACE, edgecolor=GRID)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    ax_rgb.imshow(np.asarray(image))
    ax_rgb.axis("off")
    fidx = int(codebook["frame_idx"][best_idx])
    ax_rgb.set_title(f"Recalled: frame_idx={fidx}", color=INK_PRIMARY, fontsize=12, pad=8)
    draw_boxes(ax_rgb, [], det_by_frame.get(fidx))

    fig.suptitle(f"\"What did I see?\" -- memory subset={saved['subset']}", color=INK_PRIMARY, fontsize=13)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"\nquery plot written to {out_path}")


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

def circular_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Signed angular difference a - b, wrapped to [-pi, pi]."""
    return np.arctan2(np.sin(a - b), np.cos(a - b))


def l1_l2(errors: np.ndarray) -> tuple[float, float]:
    """Mean absolute error (L1) and root-mean-square error (L2) of a real,
    already-in-physical-units error array -- both in the same units, so
    directly comparable to each other (unlike raw MSE, which would be
    squared units)."""
    return float(np.mean(np.abs(errors))), float(np.sqrt(np.mean(errors ** 2)))


def physical_error(axis: str, codebook: dict, predicted_idx: np.ndarray) -> np.ndarray:
    """Per-item error between each item's own true context and its top-1
    recalled item's true context, in physical units: meters (Euclidean) for
    position, radians (circular-aware) for heading, frames for time."""
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
    print(f"evaluating {args.memory}: subset={saved['subset']}, n={n} stored frames")

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

        # Stage: bound vectors -- content_i * context_i, before bundling.
        bound = codebook_content * context_codes
        context_corr = phasor_correlation_matrix(context_codes)
        bound_corr = phasor_correlation_matrix(bound)
        # Stage: how similar is the final bundled memory to each individual
        # pre-bundle trace -- high everywhere would mean no crosstalk; a
        # trace far from the memory's average is one bundling drowned out.
        memory_vs_bound = phasor_cross_correlation(memory[None, :], bound)[0]
        per_axis_stages[axis] = (context_corr, bound_corr, memory_vs_bound)

        # Top-1 recall: unbind by each item's own true context, find the
        # closest codebook content vector. L1/L2 of that recalled item's
        # true context vs. the query's own true context, in physical units
        # -- how far off a miss actually was, not just whether it was one.
        residuals = memory[None, :] / context_codes
        sims = phasor_cross_correlation(residuals, codebook_content)  # (n, n)
        predicted_idx = np.argmax(sims, axis=1)
        item_errors = physical_error(axis, codebook, predicted_idx)
        l1, l2 = l1_l2(item_errors)
        errors[axis] = (l1, l2)

        unit = AXIS_UNITS[axis]
        print(f"[{axis}] L1={l1:.3f}{unit}  L2={l2:.3f}{unit}")

    plot_pipeline_diagnostic(embeddings_corr, per_axis_stages, errors, saved["subset"], Path(args.out_path))


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
    """One row per axis (position/heading/time), the pipeline made visible
    stage by stage: the (whitened or raw) embeddings' own self-correlation,
    that axis's context self-correlation, the bound (content*context)
    vectors' self-correlation, how similar the final bundled memory is to
    each individual pre-bundle trace, and that axis's own L1/L2 top-1
    physical recall error as the final cell (its units differ per axis, so
    each row gets its own small panel rather than one shared scale).
    """
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
    print(f"loaded {args.memory}: subset={saved['subset']}, {len(train_row_idx)} train frames")

    frames = load_frame_data(args.repo, args.d455_rgb_config, args.odom_config, args.embeddings, None)
    n_frames = frames["n_frames"]
    val_row_idx = np.setdiff1d(np.arange(n_frames), train_row_idx)
    if args.limit is not None:
        val_row_idx = val_row_idx[:args.limit]
    print(f"{len(val_row_idx)} held-out (val) frames to query, out of {n_frames} total")

    # Project every frame's embedding with the SAME W (and PCA whitening, if
    # used) the memory was built with, so query content lands in the same
    # space as the codebook -- reusing W directly (not the seed) is what
    # keeps this the identical projection rather than a fresh random one.
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

    print(f"loading {args.d455_rgb_config} for video frames...")
    rgb = load_dataset(args.repo, args.d455_rgb_config, split="train").sort("timestamp_ns")

    val_ts = frames["timestamp_ns"][val_row_idx]
    if args.fps is None:
        duration_s = (val_ts[-1] - val_ts[0]) / 1e9 if len(val_ts) > 1 else 1.0
        fps = (len(val_row_idx) - 1) / duration_s if duration_s > 0 else 10.0
    else:
        fps = args.fps
    print(f"rendering {len(val_row_idx)} held-out frames at {fps:.2f} fps -> {args.out_path}")

    fig = plt.figure(figsize=(20, 6.5), facecolor=SURFACE, constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=(1, 1.2, 1))
    ax_rgb = fig.add_subplot(gs[0, 0])
    ax_heat = fig.add_subplot(gs[0, 1])
    ax_polar = fig.add_subplot(gs[0, 2], projection="polar")

    rgb_im = ax_rgb.imshow(np.asarray(rgb[int(val_row_idx[0])]["image"].convert("RGB")))
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
    ax_heat.set_ylabel("y (m)", color=INK_MUTED, fontsize=8)
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

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=fps, metadata={"title": "associative memory recall demo"})
    with writer.saving(fig, str(out_path), dpi=120):
        for i, row_i in enumerate(val_row_idx):
            query_content = content[row_i]

            # -- position: unbind, evaluate similarity over the whole grid --
            residual_pos = memory_position / query_content
            heat_sims = phasor_cross_correlation(residual_pos[None, :], grid_codes)[0].reshape(heat_shape)
            heat_im.set_data(heat_sims)
            heat_im.set_clim(float(heat_sims.min()), float(heat_sims.max()))
            peak_row, peak_col = np.unravel_index(np.argmax(heat_sims), heat_shape)
            gt_point.set_offsets([[frames["x"][row_i], frames["y"][row_i]]])
            est_point.set_offsets([[grid_x[peak_col], grid_y[peak_row]]])

            # -- heading: unbind, evaluate similarity over the angle sweep --
            residual_heading = memory_heading / query_content
            heading_sims = phasor_cross_correlation(residual_heading[None, :], angle_codes)[0]
            shifted = heading_sims - heading_sims.min()  # polar plots can't take negative r
            polar_line.set_data(angles, shifted)
            r_max = float(shifted.max()) * 1.05 if shifted.max() > 0 else 1.0
            ax_polar.set_ylim(0, r_max)
            gt_yaw = float(frames["yaw"][row_i])
            gt_heading_line.set_data([gt_yaw, gt_yaw], [0, r_max])
            est_angle = angles[np.argmax(shifted)]
            est_heading_point.set_offsets([[est_angle, shifted.max()]])

            # -- time: unbind, cleanup against the (discrete) train codebook --
            residual_time = memory_time / query_content
            time_sims = phasor_cross_correlation(residual_time[None, :], train_time_codes)[0]
            best_train_i = int(np.argmax(time_sims))
            recalled_frame_idx = int(codebook["frame_idx"][best_train_i])
            recalled_row = int(codebook["row_idx"][best_train_i])
            dt_s = abs(int(frames["timestamp_ns"][row_i]) - int(frames["timestamp_ns"][recalled_row])) / 1e9

            image_row = rgb[int(row_i)]
            rgb_im.set_data(np.asarray(image_row["image"].convert("RGB")))
            box_artists = draw_boxes(ax_rgb, box_artists,
                                     det_by_frame.get(int(frames["dataset_frame_idx"][row_i])))

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
    build.add_argument("--repo", default=REPO)
    build.add_argument("--d455-rgb-config", default="rgb_d455")
    build.add_argument("--odom-config", default="odometry_lio_sam")
    build.add_argument("--embeddings", default="outputs/classroom_detections/embeddings.pt")
    build.add_argument("--limit", type=int, default=None, help="only use the first N frames")
    build.add_argument("--out", default=None,
                       help="default: outputs/classroom_detections/associative_memory_<subset>.pt")
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
    build.add_argument("--pca-whiten", action="store_true",
                       help="PCA-whiten the raw embeddings (center, decorrelate, unit-variance "
                            "per component) before projecting to phasor content -- raw cosine "
                            "similarity across this dataset sits in a tight 0.70-1.00 band, so "
                            "whitening aims to spread out the discriminative structure that a "
                            "single dominant shared direction is currently swamping")
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
    query.add_argument("--repo", default=REPO)
    query.add_argument("--d455-rgb-config", default="rgb_d455")
    query.add_argument("--odom-config", default="odometry_lio_sam")
    query.add_argument("--detections", default="outputs/classroom_detections/detections.csv")
    query.add_argument("--query-x", type=float, default=None)
    query.add_argument("--query-y", type=float, default=None)
    query.add_argument("--query-yaw", type=float, default=None, help="radians")
    query.add_argument("--query-time", type=float, default=None, help="row index into the D455 sequence")
    query.add_argument("--top-k", type=int, default=3)
    query.add_argument("--out-path", default="outputs/classroom_detections/query_result.png")

    evaluate = sub.add_parser("evaluate", help="self-recall physical error: query each stored frame's own context")
    evaluate.add_argument("--memory", required=True, help="path to a .pt written by `build`")
    evaluate.add_argument("--out-path", default="outputs/classroom_detections/evaluate_result.png")

    demo = sub.add_parser("demo", help="video: 'have I seen this before?' over held-out (val) frames")
    demo.add_argument("--memory", required=True,
                      help="path to a .pt written by `build --subset stride` (or any build -- "
                           "held-out frames are just whichever weren't bundled in)")
    demo.add_argument("--repo", default=REPO)
    demo.add_argument("--d455-rgb-config", default="rgb_d455")
    demo.add_argument("--odom-config", default="odometry_lio_sam")
    demo.add_argument("--embeddings", default="outputs/classroom_detections/embeddings.pt")
    demo.add_argument("--detections", default="outputs/classroom_detections/detections.csv")
    demo.add_argument("--grid-resolution", type=int, default=60,
                      help="position heatmap grid size (grid-resolution x grid-resolution)")
    demo.add_argument("--n-angles", type=int, default=180, help="heading polar plot angle samples")
    demo.add_argument("--fps", type=float, default=None,
                      help="output video frame rate (default: matches the held-out frames' own "
                           "native capture rate)")
    demo.add_argument("--limit", type=int, default=None, help="only render the first N held-out frames")
    demo.add_argument("--out-path", default="outputs/classroom_detections/demo.mp4")

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
