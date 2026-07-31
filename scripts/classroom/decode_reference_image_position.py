"""Unbind one or more new, out-of-dataset images' embeddings from a position
associative memory, decode each to an (x, y) estimate, and check how close
those land to the robot's actual position whenever it really saw the same
class.

    python scripts/classroom/decode_reference_image_position.py \\
        --image-dir data/scissors_coco --reference-class scissors \\
        --memory outputs/classroom_dino_comparison/associative_memory_all.pt \\
        --embeddings outputs/classroom_dino_comparison/embeddings.pt \\
        --detections outputs/classroom_dino_comparison/detections.csv

Runs the same YOLO embedding extraction as compare_reference_image.py on
every image (--image, --image-dir, or both together), then projects each
into phasor content the same way classroom_associative_memory.py's
build/demo do: random_project_to_phasor using the memory's own saved W. If
the memory was built with --pca-whiten, build never persists the fitted PCA
transform (mean/basis/std) itself -- only the projected training scores --
so there is no way to apply the *exact* training-time whitening to brand
new samples. This script approximates it by refitting PCA once, jointly,
over every new image plus every training embedding in --embeddings; with a
few thousand existing samples, adding a handful more perturbs the fitted
basis only slightly, and fitting all new images together (rather than one
refit per image) keeps them on a single consistent basis.

The unbind + decode step (residual = memory_position / content, then a
dense position-grid argmax) is exactly classroom_associative_memory.py's
demo command's position recall, just run once per new image instead of
once per held-out video frame. With multiple images, this also reports
whether their *average* decoded position (the centroid) lands closer to
the real sightings than individual images typically do -- the same
noise-reduction-through-bundling idea the rest of this project relies on.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from vsa_cognitive_mapping.vsa import Phasor, pca_components, phasor_cross_correlation, random_project_to_phasor

REPO = "lorinachey/spot-telluride-workshop-dataset"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BLUE = "#2a78d6"
RED = "#e34948"


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_yolo_model(yolo_model: str):
    from ultralytics import YOLO
    return YOLO(yolo_model)


def nearest_index(sorted_ts: np.ndarray, query_ts: int) -> int:
    i = np.searchsorted(sorted_ts, query_ts)
    if i == 0:
        return 0
    if i == len(sorted_ts):
        return len(sorted_ts) - 1
    before, after = sorted_ts[i - 1], sorted_ts[i]
    return int(i - 1 if query_ts - before <= after - query_ts else i)


def encode_position(xy: np.ndarray, hd_dim: int, x_seed: int, y_seed: int, length_scale: float) -> np.ndarray:
    Bx = Phasor(dim=hd_dim, seed=x_seed)
    By = Phasor(dim=hd_dim, seed=y_seed)
    return np.stack([
        ((Bx ** float(x / length_scale)) * (By ** float(y / length_scale))).values
        for x, y in xy
    ])


def position_grid_codes(x_range: tuple[float, float], y_range: tuple[float, float], resolution: int,
                        hd_dim: int, x_seed: int, y_seed: int, length_scale: float):
    grid_x = np.linspace(*x_range, resolution)
    grid_y = np.linspace(*y_range, resolution)
    gx, gy = np.meshgrid(grid_x, grid_y)
    xy = np.stack([gx.ravel(), gy.ravel()], axis=1)
    codes = encode_position(xy, hd_dim, x_seed, y_seed, length_scale)
    return grid_x, grid_y, codes


def collect_image_paths(images: list[str] | None, image_dir: str | None) -> list[Path]:
    paths = [Path(p) for p in (images or [])]
    if image_dir is not None:
        paths += sorted(p for p in Path(image_dir).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise SystemExit("no images given -- pass --image (one or more) and/or --image-dir")
    return paths


def decode_position(residual_pos: np.ndarray, grid_x: np.ndarray, grid_y: np.ndarray,
                    grid_codes: np.ndarray, grid_resolution: int) -> tuple[float, float]:
    sims = phasor_cross_correlation(residual_pos[None, :], grid_codes)[0].reshape(grid_resolution, grid_resolution)
    peak_row, peak_col = np.unravel_index(np.argmax(sims), sims.shape)
    return float(grid_x[peak_col]), float(grid_y[peak_row])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", nargs="+", default=None,
                        help="one or more paths to new, out-of-dataset images")
    parser.add_argument("--image-dir", default=None,
                        help="a directory of images to include alongside/instead of --image")
    parser.add_argument("--reference-class", default="scissors",
                        help="COCO class to compare the decoded position(s) against, e.g. 'scissors'")
    parser.add_argument("--memory", required=True,
                        help="associative memory .pt from classroom_associative_memory.py build")
    parser.add_argument("--embeddings", default=None,
                        help="the training embeddings.pt the memory was built from -- required if "
                             "the memory used --pca-whiten, to refit a consistent whitening "
                             "transform for the new out-of-sample images (see module docstring)")
    parser.add_argument("--detections", default="outputs/classroom_detections/detections.csv")
    parser.add_argument("--repo", default=REPO)
    parser.add_argument("--odom-config", default="odometry_lio_sam")
    parser.add_argument("--yolo-model", default="yolov8n.pt")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--grid-resolution", type=int, default=100,
                        help="position decode grid size (grid-resolution x grid-resolution) -- run "
                             "only once per image here (not once per video frame like demo), so "
                             "this can afford to be finer than demo's default")
    parser.add_argument("--out-path", default=None,
                        help="default: outputs/classroom_detections/decoded_position_<reference-class>.png")
    args = parser.parse_args()

    image_paths = collect_image_paths(args.image, args.image_dir)
    print(f"{len(image_paths)} image(s) to decode")

    saved = torch.load(args.memory)
    if saved.get("embedding_model", "yolo") != "yolo":
        raise SystemExit(f"{args.memory} was built with embedding_model="
                         f"'{saved.get('embedding_model')}', but this script only extracts a YOLO "
                         "embedding for the new images -- point --memory at a yolo-built memory")
    hd_dim, bases, W = saved["hd_dim"], saved["bases"], saved["W"]

    device = resolve_device(args.device)
    print(f"loading {args.yolo_model} on device={device}...")
    model = load_yolo_model(args.yolo_model)

    images = [np.asarray(Image.open(p).convert("RGB")) for p in image_paths]
    print("running detection + extracting embeddings for every image...")
    detections = model.predict(images, device=device, conf=0.25, verbose=False)
    n_with_own_detection = sum(1 for r in detections if len(r.boxes) > 0)
    print(f"YOLO itself detected something in {n_with_own_detection}/{len(images)} of the new images")
    raw_embeddings = np.stack([e.cpu().numpy() for e in model.embed(images, device=device, verbose=False)])

    if saved.get("pca_whiten"):
        if args.embeddings is None:
            raise SystemExit(f"{args.memory} was built with --pca-whiten, so --embeddings is required "
                             "(the training embeddings.pt to refit a consistent whitening transform "
                             "against -- see module docstring)")
        train_embeddings = torch.load(args.embeddings)["embedding"].numpy()
        if train_embeddings.shape[1] != raw_embeddings.shape[1]:
            raise SystemExit(f"embedding dim mismatch: {args.embeddings} has {train_embeddings.shape[1]}"
                             f"-dim embeddings but the new images' YOLO embeddings are "
                             f"{raw_embeddings.shape[1]}-dim")
        combined = np.concatenate([raw_embeddings, train_embeddings], axis=0)
        whitened, _ = pca_components(combined, saved["pca_components"])
        content_input = whitened[:len(image_paths)].astype(np.float32)
        print(f"re-fit PCA once, jointly, with {len(train_embeddings)} training embeddings + all "
             f"{len(image_paths)} new images (build doesn't persist the fitted transform itself, so "
             "this is an approximation of it -- see module docstring)")
    else:
        content_input = raw_embeddings.astype(np.float32)

    content, _ = random_project_to_phasor(torch.from_numpy(content_input), d=hd_dim, W=W)
    content = content.numpy()

    memory_position = saved["memory_position"].numpy()

    print(f"loading {args.repo} ({args.odom_config})...")
    odom = load_dataset(args.repo, args.odom_config, split="train").sort("timestamp_ns")
    odom_ts = np.array(odom["timestamp_ns"])
    odom_x, odom_y = np.array(odom["x"]), np.array(odom["y"])

    pad = 1.0
    x_range = (float(odom_x.min() - pad), float(odom_x.max() + pad))
    y_range = (float(odom_y.min() - pad), float(odom_y.max() + pad))
    grid_x, grid_y, grid_codes = position_grid_codes(x_range, y_range, args.grid_resolution, hd_dim,
                                                     bases["x_seed"], bases["y_seed"], bases["pos_length_scale"])

    decoded = [decode_position(memory_position / content[i], grid_x, grid_y, grid_codes, args.grid_resolution)
              for i in range(len(image_paths))]
    decoded_x = np.array([d[0] for d in decoded])
    decoded_y = np.array([d[1] for d in decoded])
    centroid_x, centroid_y = float(decoded_x.mean()), float(decoded_y.mean())

    det = pd.read_csv(args.detections)
    class_rows = det[det["class_name"].str.lower() == args.reference_class.lower()]
    if class_rows.empty:
        available = ", ".join(sorted(det["class_name"].unique()))
        raise SystemExit(f"no '{args.reference_class}' detections found in {args.detections}\n"
                         f"available classes: {available}")
    actual_x = np.array([odom_x[nearest_index(odom_ts, int(ts))] for ts in class_rows["timestamp_ns"]])
    actual_y = np.array([odom_y[nearest_index(odom_ts, int(ts))] for ts in class_rows["timestamp_ns"]])

    # Null baseline: for every point the robot ever visited, how far (on
    # average) is it from the actual sighting positions? Comparing each
    # decode's own mean distance against this null distribution answers "is
    # this meaningfully closer to the real sightings than a typical point on
    # the trajectory would be" rather than something explained just by the
    # sightings being clustered somewhere much of the trajectory passes near.
    null_mean_dists = np.hypot(odom_x[:, None] - actual_x[None, :],
                               odom_y[:, None] - actual_y[None, :]).mean(axis=1)

    print(f"\nper-image decode vs. the {len(actual_x)} actual '{args.reference_class}' sighting positions:")
    mean_dist_per_image = np.empty(len(image_paths))
    for i, path in enumerate(image_paths):
        dists = np.hypot(actual_x - decoded_x[i], actual_y - decoded_y[i])
        mean_dist_per_image[i] = dists.mean()
        percentile = 100 * (null_mean_dists > dists.mean()).mean()
        print(f"  {path.name}: decoded=({decoded_x[i]:.2f}, {decoded_y[i]:.2f}) "
             f"mean_dist={dists.mean():.2f}m (closer than {percentile:.0f}% of all trajectory poses)")

    centroid_dists = np.hypot(actual_x - centroid_x, actual_y - centroid_y)
    centroid_percentile = 100 * (null_mean_dists > centroid_dists.mean()).mean()
    print(f"\ncentroid of all {len(image_paths)} decoded positions: ({centroid_x:.2f}, {centroid_y:.2f})")
    print(f"  mean_dist={centroid_dists.mean():.2f}m (closer than {centroid_percentile:.0f}% of all "
         "trajectory poses)")
    print(f"  individual images' mean_dist: average={mean_dist_per_image.mean():.2f}m, "
         f"best={mean_dist_per_image.min():.2f}m, worst={mean_dist_per_image.max():.2f}m")
    print(f"  {'centroid IS closer than the average individual image' if centroid_dists.mean() < mean_dist_per_image.mean() else 'centroid is NOT closer than the average individual image'} "
         "-- the bundling/averaging effect this project's memories rely on elsewhere")

    out_path = Path(args.out_path) if args.out_path else \
        Path(f"outputs/classroom_detections/decoded_position_{args.reference_class}.png")

    fig, ax = plt.subplots(figsize=(8, 7), facecolor=SURFACE, constrained_layout=True)
    ax.plot(odom_x, odom_y, color=INK_MUTED, linewidth=1, alpha=0.5, zorder=1, label="robot trajectory")
    ax.scatter(actual_x, actual_y, color=BLUE, s=40, zorder=2, alpha=0.7,
              label=f"actual robot position when '{args.reference_class}' seen (n={len(actual_x)})")
    ax.scatter(decoded_x, decoded_y, color=RED, marker="*", s=140, zorder=3, alpha=0.7,
              edgecolors=INK_PRIMARY, linewidths=0.5,
              label=f"decoded from each new image (n={len(image_paths)})")
    if len(image_paths) > 1:
        ax.scatter([centroid_x], [centroid_y], color=RED, marker="*", s=500, zorder=4,
                  edgecolors=INK_PRIMARY, linewidths=1.2, label="centroid of all decoded positions")
    ax.set_facecolor(SURFACE)
    ax.set_aspect("equal")
    ax.set_title(f"{len(image_paths)} new image(s) unbound from position memory,\n"
                f"vs. actual robot positions when '{args.reference_class}' was detected",
                color=INK_PRIMARY, fontsize=12, pad=10)
    ax.set_xlabel("x (m)", color=INK_MUTED, fontsize=9)
    ax.set_ylabel("y (m)", color=INK_MUTED, fontsize=9)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    legend = ax.legend(loc="best", fontsize=8, facecolor=SURFACE, edgecolor=GRID)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"\nplot written to {out_path}")


if __name__ == "__main__":
    main()
