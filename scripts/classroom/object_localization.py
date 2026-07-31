"""Shared backprojection logic for localizing detected objects to world (x, y)
using depth + odometry -- no matplotlib import here, so this module is safe
to share between a static-plot CLI (plot_object_locations.py) and a live
interactive viewer (interactive_object_locations.py) without either one's
choice of matplotlib backend fighting the other's.

For each detection box, looks up the nearest depth_d455 frame and
odometry_lio_sam pose by timestamp, takes the median valid depth within the
box (masking out no-return/out-of-range pixels), backprojects the box's
center column to a camera-frame (forward, lateral) offset using the D455's
published intrinsics, then rotates that offset into world coordinates by the
robot's odometry heading and adds it to the robot's odometry position.

The camera is approximated as coincident with the robot's own (x, y, yaw) --
this dataset does not publish a separate camera-to-base extrinsic (mount
height/tilt), only the robot's planar odometry pose, so this is accepted as
a reasonable approximation at classroom scale rather than a precise
transform.
"""

import numpy as np
import pandas as pd
from datasets import load_dataset

REPO = "lorinachey/spot-telluride-workshop-dataset"

# static/rgb_d455/camera_info.json (rectified, zero distortion) for this
# dataset's D455 -- fixed constants rather than fetched at runtime, since
# depth_d455 stores raw per-pixel range only, not a reusable calibration blob.
# Only fx/cx (horizontal) are needed: the memory's position representation is
# 2D (x, y), so the vertical camera axis (fy/cy) is intentionally discarded --
# see backproject_to_world.
D455_FX = 378.22161865234375
D455_CX = 320.43798828125

# Raw uint16 PNG depth is millimeters (RealSense convention). 0 = no return;
# this sensor saturates to 65535 past its usable range -- both excluded as
# invalid rather than treated as a real (65 m) reading.
DEPTH_SCALE_M = 1.0 / 1000.0
DEPTH_INVALID_MM = (0, 65535)


def nearest_index(sorted_ts: np.ndarray, query_ts: int) -> int:
    i = np.searchsorted(sorted_ts, query_ts)
    if i == 0:
        return 0
    if i == len(sorted_ts):
        return len(sorted_ts) - 1
    before, after = sorted_ts[i - 1], sorted_ts[i]
    return int(i - 1 if query_ts - before <= after - query_ts else i)


def quat_to_yaw(qx: np.ndarray, qy: np.ndarray, qz: np.ndarray, qw: np.ndarray) -> np.ndarray:
    return np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy ** 2 + qz ** 2))


def masked_box_depth(depth_mm: np.ndarray, x1: float, y1: float, x2: float,
                     y2: float) -> tuple[float | None, float]:
    """Median valid depth (meters) within a detection box -- the box's
    interior is mostly the detected object, so a median is already a crude
    foreground mask against the minority of background/no-return pixels a
    rectangular box picks up around an irregular silhouette.

    Returns (depth_m, valid_fraction): depth_m is None if every pixel in the
    box is invalid. valid_fraction (0-1) is returned even when depth_m isn't
    None, so callers can additionally reject boxes where the sensor mostly
    failed but happened to leave a few valid pixels behind -- see
    auto_min_valid_fraction for why a low fraction makes the median
    unreliable even when it isn't literally undefined."""
    h, w = depth_mm.shape
    xi1, yi1 = max(int(x1), 0), max(int(y1), 0)
    xi2, yi2 = min(int(np.ceil(x2)), w), min(int(np.ceil(y2)), h)
    crop = depth_mm[yi1:yi2, xi1:xi2]
    valid = crop[(crop > DEPTH_INVALID_MM[0]) & (crop < DEPTH_INVALID_MM[1])]
    valid_fraction = valid.size / crop.size if crop.size else 0.0
    if valid.size == 0:
        return None, valid_fraction
    return float(np.median(valid)) * DEPTH_SCALE_M, valid_fraction


def backproject_to_world(x1: float, x2: float, depth_m: float,
                         robot_x: float, robot_y: float, robot_yaw: float) -> tuple[float, float]:
    """Box center + depth -> world (x, y), approximating the camera as
    coincident with the robot's own odometry pose (see module docstring)."""
    u = (x1 + x2) / 2.0
    forward = depth_m
    lateral = (u - D455_CX) * depth_m / D455_FX  # camera-right-positive
    world_x = robot_x + forward * np.cos(robot_yaw) + lateral * np.sin(robot_yaw)
    world_y = robot_y + forward * np.sin(robot_yaw) - lateral * np.cos(robot_yaw)
    return float(world_x), float(world_y)


def load_alignment_data(repo: str, depth_config: str, odom_config: str) -> dict:
    """One-time load of the depth stream + odometry needed to localize any
    detection -- expensive (dataset load), so callers localizing more than
    one class (e.g. an interactive class picker) should load this once and
    reuse it across selections rather than reloading per class."""
    print(f"loading {repo} ({depth_config}, {odom_config})...")
    depth_ds = load_dataset(repo, depth_config, split="train").sort("timestamp_ns")
    odom = load_dataset(repo, odom_config, split="train").sort("timestamp_ns")
    odom_x, odom_y = np.array(odom["x"]), np.array(odom["y"])
    odom_yaw = quat_to_yaw(np.array(odom["qx"]), np.array(odom["qy"]),
                           np.array(odom["qz"]), np.array(odom["qw"]))
    return {
        "depth_ds": depth_ds,
        "depth_ts": np.array(depth_ds["timestamp_ns"]),
        "odom_ts": np.array(odom["timestamp_ns"]),
        "odom_x": odom_x, "odom_y": odom_y, "odom_yaw": odom_yaw,
    }


def localize_detections(det: pd.DataFrame, alignment: dict,
                        min_valid_fraction: float = 0.0) -> tuple[list, list, list, int]:
    """Backproject every row in det (already filtered to whichever class/
    confidence threshold the caller wants) to world (x, y), using the
    preloaded alignment data from load_alignment_data.

    min_valid_fraction rejects boxes where fewer than that fraction (0-1) of
    pixels had valid depth -- see auto_min_valid_fraction for how to
    calibrate this rather than guessing a number. Boxes with literally zero
    valid pixels are always rejected regardless of this threshold (there's
    no depth to even compute).

    Returns:
        world_x, world_y, confidences: parallel lists, one entry per
            successfully localized detection
        n_skipped: how many rows were rejected (no valid depth, or below
            min_valid_fraction)
    """
    world_x, world_y, kept_conf = [], [], []
    n_skipped = 0
    for _, row in det.iterrows():
        ts = int(row["timestamp_ns"])
        di = nearest_index(alignment["depth_ts"], ts)
        oi = nearest_index(alignment["odom_ts"], ts)
        depth_mm = np.array(alignment["depth_ds"][di]["depth"])
        depth_m, valid_fraction = masked_box_depth(depth_mm, row["x1"], row["y1"], row["x2"], row["y2"])
        if depth_m is None or valid_fraction < min_valid_fraction:
            n_skipped += 1
            continue
        wx, wy = backproject_to_world(row["x1"], row["x2"], depth_m,
                                      alignment["odom_x"][oi], alignment["odom_y"][oi],
                                      alignment["odom_yaw"][oi])
        world_x.append(wx)
        world_y.append(wy)
        kept_conf.append(float(row["confidence"]))
    return world_x, world_y, kept_conf, n_skipped


def compute_valid_depth_fractions(det: pd.DataFrame, alignment: dict) -> np.ndarray:
    """Fraction of each row's detection box with valid depth -- the input to
    auto_min_valid_fraction's threshold calibration. Should be run over a
    broad population (e.g. every class in detections.csv), not a single
    class's own rows -- see auto_min_valid_fraction. Caches decoded depth
    frames by nearest-frame index, since many detections in the same video
    frame share one depth image."""
    cache: dict[int, np.ndarray] = {}
    fractions = np.empty(len(det))
    for i, (_, row) in enumerate(det.iterrows()):
        di = nearest_index(alignment["depth_ts"], int(row["timestamp_ns"]))
        if di not in cache:
            cache[di] = np.array(alignment["depth_ds"][di]["depth"])
        depth_mm = cache[di]
        h, w = depth_mm.shape
        xi1, yi1 = max(int(row["x1"]), 0), max(int(row["y1"]), 0)
        xi2, yi2 = min(int(np.ceil(row["x2"])), w), min(int(np.ceil(row["y2"])), h)
        crop = depth_mm[yi1:yi2, xi1:xi2]
        valid = crop[(crop > DEPTH_INVALID_MM[0]) & (crop < DEPTH_INVALID_MM[1])]
        fractions[i] = valid.size / crop.size if crop.size else 0.0
    return fractions


def auto_min_valid_fraction(fractions: np.ndarray, n_bins: int = 256) -> float:
    """Otsu's method -- the standard image-thresholding algorithm for
    picking the split point that maximizes between-class variance in a
    histogram -- applied to a population of per-detection valid-depth
    fractions, in log1p space.

    Why log1p: the failure mode this guards against (the depth sensor
    returning almost nothing usable inside a box -- see
    object_localization.py's module docstring) clusters close to 0, while
    the bulk of well-behaved detections cluster close to 1. On a linear
    scale that near-zero failure spike is squeezed into a sliver next to
    the dominant high-fraction mode and Otsu's variance-maximizing split
    lands inside the good population instead (empirically ~0.65 on this
    dataset, rejecting 13% of ALL detections including many perfectly fine
    ones); log1p spreads the near-zero range out enough for Otsu to find
    the actual boundary (empirically ~0.13, rejecting the 3-4% that are
    genuinely mostly-invalid) -- BUT only if log1p is given enough dynamic
    range to spread: log1p is not scale-invariant, and applied directly to
    fractions already confined to [0, 1] it's close enough to linear
    (log1p(1)=0.69 vs. log1p(0)=0) that it barely spreads the near-zero
    spike at all, reproducing the bad linear-scale split (~0.46 empirically
    -- almost back to the too-aggressive 0.65 case above). Rescaling to
    percentage-like units (0-100) first gives log1p the two-orders-of-
    magnitude spread it actually needs.

    Calibrate this from a broad population (e.g. every class in
    detections.csv via compute_valid_depth_fractions), not a single class's
    own detections -- a rare or uniformly well-behaved class may not
    contain enough (or any) of the sensor's own failure cases for the split
    to mean anything (empirically, per-class Otsu on an 8-instance class
    rejected 87% of it)."""
    log_vals = np.log1p(fractions * 100.0)
    hist, edges = np.histogram(log_vals, bins=n_bins, range=(log_vals.min(), log_vals.max()))
    hist = hist.astype(np.float64)
    centers = (edges[:-1] + edges[1:]) / 2
    total = hist.sum()
    total_sum = (hist * centers).sum()
    best_t, best_var = 0.0, -1.0
    w0, sum0 = 0.0, 0.0
    for i in range(n_bins):
        w0 += hist[i]
        if w0 == 0:
            continue
        w1 = total - w0
        if w1 == 0:
            break
        sum0 += hist[i] * centers[i]
        mu0, mu1 = sum0 / w0, (total_sum - sum0) / w1
        var = w0 * w1 * (mu0 - mu1) ** 2
        if var > best_var:
            best_var, best_t = var, centers[i]
    return float(np.expm1(best_t)) / 100.0
