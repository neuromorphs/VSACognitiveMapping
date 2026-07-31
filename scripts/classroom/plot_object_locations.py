"""Backproject one object class's detections to world (x, y) using depth +
odometry, and plot every instance's estimated global location.

    python scripts/classroom/plot_object_locations.py --class-name scissors

See object_localization.py for the backprojection math and its accepted
simplification (camera pose approximated as the robot's own odometry pose).
For picking the class interactively instead of via --class-name, see
interactive_object_locations.py.
"""

import argparse
from pathlib import Path

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from object_localization import (
    REPO, auto_min_valid_fraction, compute_valid_depth_fractions, load_alignment_data, localize_detections,
)

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
RED = "#e34948"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--class-name", required=True, help="COCO class to localize, e.g. 'scissors'")
    parser.add_argument("--repo", default=REPO)
    parser.add_argument("--depth-config", default="depth_d455")
    parser.add_argument("--odom-config", default="odometry_lio_sam")
    parser.add_argument("--detections", default="outputs/classroom_detections/detections.csv")
    parser.add_argument("--min-confidence", type=float, default=0.5,
                        help="skip detections below this YOLO confidence score (0-1). Unlike "
                             "--min-valid-fraction, confidence has no natural bimodal split to "
                             "auto-calibrate against here (detections.csv is already pre-filtered "
                             "by the detector's own --conf threshold, leaving a continuous, roughly "
                             "unimodal distribution) -- 0.5 is a conventional fixed cutoff, not a "
                             "data-derived one")
    parser.add_argument("--min-valid-fraction", type=float, default=None,
                        help="skip detections whose box has less than this fraction (0-1) of pixels "
                             "with valid depth -- guards against boxes where the depth sensor mostly "
                             "failed (e.g. thin/reflective objects like chair legs), letting a few "
                             "stray background pixels dominate the median depth and produce an "
                             "implausible location. default: auto-calibrated via Otsu's method over "
                             "every class in --detections (not just --class-name -- see "
                             "object_localization.py's auto_min_valid_fraction for why)")
    parser.add_argument("--out-path", default=None,
                        help="default: outputs/classroom_detections/object_locations_<class-name>.png")
    args = parser.parse_args()

    all_det = pd.read_csv(args.detections)
    det = all_det[(all_det["class_name"].str.lower() == args.class_name.lower()) &
                  (all_det["confidence"] >= args.min_confidence)].reset_index(drop=True)
    if det.empty:
        available = ", ".join(sorted(all_det["class_name"].unique()))
        raise SystemExit(f"no '{args.class_name}' detections found in {args.detections} "
                         f"(>= confidence {args.min_confidence})\navailable classes: {available}")
    print(f"{len(det)} '{args.class_name}' detections found")

    alignment = load_alignment_data(args.repo, args.depth_config, args.odom_config)

    if args.min_valid_fraction is None:
        fractions = compute_valid_depth_fractions(all_det, alignment)
        min_valid_fraction = auto_min_valid_fraction(fractions)
        print(f"auto-calibrated --min-valid-fraction: {min_valid_fraction:.3f} (Otsu's method over all "
             f"{len(all_det)} detections in {args.detections}, {(fractions < min_valid_fraction).sum()} "
             f"would be excluded across all classes; pass --min-valid-fraction to override)")
    else:
        min_valid_fraction = args.min_valid_fraction

    world_x, world_y, kept_conf, n_skipped = localize_detections(det, alignment, min_valid_fraction)

    if n_skipped:
        print(f"skipped {n_skipped}/{len(det)} detections with insufficient valid depth in their box "
             f"(< {min_valid_fraction:.1%} of pixels valid)")
    print(f"{len(world_x)} '{args.class_name}' instances localized")

    out_path = Path(args.out_path) if args.out_path else \
        Path(f"outputs/classroom_detections/object_locations_{args.class_name}.png")

    fig, ax = plt.subplots(figsize=(8, 7), facecolor=SURFACE, constrained_layout=True)
    ax.plot(alignment["odom_x"], alignment["odom_y"], color=INK_MUTED, linewidth=1, alpha=0.5,
           zorder=1, label="robot trajectory")
    # Auto-ranged (not a hardcoded [0, 1]) -- detections.csv is already
    # filtered to some --conf threshold at detection time, so the surviving
    # confidences typically sit in a tight high band; a fixed [0, 1] scale
    # would wash that band out to a near-uniform color and hide structure
    # (same reasoning plot_embedding_correlation.py's phasor panels use).
    sc = ax.scatter(world_x, world_y, c=kept_conf, cmap="viridis",
                    s=40, edgecolors=RED, linewidths=0.6, zorder=2, label=f"{args.class_name} detections")
    ax.set_facecolor(SURFACE)
    ax.set_aspect("equal")
    ax.set_title(f"estimated global locations: '{args.class_name}' ({len(world_x)} detections)",
                color=INK_PRIMARY, fontsize=12, pad=10)
    ax.set_xlabel("x (m)", color=INK_MUTED, fontsize=9)
    ax.set_ylabel("y (m)", color=INK_MUTED, fontsize=9)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    legend = ax.legend(loc="best", fontsize=8, facecolor=SURFACE, edgecolor=GRID)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("detection confidence", color=INK_SECONDARY, fontsize=8)
    cbar.ax.tick_params(colors=INK_MUTED, labelsize=7)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"plot written to {out_path}")


if __name__ == "__main__":
    main()
