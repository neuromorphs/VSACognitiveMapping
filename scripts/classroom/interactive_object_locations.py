"""Interactively pick a detected object class and see every instance's
estimated global location, live -- click a different class in the list and
the plot redraws in place.

    python scripts/classroom/interactive_object_locations.py

See object_localization.py for the backprojection math (depth + odometry ->
world (x, y)) and its accepted simplification (camera pose approximated as
the robot's own odometry pose) -- this script only adds the interactive
class picker on top of it. The depth/odometry alignment data is loaded once
at startup and reused for every class you click, so switching classes is
near-instant; only the initial load takes a few seconds.
"""

import argparse

import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons

from object_localization import (
    REPO, auto_min_valid_fraction, compute_valid_depth_fractions, load_alignment_data, localize_detections,
)

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
RED = "#e34948"


def redraw(ax_plot, fig, class_name: str, det: pd.DataFrame, alignment: dict, cbar_holder: list,
          min_valid_fraction: float = 0.0) -> tuple[int, int]:
    """(Re)draw the scatter plot for class_name onto ax_plot. cbar_holder is
    a one-item list holding the previous colorbar (if any) so it can be
    removed first -- a colorbar is attached to the figure, not its axes, so
    ax_plot.clear() alone won't get rid of the old one.

    Returns (n_localized, n_skipped), so callers/tests can check the result
    without re-deriving it from the plot.
    """
    sub = det[det["class_name"] == class_name].reset_index(drop=True)
    world_x, world_y, kept_conf, n_skipped = localize_detections(sub, alignment, min_valid_fraction)

    if cbar_holder[0] is not None:
        # Must remove the old colorbar before clearing ax_plot: Colorbar.remove()
        # looks up its mappable's .axes to restore the plot axes' subplotspec,
        # and ax_plot.clear() would already have detached that old scatter
        # artist from its axes (making .axes None) if done first.
        cbar_holder[0].remove()
        cbar_holder[0] = None
    ax_plot.clear()
    ax_plot.plot(alignment["odom_x"], alignment["odom_y"], color=INK_MUTED, linewidth=1, alpha=0.5,
                zorder=1, label="robot trajectory")
    if world_x:
        # Auto-ranged (not a hardcoded [0, 1]) -- see plot_object_locations.py's
        # identical reasoning: a fixed [0, 1] scale washes out the tight-high-band
        # confidences that survive the detector's own --conf threshold.
        sc = ax_plot.scatter(world_x, world_y, c=kept_conf, cmap="viridis", s=40,
                             edgecolors=RED, linewidths=0.6, zorder=2, label=f"{class_name} detections")
        cbar_holder[0] = fig.colorbar(sc, ax=ax_plot, fraction=0.046, pad=0.04)
        cbar_holder[0].set_label("detection confidence", color=INK_SECONDARY, fontsize=8)
        cbar_holder[0].ax.tick_params(colors=INK_MUTED, labelsize=7)

    ax_plot.set_facecolor(SURFACE)
    ax_plot.set_aspect("equal")
    title = f"estimated global locations: '{class_name}' ({len(world_x)} detections"
    title += f", {n_skipped} skipped)" if n_skipped else ")"
    ax_plot.set_title(title, color=INK_PRIMARY, fontsize=12, pad=10)
    ax_plot.set_xlabel("x (m)", color=INK_MUTED, fontsize=9)
    ax_plot.set_ylabel("y (m)", color=INK_MUTED, fontsize=9)
    ax_plot.tick_params(colors=INK_MUTED, labelsize=8)
    for spine in ax_plot.spines.values():
        spine.set_color(GRID)
    legend = ax_plot.legend(loc="best", fontsize=8, facecolor=SURFACE, edgecolor=GRID)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)
    fig.canvas.draw_idle()
    return len(world_x), n_skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
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
                             "with valid depth (see object_localization.py's auto_min_valid_fraction). "
                             "default: auto-calibrated once at startup via Otsu's method over every "
                             "class, then reused for whichever class you click on")
    args = parser.parse_args()

    all_det = pd.read_csv(args.detections)
    det = all_det[all_det["confidence"] >= args.min_confidence]
    counts = det["class_name"].value_counts()  # most-common class first
    if counts.empty:
        raise SystemExit(f"no detections found in {args.detections} (>= confidence {args.min_confidence})")
    classes = counts.index.tolist()
    print(f"loaded {len(det)} detections across {len(classes)} classes")

    alignment = load_alignment_data(args.repo, args.depth_config, args.odom_config)

    if args.min_valid_fraction is None:
        fractions = compute_valid_depth_fractions(all_det, alignment)
        min_valid_fraction = auto_min_valid_fraction(fractions)
        print(f"auto-calibrated --min-valid-fraction: {min_valid_fraction:.3f} (Otsu's method over all "
             f"{len(all_det)} detections; pass --min-valid-fraction to override)")
    else:
        min_valid_fraction = args.min_valid_fraction

    fig = plt.figure(figsize=(11, 8), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 4, width_ratios=(1, 3, 3, 3), wspace=0.05)
    ax_list = fig.add_subplot(gs[0, 0])
    ax_plot = fig.add_subplot(gs[0, 1:])

    ax_list.set_facecolor(SURFACE)
    ax_list.set_title("class", color=INK_PRIMARY, fontsize=10, pad=6)
    for spine in ax_list.spines.values():
        spine.set_color(GRID)

    labels = [f"{c} ({counts[c]})" for c in classes]
    label_to_class = dict(zip(labels, classes))
    radio = RadioButtons(ax_list, labels, active=0,
                         label_props={"fontsize": [7], "color": [INK_SECONDARY]},
                         radio_props={"s": 32})

    cbar_holder = [None]

    def on_select(label: str) -> None:
        redraw(ax_plot, fig, label_to_class[label], det, alignment, cbar_holder, min_valid_fraction)

    radio.on_clicked(on_select)
    redraw(ax_plot, fig, classes[0], det, alignment, cbar_holder, min_valid_fraction)
    plt.show()


if __name__ == "__main__":
    main()
