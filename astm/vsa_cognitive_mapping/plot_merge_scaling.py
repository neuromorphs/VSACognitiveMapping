"""K-scaling summary of the lidar map-joining head-to-head: read every
``merge_comparison_lidar_k*.json`` that merge_comparison_lidar.py wrote to
outputs/lidar_merge/ and plot, per method, how joined-map quality and
per-robot payload scale with the number of robots K.

One figure, three panels:
  (a) median localization error of held-out scans vs K,
  (b) R@0.5 (fraction of held-out scans localized within 0.5 code units) vs K,
  (c) bytes transmitted per robot vs K (log-y; the VSA signature and the
      hit-count grid are K-independent, the raw scan database shrinks per
      robot because the fixed 800-scan walk is split K ways).

The hypothesis on trial: VSA signature addition stays flat as K grows
(crosstalk adds gracefully) while the raw scan-DB union degrades (each
robot's block covers less of the room).

    python -m vsa_cognitive_mapping.plot_merge_scaling
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=os.path.join(HERE, "..", "outputs",
                                                      "lidar_merge"))
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.out_dir,
                                          "merge_comparison_lidar_k*.json")))
    if not paths:
        raise SystemExit(f"no merge_comparison_lidar_k*.json in {args.out_dir}; "
                         "run merge_comparison_lidar.py first")

    # per-method series keyed by method name, sorted by K
    runs = []
    for p in paths:
        with open(p) as f:
            runs.append(json.load(f))
    runs.sort(key=lambda d: d["robots"])
    ks = [d["robots"] for d in runs]
    methods = [r["method"] for r in runs[0]["rows"]]
    series = {m: dict(med_err=[], r_at_05=[], bytes=[]) for m in methods}
    for d in runs:
        by_name = {r["method"]: r for r in d["rows"]}
        for m in methods:
            series[m]["med_err"].append(by_name[m]["med_err"])
            series[m]["r_at_05"].append(by_name[m]["r_at_05"])
            series[m]["bytes"].append(by_name[m]["bytes_per_robot"])

    styles = {m: (marker, color) for m, marker, color in zip(
        methods, "osD^v", ["tab:blue", "tab:orange", "tab:green",
                           "tab:red", "tab:purple"])}

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    panels = [("med_err", "median localization error [code units]",
               "(a) accuracy vs K"),
              ("r_at_05", "R@0.5 (fraction within 0.5u)",
               "(b) recall vs K"),
              ("bytes", "bytes transmitted per robot",
               "(c) per-robot payload vs K")]
    for ax, (key, ylab, title) in zip(axes, panels):
        for m in methods:
            marker, color = styles[m]
            ax.plot(ks, series[m][key], marker=marker, color=color,
                    label=m, lw=1.6, ms=6)
        ax.set_xlabel("K robots")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=10)
        ax.set_xscale("log", base=2)
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks])
        ax.grid(alpha=0.3)
        if key == "bytes":
            ax.set_yscale("log")
        if key == "r_at_05":
            ax.set_ylim(0, 1.05)
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Joining K single-session lidar maps: quality and payload vs K "
                 f"(fixed {runs[0]['steps']}-scan walk split K ways, "
                 f"D={runs[0]['dim']}, {runs[0]['tests']} held-out tests)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    png = os.path.join(args.out_dir, "merge_scaling.png")
    fig.savefig(png, dpi=140)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
