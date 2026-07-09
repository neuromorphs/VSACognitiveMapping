"""Derive two "uncertainty" signals from embedding similarity and flag each
one's two biggest spikes. Computed both on the raw YOLO embeddings and, as
"phasor uncertainty", on the same embeddings after random_project_to_phasor
projects them into VSA/HD space.

    python scripts/classroom/plot_embedding_uncertainty.py

Reads embeddings.pt (see scripts/classroom/detect_and_embed_classroom.py) and writes
two files to outputs/classroom_detections/:
- embedding_uncertainty.png: the raw embedding self-correlation matrix
  (cosine) in the center, with two different marginal line plots -- per-frame
  uncertainty below it (x = frame index, shared with the matrix) and global
  frame uncertainty to its right (y = frame index, shared with the matrix).
- embedding_uncertainty_phasor.png: the same layout and signal definitions,
  but computed on phasor_correlation_matrix over random_project_to_phasor's
  output instead of cosine_self_correlation over the raw embeddings -- how
  much of the raw uncertainty signals survive the VSA projection.

Two signals, deliberately simple and interpretable rather than a trained
model, answering two different questions:

    per_frame_uncertainty[i] = 1 - similarity(embedding[i-1], embedding[i])
    global_uncertainty[i]    = 1 - mean_j(similarity(embedding[i], embedding[j]))

Per-frame uncertainty is exactly the matrix's immediate super-/sub-diagonal
band (how big was the last step). Global uncertainty is each frame's row
mean across the whole matrix (how well this frame fits in with the rest of
the video overall). The two can diverge: a segment can drift smoothly frame
to frame (low per-frame uncertainty throughout) while still ending up in a
part of embedding space unlike most of the rest of the video, which shows up
as a persistent deep-blue row/column band in the matrix and a sustained
elevated stretch in global_uncertainty, not a per-frame spike. Each signal's
own two largest, well-separated peaks (--min-peak-distance apart) are
highlighted on its marginal plot and as crosshairs on the matrix -- dashed
for per-frame, dotted for global.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from vsa_cognitive_mapping.vsa import cosine_self_correlation, phasor_correlation_matrix, random_project_to_phasor

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BLUE = "#2a78d6"
RED = "#e34948"


def adjacent_frame_uncertainty(corr: np.ndarray) -> np.ndarray:
    """(N, N) self-correlation matrix -> (N,) 1 - similarity to the previous
    frame, i.e. the matrix's immediate sub-diagonal. uncertainty[0] = 0 (no
    previous frame to compare against)."""
    n = corr.shape[0]
    uncertainty = np.zeros(n)
    uncertainty[1:] = 1.0 - corr[np.arange(1, n), np.arange(0, n - 1)]
    return uncertainty


def global_frame_uncertainty(corr: np.ndarray) -> np.ndarray:
    """(N, N) self-correlation matrix -> (N,) 1 - mean similarity to every
    other frame (row mean, excluding the self-similarity of 1.0 on the
    diagonal) -- how atypical this frame is relative to the whole sequence,
    not just its immediate neighbor. A frame (or a smoothly-drifting run of
    frames) that's dissimilar to most of the video shows up here even when
    adjacent_frame_uncertainty stays low the whole way through it."""
    n = corr.shape[0]
    row_sum = corr.sum(axis=1) - np.diag(corr)
    return 1.0 - row_sum / (n - 1)


def top_k_peaks(signal: np.ndarray, k: int, min_distance: int) -> list[int]:
    """Indices of the k largest values in signal, greedily picked highest
    first and skipping any candidate within min_distance frames of an
    already-picked peak -- so one drastic transition (which spans a few
    frames while the embedding settles) isn't counted as several peaks."""
    order = np.argsort(signal)[::-1]
    picked: list[int] = []
    for idx in order:
        if all(abs(idx - p) >= min_distance for p in picked):
            picked.append(int(idx))
        if len(picked) == k:
            break
    return sorted(picked)


def _style_marginal(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=INK_MUTED, labelsize=7)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_uncertainty(corr: np.ndarray, local_uncertainty: np.ndarray, local_peaks: list[int],
                     global_uncertainty: np.ndarray, global_peaks: list[int], out_path: Path,
                     matrix_title: str, matrix_label: str, suptitle: str) -> Path:
    n = corr.shape[0]
    frame_idx = np.arange(n)

    fig = plt.figure(figsize=(12, 11), facecolor=SURFACE, constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(4, 1))
    ax_main = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1], sharey=ax_main)
    ax_bottom = fig.add_subplot(gs[1, 0], sharex=ax_main)
    ax_corner = fig.add_subplot(gs[1, 1])
    ax_corner.axis("off")

    vmin, vmax = float(corr.min()), float(corr.max())
    im = ax_main.imshow(corr, cmap="viridis", vmin=vmin, vmax=vmax)
    ax_main.set_facecolor(SURFACE)
    ax_main.set_title(matrix_title, color=INK_PRIMARY, fontsize=12, pad=10)
    ax_main.set_ylabel("frame index", color=INK_MUTED, fontsize=9, labelpad=8)
    ax_main.tick_params(colors=INK_MUTED, labelsize=7)
    ax_main.tick_params(axis="x", labelbottom=False)
    for spine in ax_main.spines.values():
        spine.set_color(GRID)
    for peak in local_peaks:
        ax_main.axvline(peak, color=RED, linewidth=1, linestyle="--", alpha=0.8, zorder=5)
        ax_main.axhline(peak, color=RED, linewidth=1, linestyle="--", alpha=0.8, zorder=5)
    for peak in global_peaks:
        ax_main.axvline(peak, color=RED, linewidth=1, linestyle=":", alpha=0.8, zorder=5)
        ax_main.axhline(peak, color=RED, linewidth=1, linestyle=":", alpha=0.8, zorder=5)
    legend_handles = [
        plt.Line2D([], [], color=RED, linewidth=1, linestyle="--", label="per-frame transition"),
        plt.Line2D([], [], color=RED, linewidth=1, linestyle=":", label="globally atypical"),
    ]
    legend = ax_main.legend(handles=legend_handles, loc="lower right", fontsize=7,
                            facecolor=SURFACE, edgecolor=GRID)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    _style_marginal(ax_bottom)
    for peak in local_peaks:
        ax_bottom.axvspan(peak - 0.5, peak + 0.5, color=RED, alpha=0.15, zorder=1)
    ax_bottom.plot(frame_idx, local_uncertainty, color=BLUE, linewidth=1.2, zorder=3)
    ax_bottom.scatter(local_peaks, local_uncertainty[local_peaks], color=RED, s=30, zorder=4)
    ax_bottom.set_xlabel("frame index", color=INK_MUTED, fontsize=9, labelpad=6)
    ax_bottom.set_ylabel("per-frame uncertainty", color=INK_MUTED, fontsize=9, labelpad=6)

    _style_marginal(ax_right)
    for peak in global_peaks:
        ax_right.axhspan(peak - 0.5, peak + 0.5, color=RED, alpha=0.15, zorder=1)
    # Axes swapped so "frame index" lines up with the main heatmap's shared
    # y-axis (imshow's row 0 is at the top) -- a different signal than the
    # bottom plot's (global, not per-frame), not just a rotated duplicate.
    ax_right.plot(global_uncertainty, frame_idx, color=BLUE, linewidth=1.2, zorder=3)
    ax_right.scatter(global_uncertainty[global_peaks], global_peaks, color=RED, s=30, zorder=4)
    ax_right.invert_yaxis()
    ax_right.set_xlabel("global uncertainty", color=INK_MUTED, fontsize=9, labelpad=6)
    ax_right.tick_params(axis="y", labelleft=False)

    cbar = fig.colorbar(im, ax=ax_corner, fraction=0.9, pad=0.0, aspect=12)
    cbar.set_label(matrix_label, color=INK_SECONDARY, fontsize=8)
    cbar.ax.tick_params(colors=INK_MUTED, labelsize=7)

    fig.suptitle(suptitle, color=INK_PRIMARY, fontsize=13)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out_path


def analyze_and_plot(corr: np.ndarray, frame_idx: np.ndarray, out_path: Path, tag: str,
                     matrix_title: str, matrix_label: str, suptitle: str,
                     top_k: int, min_peak_distance: int) -> None:
    local = adjacent_frame_uncertainty(corr)
    local_peaks = top_k_peaks(local, top_k, min_peak_distance)
    for peak in local_peaks:
        print(f"{tag}[per-frame] transition at frame {frame_idx[peak]} (index {peak}): "
              f"uncertainty={local[peak]:.3f}")

    glob = global_frame_uncertainty(corr)
    global_peaks = top_k_peaks(glob, top_k, min_peak_distance)
    for peak in global_peaks:
        print(f"{tag}[global] atypical frame {frame_idx[peak]} (index {peak}): "
              f"uncertainty={glob[peak]:.3f}")

    plot_uncertainty(corr, local, local_peaks, glob, global_peaks, out_path,
                     matrix_title=matrix_title, matrix_label=matrix_label, suptitle=suptitle)
    print(f"{tag}uncertainty plot written to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embeddings", default="outputs/classroom_detections/embeddings.pt")
    parser.add_argument("--out-path", default="outputs/classroom_detections/embedding_uncertainty.png")
    parser.add_argument("--phasor-out-path",
                        default="outputs/classroom_detections/embedding_uncertainty_phasor.png")
    parser.add_argument("--hd-dim", type=int, default=256, help="phasor projection dimensionality")
    parser.add_argument("--phasor-seed", type=int, default=0, help="seed for the random projection matrix")
    parser.add_argument("--top-k", type=int, default=2, help="number of major transitions to flag per signal")
    parser.add_argument("--min-peak-distance", type=int, default=30,
                        help="minimum frame gap between flagged transitions, so one drastic "
                             "change spanning a few frames isn't counted twice")
    args = parser.parse_args()

    data = torch.load(args.embeddings)
    embeddings_t = data["embedding"]
    embeddings = embeddings_t.numpy()
    n, dim = embeddings.shape
    print(f"loaded {n} embeddings of dim {dim} from {args.embeddings}")

    frame_idx = data["frame_idx"].numpy()

    corr = cosine_self_correlation(embeddings)
    analyze_and_plot(corr, frame_idx, Path(args.out_path), "",
                     matrix_title="D455 frame embedding self-correlation (cosine)",
                     matrix_label="cosine similarity",
                     suptitle="Embedding uncertainty -- per-frame (bottom) vs. global (right)",
                     top_k=args.top_k, min_peak_distance=args.min_peak_distance)

    phasor_z, _W = random_project_to_phasor(embeddings_t, d=args.hd_dim, seed=args.phasor_seed)
    phasor_corr = phasor_correlation_matrix(phasor_z.numpy())
    analyze_and_plot(phasor_corr, frame_idx, Path(args.phasor_out_path), "[phasor] ",
                     matrix_title="D455 frame embedding self-correlation (phasor-projected, HD)",
                     matrix_label="phasor similarity",
                     suptitle="Phasor uncertainty -- per-frame (bottom) vs. global (right) "
                              f"(hd-dim={args.hd_dim}, seed={args.phasor_seed})",
                     top_k=args.top_k, min_peak_distance=args.min_peak_distance)


if __name__ == "__main__":
    main()
