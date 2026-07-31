"""Plot the pairwise Pearson correlation between all D455 frame embeddings,
and how well a random phasor projection preserves that structure.

    python scripts/classroom/plot_embedding_correlation.py

Reads embeddings.pt (see scripts/classroom/detect_and_embed_classroom.py) and writes
two files to outputs/classroom_detections/:
- embedding_correlation.png: Pearson correlation heatmap over the raw
  embeddings. Frames are in timestamp order, so revisited locations along
  the loop show up as bright off-diagonal bands.
- embedding_phasor_fidelity.png: the raw embeddings' cosine self-correlation
  next to the cosine self-correlation of the same embeddings after
  random_project_to_phasor (src/vsa_cognitive_mapping/vsa.py) projects them
  into phasor/HD space -- fidelity_score/orthogonality_score quantify how
  well the projection preserves the raw neighbor structure.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.colors as mcolors  # noqa: E402

from vsa_cognitive_mapping.vsa import (
    cosine_self_correlation,
    fidelity_score,
    orthogonality_score,
    phasor_correlation_matrix,
    random_project_to_phasor,
)

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"

# Signed [-1, 1] correlation -> diverging, same validated pair
# scripts/associative_memory.py and scripts/encoder_sweep.py each define
# locally for their own correlation-comparison plots.
DIVERGING_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#2a78d6", "#f0efec", "#e34948"])


def with_suffix_for_model(path: Path, embedding_model: str) -> Path:
    """Insert a _dino suffix before the extension when embedding_model is
    'dino', so both backends' default outputs coexist under the same
    output directory without clobbering each other; 'yolo' leaves the path
    unchanged (backward compatible with every existing default filename)."""
    return path if embedding_model == "yolo" else path.with_stem(path.stem + "_dino")


def plot_phasor_fidelity(ref_corr: np.ndarray, phasor_corr: np.ndarray, fidelity: float,
                         orthogonality: float, title: str, out_path: Path) -> Path:
    # A shared scale (not a hardcoded [-1, 1]) keeps the two panels directly
    # comparable while staying legible: this dataset's raw cosine
    # similarities all sit in a tight high band (e.g. ~0.7-1.0, frames from
    # one room on one walk), so the full theoretical range would saturate
    # both panels into a near-uniform block and hide the actual structure.
    vmin = min(ref_corr.min(), phasor_corr.min())
    vmax = max(ref_corr.max(), phasor_corr.max())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), facecolor=SURFACE, constrained_layout=True)
    for ax, corr, subtitle in ((axes[0], ref_corr, "raw embedding (cosine)"),
                               (axes[1], phasor_corr, "phasor-projected (HD)")):
        im = ax.imshow(corr, cmap=DIVERGING_CMAP, vmin=vmin, vmax=vmax)
        ax.set_facecolor(SURFACE)
        ax.set_title(subtitle, color=INK_PRIMARY, fontsize=11, pad=8)
        ax.set_xlabel("frame index", color=INK_MUTED, fontsize=9, labelpad=6)
        ax.set_ylabel("frame index", color=INK_MUTED, fontsize=9, labelpad=8)
        ax.tick_params(colors=INK_MUTED, labelsize=7)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("similarity", color=INK_SECONDARY, fontsize=8)
        cbar.ax.tick_params(colors=INK_MUTED, labelsize=7)

    fig.suptitle(f"{title}\nfidelity={fidelity:.3f}  orthogonality={orthogonality:.3f}",
                color=INK_PRIMARY, fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embedding-model", choices=["yolo", "dino"], default="yolo",
                        help="embedding backend the input embeddings.pt was produced with -- "
                             "only changes the default --embeddings/--out-path/--phasor-out-path "
                             "(adds a _dino suffix); this script is fully embedding-dimension-"
                             "agnostic and does no backend-specific math")
    parser.add_argument("--embeddings", default=None,
                        help="default: outputs/classroom_detections/embeddings.pt, or "
                             "embeddings_dino.pt if --embedding-model dino")
    parser.add_argument("--out-path", default=None,
                        help="default: outputs/classroom_detections/embedding_correlation.png, "
                             "or ..._dino.png if --embedding-model dino")
    parser.add_argument("--hd-dim", type=int, default=256, help="phasor projection dimensionality")
    parser.add_argument("--phasor-seed", type=int, default=0, help="seed for the random projection matrix")
    parser.add_argument("--phasor-out-path", default=None,
                        help="default: outputs/classroom_detections/embedding_phasor_fidelity.png, "
                             "or ..._dino.png if --embedding-model dino")
    args = parser.parse_args()

    embeddings_path = Path(args.embeddings) if args.embeddings else with_suffix_for_model(
        Path("outputs/classroom_detections/embeddings.pt"), args.embedding_model)
    out_path = Path(args.out_path) if args.out_path else with_suffix_for_model(
        Path("outputs/classroom_detections/embedding_correlation.png"), args.embedding_model)
    phasor_out_path = Path(args.phasor_out_path) if args.phasor_out_path else with_suffix_for_model(
        Path("outputs/classroom_detections/embedding_phasor_fidelity.png"), args.embedding_model)

    data = torch.load(embeddings_path)
    embeddings_t = data["embedding"]
    embeddings = embeddings_t.numpy()
    n, dim = embeddings.shape
    print(f"loaded {n} embeddings of dim {dim} from {embeddings_path}")

    corr = np.corrcoef(embeddings)
    vmin, vmax = corr.min(), corr.max()
    print(f"correlation range: [{vmin:.3f}, {vmax:.3f}]")

    fig, ax = plt.subplots(figsize=(9, 8), facecolor=SURFACE, constrained_layout=True)
    im = ax.imshow(corr, cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_facecolor(SURFACE)
    ax.set_title("D455 frame embedding self-correlation", color=INK_PRIMARY, fontsize=13, pad=10)
    ax.set_xlabel("frame index", color=INK_MUTED, fontsize=10, labelpad=8)
    ax.set_ylabel("frame index", color=INK_MUTED, fontsize=10, labelpad=12)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson correlation", color=INK_SECONDARY, fontsize=9)
    cbar.ax.tick_params(colors=INK_MUTED, labelsize=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"correlation heatmap written to {out_path}")

    ref_corr = cosine_self_correlation(embeddings)
    phasor_z, _W = random_project_to_phasor(embeddings_t, d=args.hd_dim, seed=args.phasor_seed)
    phasor_corr = phasor_correlation_matrix(phasor_z.numpy())
    fidelity = fidelity_score(ref_corr, phasor_corr)
    orthogonality = orthogonality_score(phasor_corr)
    print(f"[phasor] fidelity={fidelity:.3f} orthogonality={orthogonality:.3f} "
          f"(hd-dim={args.hd_dim}, seed={args.phasor_seed})")

    plot_phasor_fidelity(ref_corr, phasor_corr, fidelity, orthogonality,
                         "D455 frame embedding: raw vs. phasor-projected", phasor_out_path)
    print(f"phasor fidelity heatmap written to {phasor_out_path}")


if __name__ == "__main__":
    main()
