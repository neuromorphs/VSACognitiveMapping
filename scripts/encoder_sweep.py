"""Compare VSA content encoders for JEPA latents (z_t): how well does each
one preserve the raw latent's similarity structure, and how mutually
orthogonal are the encoded vectors?

Requires an embeddings file produced by `scripts/eval.py --save-embeddings`.

    python scripts/encoder_sweep.py --embeddings checkpoints/jepa_sim/embeddings.pt --split train

Three encoders, all mapping z_t (N, d_in real) -> content (N, hd_dim complex):
  - random-proj: one random projection matrix, d_in -> 2*hd_dim -> hd_dim
    complex (the encoder scripts/associative_memory.py's phase 2 uses).
  - pca-fpe: reduce z_t to its top-K principal components (standardized to
    unit variance), FPE-encode each with its own base phasor, then bundle
    (elementwise mean) across components. K is fixed (default 4, `--n-components`).
  - full-fpe: same FPE-and-bundle scheme, but over every raw z_t dimension
    (no PCA reduction).

`--length-scale` (default 1.0) and `--n-components` (default 4) are both
fixed rather than swept: on this dataset, sweeping length_scale just trades
fidelity for orthogonality along one curve in both directions (large
length_scale shrinks the FPE exponent toward 0, collapsing every frame to
the same encoded vector; small length_scale aliases nearby scores into
near-random phase), and sweeping K showed a non-monotonic fidelity peak
around K=4-8 with orthogonality falling off monotonically past it — so K=4
is the reasonable default (also the sweet spot the exploratory notebook
found for its own top-K PCA + FPE approach). Neither knob, nor switching
bundle for bind, closes the gap with random-proj's fidelity/orthogonality
combination on this data.

Two metrics per hyperparameter setting, both derived from comparing the
encoded content's pairwise similarity matrix against the raw z_t cosine
similarity matrix (and against itself):
  - fidelity: correlation between the two similarity matrices — does the
    encoding preserve which frames look alike?
  - orthogonality: 1 - mean |off-diagonal content similarity| — are the
    encoded vectors close to mutually orthogonal (low interference when
    bundled into a shared associative-memory trace), independent of whether
    that matches the raw latent structure?

Writes PNGs under separate directories per encoder:
    <out-dir>/random-proj/, <out-dir>/pca-fpe/, <out-dir>/full-fpe/
"""

import argparse
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch

from vsa_cognitive_mapping.vsa import (
    cosine_self_correlation,
    fidelity_score,
    fpe_bundle_encode,
    make_axis_bases,
    orthogonality_score,
    pca_components,
    phasor_correlation_matrix,
    random_project_to_phasor,
)

# Fidelity is a correlation coefficient (signed, [-1, 1]) -> diverging.
DIVERGING_CMAP = mcolors.LinearSegmentedColormap.from_list("blue_gray_red", ["#2a78d6", "#f0efec", "#e34948"])
PRIMARY_COLOR = "#2a78d6"     # explained variance line
GRID_COLOR = "#e1e0d9"
MUTED_COLOR = "#898781"


def _style_axis(ax) -> None:
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID_COLOR, linewidth=0.6)
    ax.tick_params(colors=MUTED_COLOR)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_correlation_comparison(raw_corr: np.ndarray, vsa_corr: np.ndarray, fidelity: float,
                                orthogonality: float, title: str, out_path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    for ax, corr, subtitle in ((axes[0], raw_corr, "raw z_t (cosine)"), (axes[1], vsa_corr, "VSA content (phasor)")):
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap=DIVERGING_CMAP)
        ax.set_xlabel("frame")
        ax.set_ylabel("frame")
        ax.set_title(subtitle)
        fig.colorbar(im, ax=ax, label="similarity")
    fig.suptitle(f"{title}\nfidelity={fidelity:.3f}  orthogonality={orthogonality:.3f}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_explained_variance(explained_variance_ratio: np.ndarray, max_k: int, out_path: Path) -> Path:
    cumulative = np.cumsum(explained_variance_ratio[:max_k])
    fig, ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
    ax.plot(np.arange(1, max_k + 1), cumulative, color=PRIMARY_COLOR, marker="o", markersize=4)
    ax.set_xlabel("number of components")
    ax.set_ylabel("cumulative explained variance")
    ax.set_ylim(0, 1.02)
    ax.set_title("PCA explained variance — z_t")
    _style_axis(ax)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path



# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------

def run_random_proj(z_t: torch.Tensor, hd_dim: int, seed: int) -> dict:
    content = random_project_to_phasor(z_t, d=hd_dim, seed=seed)[0].numpy()
    raw_corr = cosine_self_correlation(z_t.numpy())
    vsa_corr = phasor_correlation_matrix(content)
    return {"raw_corr": raw_corr, "vsa_corr": vsa_corr,
            "fidelity": fidelity_score(raw_corr, vsa_corr), "orthogonality": orthogonality_score(vsa_corr)}


def run_pca_fpe(z: np.ndarray, hd_dim: int, n_components: int, length_scale: float, seed: int,
                variance_plot_max_k: int) -> dict:
    raw_corr = cosine_self_correlation(z)
    _, explained_variance_ratio = pca_components(z, variance_plot_max_k)

    scores, _ = pca_components(z, n_components)
    bases = make_axis_bases(n_components, hd_dim, seed)
    content = fpe_bundle_encode(scores, bases, length_scale)
    vsa_corr = phasor_correlation_matrix(content)
    return {"raw_corr": raw_corr, "vsa_corr": vsa_corr,
            "fidelity": fidelity_score(raw_corr, vsa_corr), "orthogonality": orthogonality_score(vsa_corr),
            "explained_variance_ratio": explained_variance_ratio, "variance_plot_max_k": variance_plot_max_k}


def run_full_fpe(z: np.ndarray, hd_dim: int, length_scale: float, seed: int) -> dict:
    raw_corr = cosine_self_correlation(z)
    z_std = (z - z.mean(axis=0, keepdims=True)) / (z.std(axis=0, keepdims=True) + 1e-8)
    bases = make_axis_bases(z.shape[1], hd_dim, seed)
    content = fpe_bundle_encode(z_std, bases, length_scale)
    vsa_corr = phasor_correlation_matrix(content)
    return {"raw_corr": raw_corr, "vsa_corr": vsa_corr,
            "fidelity": fidelity_score(raw_corr, vsa_corr), "orthogonality": orthogonality_score(vsa_corr)}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(embeddings_path: str, split: str, out_dir: str | Path, hd_dim: int, n_components: int,
       length_scale: float, seed: int) -> None:
    data = torch.load(embeddings_path)[split]
    z_t = data["z_t"]
    z_np = z_t.numpy()
    out_dir = Path(out_dir)

    max_valid_k = min(z_np.shape[0] - 1, z_np.shape[1])
    if n_components > max_valid_k:
        print(f"note: clipped n-components {n_components} -> {max_valid_k} "
             f"(min(N-1, d_in) for split {split!r})")
        n_components = max_valid_k

    # --- random-proj ---
    rp_dir = out_dir / "random-proj"
    rp = run_random_proj(z_t, hd_dim, seed)
    plot_correlation_comparison(rp["raw_corr"], rp["vsa_corr"], rp["fidelity"], rp["orthogonality"],
                                f"Random projection ({split})", rp_dir / f"correlation_{split}.png")
    print(f"[random-proj] fidelity={rp['fidelity']:.3f} orthogonality={rp['orthogonality']:.3f}")

    # --- pca-fpe ---
    pf_dir = out_dir / "pca-fpe"
    variance_plot_max_k = min(32, max_valid_k)
    pf = run_pca_fpe(z_np, hd_dim, n_components, length_scale, seed, variance_plot_max_k)
    plot_explained_variance(pf["explained_variance_ratio"], pf["variance_plot_max_k"],
                            pf_dir / f"explained_variance_{split}.png")
    plot_correlation_comparison(pf["raw_corr"], pf["vsa_corr"], pf["fidelity"], pf["orthogonality"],
                                f"PCA+FPE (K={n_components}, length_scale={length_scale}, {split})",
                                pf_dir / f"correlation_{split}.png")
    print(f"[pca-fpe] fidelity={pf['fidelity']:.3f} orthogonality={pf['orthogonality']:.3f} "
         f"(K={n_components}, length_scale={length_scale})")

    # --- full-fpe ---
    ff_dir = out_dir / "full-fpe"
    ff = run_full_fpe(z_np, hd_dim, length_scale, seed)
    plot_correlation_comparison(ff["raw_corr"], ff["vsa_corr"], ff["fidelity"], ff["orthogonality"],
                                f"Full FPE (length_scale={length_scale}, {split})",
                                ff_dir / f"correlation_{split}.png")
    print(f"[full-fpe] fidelity={ff['fidelity']:.3f} orthogonality={ff['orthogonality']:.3f} "
         f"(length_scale={length_scale})")

    print(f"plots written under {out_dir}/{{random-proj,pca-fpe,full-fpe}}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--out-dir", default=None, help="default: <embeddings dir>/plots")
    parser.add_argument("--hd-dim", type=int, default=256)
    parser.add_argument("--n-components", type=int, default=4,
                        help="fixed number of PCA components for pca-fpe")
    parser.add_argument("--length-scale", type=float, default=1.0,
                        help="fixed FPE length scale, used by both pca-fpe and full-fpe")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.embeddings).parent / "plots"
    run(args.embeddings, args.split, out_dir, args.hd_dim, args.n_components, args.length_scale, args.seed)
