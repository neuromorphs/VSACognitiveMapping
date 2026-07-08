"""Associative-memory pipeline over exported JEPA embeddings, run in phases.

Requires an embeddings file produced by `scripts/eval.py --save-embeddings`.

    python scripts/associative_memory.py phase1 --embeddings checkpoints/jepa_sim/embeddings.pt --split train
    python scripts/associative_memory.py phase2 --embeddings checkpoints/jepa_sim/embeddings.pt --split train
    python scripts/associative_memory.py phase3 --embeddings checkpoints/jepa_sim/embeddings.pt --root data/dataset_vjepa_bezier

Phase 1 — correlations: cosine-similarity matrices comparing the raw JEPA
latents against themselves, saved as PNGs:
  - z_t vs z_t:       encoder self-consistency across frames.
  - z_pred vs z_pred: predictor self-consistency (does it collapse frame-to-frame?).
  - z_pred vs z_tp1:  predicted vs. actual next latent — a strong diagonal
    means the predictor recovers the right frame, not just "a plausible one".

Phase 2 — HD memory: builds (but does not query) one bundled HD associative
memory trace, binding each frame's latent to its time and position:

    content = random_project_to_phasor(z_t)      # 128-dim real -> hd_dim complex
    trace_i = content ⊗ Bt**frame_t ⊗ Bx**x_t ⊗ By**y_t

then bundles (elementwise mean) all `trace_i` into one trace, saving it plus
the projection matrix and base phasors — everything needed to unbind/query
later. Requires position data (transitions.csv must have been present for
the dataset the embeddings came from).

Phase 2 stops at "build and save the trace." For how to query it back out
(unbind by a candidate time or position, then sweep similarity against
candidates), see the "Associative Memory 1/2" sections of
notebooks/loading_sample_blender_data.ipynb — the same pattern applies here.

Phase 2b — encoding fidelity: before phase 2's choices get baked into a
trace (and before that trace is combined with the JEPA content vectors),
checks each axis's HD encoding on its own terms:

    python scripts/associative_memory.py phase2b --embeddings checkpoints/jepa_sim/embeddings.pt --split train --root data/dataset_vjepa_bezier

For each of time (frame_t), position (x, y), heading (yaw), and action
(one-hot forward/stop/left/right), encodes the channel into HD space and
compares its pairwise phasor-similarity matrix against a domain-appropriate
ground-truth reference (fidelity_score), plus how mutually orthogonal the
encoded vectors are (orthogonality_score) — the same evaluation
`scripts/encoder_sweep.py` runs for z_t, applied instead to the axes phase 2
binds around it. Position and time reuse ordinary FPE (Bx/By/Bt), which is a
good fit for open, unbounded scalars. Heading is different: yaw is periodic,
and FPE with a random-phase base is *not* circular-safe (`base**angle`
doesn't return to the same phasor at `angle + 2*pi` unless the base's phase
happens to be an integer number of radians) — so heading uses
`Phasor(..., circular=True)`, and this phase also renders a wraparound-sweep
plot showing exactly where a naive continuous-phase base breaks down
relative to the circular-safe one and ground truth. Action is categorical
rather than a scalar, so instead of FPE it gets one fixed random Phasor per
label from a small codebook, scored against a ground-truth reference where
same-action pairs are 1 and different-action pairs are 0 (one-hot vectors
are already mutually orthogonal). If `--root` is given, also scores the
per-transition pose-change deltas (dx/dy/dz_world, dist_ground, dyaw) the
same way. Finally, prints a cross-channel leakage check between the axis
codes themselves (time vs. position vs. heading vs. action) — are the
independently-bound axes actually independent, or does one leak structure
into another before they're bound together.

Phase 3 — validate data loading: sanity-check plots for the pose, heading,
and delta data that phases 1/2 rely on, split by train/val, so a bad
frame_t/pos_t alignment or a lopsided split shows up before it's baked into
a memory trace. Pass --root to also validate the per-transition deltas
(dx/dy/dz_world, dist_ground, dyaw_rad/deg); without it, only pose is
checked.
"""

import argparse
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch

from vsa_cognitive_mapping.data import ACTIONS, load_deltas_by_transition
from vsa_cognitive_mapping.vsa import (
    Phasor,
    circular_similarity,
    circular_wraparound_sweep,
    cosine_self_correlation,
    fidelity_score,
    fpe_bundle_encode,
    make_axis_bases,
    neg_abs_diff,
    orthogonality_score,
    phasor_correlation_matrix,
    phasor_cross_correlation,
    random_project_to_phasor,
)

# Fixed categorical identity: train is always blue, val is always orange,
# across every plot in this file (see dataviz skill — color follows the
# entity, never its rank). Grid/muted values match the same reference palette.
TRAIN_COLOR = "#2a78d6"
VAL_COLOR = "#eb6834"
GRID_COLOR = "#e1e0d9"
MUTED_COLOR = "#898781"
# Fidelity is a correlation coefficient (signed, [-1, 1]) -> diverging, same
# palette scripts/encoder_sweep.py uses for its own correlation-comparison plots.
DIVERGING_CMAP = mcolors.LinearSegmentedColormap.from_list("blue_gray_red", ["#2a78d6", "#f0efec", "#e34948"])


def _style_axis(ax) -> None:
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID_COLOR, linewidth=0.6)
    ax.tick_params(colors=MUTED_COLOR)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


# ---------------------------------------------------------------------------
# Phase 1: correlation matrices
# ---------------------------------------------------------------------------

def cosine_correlation_matrix(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    """(N, D) x (M, D) real latents -> (N, M) pairwise cosine similarity."""
    a_unit = a / a.norm(dim=-1, keepdim=True)
    b_unit = b / b.norm(dim=-1, keepdim=True)
    return (a_unit @ b_unit.T).numpy()


def plot_correlation_matrix(corr: np.ndarray, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5.5), constrained_layout=True)
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="viridis")
    ax.set_xlabel("frame")
    ax.set_ylabel("frame")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="cosine similarity")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def phase1_correlations(embeddings_path: str, split: str, out_dir: str | Path) -> dict[str, Path]:
    data = torch.load(embeddings_path)[split]
    out_dir = Path(out_dir)

    matrices = {
        "z_t_vs_z_t": (data["z_t"], data["z_t"], "z_t vs z_t — encoder self-consistency"),
        "z_pred_vs_z_pred": (data["z_pred"], data["z_pred"], "z_pred vs z_pred — predictor self-consistency"),
        "z_pred_vs_z_tp1": (data["z_pred"], data["z_tp1"], "z_pred vs z_tp1 — predicted vs. actual next latent"),
    }

    paths = {}
    for name, (a, b, title) in matrices.items():
        corr = cosine_correlation_matrix(a, b)
        out_path = out_dir / f"{split}_{name}.png"
        plot_correlation_matrix(corr, f"{split} split — {title}", out_path)
        paths[name] = out_path
        print(f"wrote {out_path}")
    return paths


# ---------------------------------------------------------------------------
# Phase 2: HD associative memory trace
# ---------------------------------------------------------------------------

def build_memory(z_t: torch.Tensor, frame_t: torch.Tensor, pos_t: torch.Tensor, hd_dim: int,
                 content_seed: int, time_seed: int, x_seed: int, y_seed: int) -> dict:
    content, W = random_project_to_phasor(z_t, d=hd_dim, seed=content_seed)
    content = content.numpy()

    Bt = Phasor(dim=hd_dim, seed=time_seed)
    Bx = Phasor(dim=hd_dim, seed=x_seed)
    By = Phasor(dim=hd_dim, seed=y_seed)

    bound = []
    for i in range(content.shape[0]):
        frame_content = Phasor(data=content[i])
        time_code = Bt ** float(frame_t[i])
        pos_code = (Bx ** float(pos_t[i, 0])) * (By ** float(pos_t[i, 1]))
        bound.append((frame_content * time_code * pos_code).values)

    memory = np.stack(bound).mean(axis=0)
    # Store as native torch tensors (not numpy arrays) so the file round-trips
    # through torch.load's default weights_only=True.
    return {"memory": torch.from_numpy(memory), "W": W, "hd_dim": hd_dim,
            "bases": {"Bt": torch.from_numpy(Bt.values), "Bx": torch.from_numpy(Bx.values),
                     "By": torch.from_numpy(By.values)}}


def load_deltas_for_split(root: str | Path, frame_t: torch.Tensor, frame_tp1: torch.Tensor) -> torch.Tensor | None:
    """Per-transition (dx_world, dy_world, dz_world, dist_ground, dyaw_rad, dyaw_deg),
    aligned row-for-row to a split's frame_t/frame_tp1, or None if transitions.csv
    isn't present under `root`. Not yet bound into the memory trace — attached to
    the saved result so a later phase can decide how to fold it in."""
    deltas = load_deltas_by_transition(root)
    if deltas is None:
        return None
    return torch.tensor([deltas[(int(t), int(tp1))] for t, tp1 in zip(frame_t, frame_tp1)],
                        dtype=torch.float32)


def phase2_build_memory(embeddings_path: str, split: str, hd_dim: int, out_path: str | Path,
                        content_seed: int, time_seed: int, x_seed: int, y_seed: int,
                        root: str | Path | None = None) -> Path:
    data = torch.load(embeddings_path)[split]
    if data["pos_t"] is None:
        raise ValueError(f"{embeddings_path!r} split {split!r} has no position data — "
                         "re-run eval.py --save-embeddings on a dataset with transitions.csv")

    result = build_memory(data["z_t"], data["frame_t"], data["pos_t"], hd_dim,
                          content_seed, time_seed, x_seed, y_seed)
    result.update(split=split, n_frames=data["z_t"].shape[0], embeddings_path=str(embeddings_path))

    if root is not None:
        delta_t = load_deltas_for_split(root, data["frame_t"], data["frame_tp1"])
        if delta_t is None:
            print(f"note: no transitions.csv under {root} — saving without deltas")
        result["delta_t"] = delta_t

    out_path = Path(out_path)
    torch.save(result, out_path)
    print(f"memory trace over {result['n_frames']} frames ({split}) written to {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Phase 2b: encoding fidelity — time, position, heading, pose-change deltas
# ---------------------------------------------------------------------------

def encode_time(frame_t: np.ndarray, hd_dim: int, seed: int, length_scale: float) -> np.ndarray:
    base = Phasor(dim=hd_dim, seed=seed)
    return np.stack([(base ** float(t / length_scale)).values for t in frame_t])


def encode_position(xy: np.ndarray, hd_dim: int, x_seed: int, y_seed: int, length_scale: float) -> np.ndarray:
    Bx = Phasor(dim=hd_dim, seed=x_seed)
    By = Phasor(dim=hd_dim, seed=y_seed)
    return np.stack([
        ((Bx ** float(x / length_scale)) * (By ** float(y / length_scale))).values
        for x, y in xy
    ])


def encode_heading(yaw: np.ndarray, hd_dim: int, seed: int, max_freq: int) -> np.ndarray:
    """FPE with `circular=True`, not an ordinary Phasor base -- see the
    Phase 2b module docstring and `Phasor.__init__`'s `circular` branch for
    why a plain random-phase base isn't safe for a periodic scalar."""
    base = Phasor(dim=hd_dim, seed=seed, circular=True, max_freq=max_freq)
    return np.stack([(base ** float(a)).values for a in yaw])


def encode_action(action_onehot: np.ndarray, hd_dim: int, seed: int) -> np.ndarray:
    """Action (see `ACTIONS`) is categorical, not continuous like time/
    position/heading -- there's no scalar to raise a base to, so each label
    gets its own independent random Phasor from a small codebook (one entry
    per column of the one-hot), looked up by the active label per row."""
    codebook = make_axis_bases(action_onehot.shape[1], hd_dim, seed=seed)
    labels = action_onehot.argmax(axis=1)
    return np.stack([codebook[label].values for label in labels])


def plot_encoding_fidelity(ref_corr: np.ndarray, vsa_corr: np.ndarray, fidelity: float,
                           orthogonality: float, title: str, out_path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    for ax, corr, subtitle in ((axes[0], ref_corr, "ground truth"), (axes[1], vsa_corr, "VSA content (phasor)")):
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


def plot_yaw_wraparound_demo(hd_dim: int, seed: int, max_freq: int, out_path: Path) -> Path:
    """Visualizes exactly why yaw needs a circular-safe base: sweep a query
    angle across two full turns against a fixed reference of 0, comparing
    ground-truth cos(angle), an ordinary continuous-phase base (which drifts
    and does not repeat past +/-pi), and the circular-safe base actually
    used by `encode_heading`."""
    continuous_base = Phasor(dim=hd_dim, seed=seed)
    circular_base = Phasor(dim=hd_dim, seed=seed, circular=True, max_freq=max_freq)

    angles, ground_truth, continuous_encoded = circular_wraparound_sweep(continuous_base)
    _, _, circular_encoded = circular_wraparound_sweep(circular_base)

    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    ax.plot(angles, ground_truth, color=MUTED_COLOR, linewidth=2, linestyle="--", label="ground truth: cos(angle)")
    ax.plot(angles, continuous_encoded, color=VAL_COLOR, linewidth=1.5, label="continuous-phase base (naive FPE)")
    ax.plot(angles, circular_encoded, color=TRAIN_COLOR, linewidth=1.5, label=f"circular=True base (max_freq={max_freq})")
    for boundary in (-np.pi, np.pi):
        ax.axvline(boundary, color=GRID_COLOR, linewidth=1, zorder=0)
    ax.set_xlabel("query angle (radians), reference fixed at 0")
    ax.set_ylabel("similarity to reference")
    ax.set_title("Why yaw needs a circular-safe base — similarity vs. angle, two full turns")
    ax.legend(frameon=False, fontsize=8)
    _style_axis(ax)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def phase2b_encoding_fidelity(embeddings_path: str, split: str, hd_dim: int, out_dir: str | Path,
                              time_seed: int, x_seed: int, y_seed: int, yaw_seed: int, action_seed: int,
                              length_scale: float, yaw_max_freq: int,
                              root: str | Path | None = None) -> dict[str, Path]:
    data = torch.load(embeddings_path)[split]
    if data["pos_t"] is None:
        raise ValueError(f"{embeddings_path!r} split {split!r} has no position data — "
                         "re-run eval.py --save-embeddings on a dataset with transitions.csv")

    out_dir = Path(out_dir)
    frame_t = data["frame_t"].numpy().astype(np.float64)
    pos_t = data["pos_t"].numpy()
    xy, yaw = pos_t[:, :2], pos_t[:, 3]

    codes: dict[str, np.ndarray] = {}
    paths: dict[str, Path] = {}

    # --- time ---
    codes["time"] = encode_time(frame_t, hd_dim, time_seed, length_scale)
    ref = neg_abs_diff(frame_t)
    vsa_corr = phasor_correlation_matrix(codes["time"])
    fid, orth = fidelity_score(ref, vsa_corr), orthogonality_score(vsa_corr)
    paths["time"] = plot_encoding_fidelity(ref, vsa_corr, fid, orth, f"Time encoding ({split})",
                                           out_dir / f"validate_encoding_time_{split}.png")
    print(f"[time] fidelity={fid:.3f} orthogonality={orth:.3f}")

    # --- position ---
    codes["position"] = encode_position(xy, hd_dim, x_seed, y_seed, length_scale)
    ref = cosine_self_correlation(xy)
    vsa_corr = phasor_correlation_matrix(codes["position"])
    fid, orth = fidelity_score(ref, vsa_corr), orthogonality_score(vsa_corr)
    paths["position"] = plot_encoding_fidelity(ref, vsa_corr, fid, orth, f"Position encoding ({split})",
                                               out_dir / f"validate_encoding_position_{split}.png")
    print(f"[position] fidelity={fid:.3f} orthogonality={orth:.3f}")

    # --- heading (yaw) ---
    codes["heading"] = encode_heading(yaw, hd_dim, yaw_seed, yaw_max_freq)
    ref = circular_similarity(yaw)
    vsa_corr = phasor_correlation_matrix(codes["heading"])
    fid, orth = fidelity_score(ref, vsa_corr), orthogonality_score(vsa_corr)
    paths["heading"] = plot_encoding_fidelity(ref, vsa_corr, fid, orth, f"Heading (yaw) encoding ({split})",
                                              out_dir / f"validate_encoding_heading_{split}.png")
    print(f"[heading] fidelity={fid:.3f} orthogonality={orth:.3f}")

    paths["heading_wraparound"] = plot_yaw_wraparound_demo(
        hd_dim, yaw_seed, yaw_max_freq, out_dir / f"validate_encoding_heading_wraparound_{split}.png")

    # --- action ---
    action_onehot = data["action"].numpy()
    codes["action"] = encode_action(action_onehot, hd_dim, action_seed)
    ref = cosine_self_correlation(action_onehot.astype(np.float64))
    vsa_corr = phasor_correlation_matrix(codes["action"])
    fid, orth = fidelity_score(ref, vsa_corr), orthogonality_score(vsa_corr)
    paths["action"] = plot_encoding_fidelity(ref, vsa_corr, fid, orth, f"Action encoding ({split})",
                                             out_dir / f"validate_encoding_action_{split}.png")
    print(f"[action] fidelity={fid:.3f} orthogonality={orth:.3f}")

    # --- pose-change deltas (optional) ---
    if root is not None:
        delta_t = load_deltas_for_split(root, data["frame_t"], data["frame_tp1"])
        if delta_t is None:
            print(f"note: no transitions.csv under {root} — skipping delta encoding fidelity")
        else:
            delta_np = delta_t.numpy()
            delta_std = (delta_np - delta_np.mean(axis=0, keepdims=True)) / (delta_np.std(axis=0, keepdims=True) + 1e-8)
            bases = make_axis_bases(delta_np.shape[1], hd_dim, seed=x_seed)
            codes["deltas"] = fpe_bundle_encode(delta_std, bases, length_scale)
            ref = cosine_self_correlation(delta_np)
            vsa_corr = phasor_correlation_matrix(codes["deltas"])
            fid, orth = fidelity_score(ref, vsa_corr), orthogonality_score(vsa_corr)
            paths["deltas"] = plot_encoding_fidelity(ref, vsa_corr, fid, orth, f"Pose-change (delta) encoding ({split})",
                                                     out_dir / f"validate_encoding_deltas_{split}.png")
            print(f"[deltas] fidelity={fid:.3f} orthogonality={orth:.3f}")
    else:
        print("note: no --root given — skipping delta encoding fidelity")

    # --- cross-channel leakage: do the independently-bound axes interfere? ---
    print("cross-channel leakage (mean |cross-correlation| between axis codes; lower = more independent):")
    axis_names = [name for name in ("time", "position", "heading", "action") if name in codes]
    for i, name_a in enumerate(axis_names):
        for name_b in axis_names[i + 1:]:
            cross = phasor_cross_correlation(codes[name_a], codes[name_b])
            leakage = float(np.abs(cross).mean())
            print(f"  [{name_a} vs {name_b}] mean|cross-corr|={leakage:.3f}")

    for path in paths.values():
        print(f"wrote {path}")
    return paths


# ---------------------------------------------------------------------------
# Phase 3: validate data loading (pose, heading, deltas, train/val balance)
# ---------------------------------------------------------------------------

SPLITS = (("train", TRAIN_COLOR), ("val", VAL_COLOR))
POSE_CHANNELS = [("x", 0), ("y", 1), ("z", 2), ("yaw (deg)", 3)]
DELTA_CHANNELS = [("dx_world", 0), ("dy_world", 1), ("dz_world", 2), ("dist_ground", 3), ("dyaw_deg", 5)]


def check_pose_continuity(all_data: dict[str, dict]) -> None:
    """For each split, frame i+1's pos_t should equal frame i's pos_tp1 —
    they're the same physical frame, read via two different columns of
    transitions.csv. A mismatch means frame_t/pos_t got misaligned during
    loading, not just a numerically-noisy simulator. Skipped wherever
    frame_skip > 1 chains rows together, since then consecutive rows aren't
    adjacent frames by construction."""
    for split, _ in SPLITS:
        data = all_data[split]
        frame_t, frame_tp1 = data["frame_t"], data["frame_tp1"]
        contiguous = frame_t[1:] == frame_tp1[:-1]
        if not contiguous.all():
            print(f"[{split}] frame_skip > 1 detected — skipping pose continuity check")
            continue
        diff = (data["pos_t"][1:, :3] - data["pos_tp1"][:-1, :3]).abs().max().item()
        print(f"[{split}] pose continuity check (pos_tp1[i] vs pos_t[i+1], xyz): max abs diff = {diff:.6g}")


def plot_trajectory(pos_by_split: dict[str, torch.Tensor], out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
    for split, color in SPLITS:
        x, y, yaw = (pos_by_split[split][:, i].numpy() for i in (0, 1, 3))
        ax.plot(x, y, color=color, alpha=0.5, linewidth=1, zorder=1)
        ax.scatter(x, y, color=color, s=12, label=split, zorder=2)
        step = max(1, len(x) // 25)
        ax.quiver(x[::step], y[::step], np.cos(yaw[::step]), np.sin(yaw[::step]),
                  color=color, angles="xy", scale_units="xy", scale=1.2, width=0.006, zorder=3)
    ax.set_xlabel("x (world)")
    ax.set_ylabel("y (world)")
    ax.set_title("Ground-truth top-down trajectory — train vs val")
    ax.set_aspect("equal")
    ax.legend(frameon=False)
    _style_axis(ax)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_pose_over_frames(pos_by_split: dict[str, torch.Tensor], frame_by_split: dict[str, torch.Tensor],
                          out_path: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for ax, (name, idx) in zip(axes.flat, POSE_CHANNELS):
        for split, color in SPLITS:
            vals = pos_by_split[split][:, idx].numpy()
            if name.startswith("yaw"):
                vals = np.degrees(vals)
            ax.plot(frame_by_split[split].numpy(), vals, color=color, linewidth=1.2, label=split)
        ax.set_xlabel("frame")
        ax.set_ylabel(name)
        _style_axis(ax)
    axes.flat[0].legend(frameon=False)
    fig.suptitle("Ground-truth pose per frame — train vs val")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_pose_distributions(pos_by_split: dict[str, torch.Tensor], out_path: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for ax, (name, idx) in zip(axes.flat, POSE_CHANNELS):
        for split, color in SPLITS:
            vals = pos_by_split[split][:, idx].numpy()
            if name.startswith("yaw"):
                vals = np.degrees(vals)
            ax.hist(vals, bins=30, color=color, alpha=0.55, label=split, edgecolor="none")
        ax.set_xlabel(name)
        ax.set_ylabel("count")
        _style_axis(ax)
    axes.flat[0].legend(frameon=False)
    fig.suptitle("Pose distributions — train vs val")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_deltas_over_frames(delta_by_split: dict[str, torch.Tensor | None], frame_by_split: dict[str, torch.Tensor],
                            out_path: Path) -> Path | None:
    if all(delta_by_split[split] is None for split, _ in SPLITS):
        print("skipping delta-over-frame plot: no deltas available for either split (pass --root)")
        return None
    fig, axes = plt.subplots(3, 2, figsize=(11, 9), constrained_layout=True)
    for ax, (name, idx) in zip(axes.flat, DELTA_CHANNELS):
        for split, color in SPLITS:
            deltas = delta_by_split[split]
            if deltas is None:
                continue
            ax.plot(frame_by_split[split].numpy(), deltas[:, idx].numpy(), color=color, linewidth=1.2, label=split)
        ax.set_xlabel("frame")
        ax.set_ylabel(name)
        _style_axis(ax)
    for ax in axes.flat[len(DELTA_CHANNELS):]:
        ax.axis("off")
    axes.flat[0].legend(frameon=False)
    fig.suptitle("Ground-truth per-step deltas — train vs val")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_delta_distributions(delta_by_split: dict[str, torch.Tensor | None], out_path: Path) -> Path | None:
    if all(delta_by_split[split] is None for split, _ in SPLITS):
        print("skipping delta distribution plot: no deltas available for either split (pass --root)")
        return None
    channels = [("dist_ground", 3), ("dyaw_deg", 5)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, (name, idx) in zip(axes, channels):
        for split, color in SPLITS:
            deltas = delta_by_split[split]
            if deltas is None:
                continue
            ax.hist(deltas[:, idx].numpy(), bins=30, color=color, alpha=0.55, label=split, edgecolor="none")
        ax.set_xlabel(name)
        ax.set_ylabel("count")
        _style_axis(ax)
    axes[0].legend(frameon=False)
    fig.suptitle("Per-step delta distributions — train vs val")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_action_balance(action_by_split: dict[str, torch.Tensor], out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    x = np.arange(len(ACTIONS))
    width = 0.35
    for (split, color), offset in zip(SPLITS, (-width / 2, width / 2)):
        counts = action_by_split[split].numpy().sum(axis=0)
        ax.bar(x + offset, counts / counts.sum(), width=width, color=color, label=split)
    ax.set_xticks(x)
    ax.set_xticklabels(ACTIONS)
    ax.set_ylabel("fraction of transitions")
    ax.set_title("Action balance — train vs val")
    ax.legend(frameon=False)
    _style_axis(ax)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def phase3_validate_data(embeddings_path: str, out_dir: str | Path, root: str | Path | None = None) -> dict[str, Path]:
    all_data = torch.load(embeddings_path)
    out_dir = Path(out_dir)

    pos_by_split = {split: all_data[split]["pos_t"] for split, _ in SPLITS}
    frame_by_split = {split: all_data[split]["frame_t"] for split, _ in SPLITS}
    action_by_split = {split: all_data[split]["action"] for split, _ in SPLITS}
    if any(pos is None for pos in pos_by_split.values()):
        raise ValueError(f"{embeddings_path!r} is missing position data for one or both splits — "
                         "re-run eval.py --save-embeddings on a dataset with transitions.csv")

    delta_by_split: dict[str, torch.Tensor | None] = {}
    if root is not None:
        for split, _ in SPLITS:
            delta_by_split[split] = load_deltas_for_split(root, frame_by_split[split], all_data[split]["frame_tp1"])
        if all(d is None for d in delta_by_split.values()):
            print(f"note: no transitions.csv under {root} — skipping delta plots")
    else:
        delta_by_split = {split: None for split, _ in SPLITS}
        print("note: no --root given — skipping delta plots (pose-only validation)")

    check_pose_continuity(all_data)

    plot_fns = {
        "trajectory": lambda p: plot_trajectory(pos_by_split, p),
        "pose_over_frames": lambda p: plot_pose_over_frames(pos_by_split, frame_by_split, p),
        "pose_distributions": lambda p: plot_pose_distributions(pos_by_split, p),
        "deltas_over_frames": lambda p: plot_deltas_over_frames(delta_by_split, frame_by_split, p),
        "delta_distributions": lambda p: plot_delta_distributions(delta_by_split, p),
        "action_balance": lambda p: plot_action_balance(action_by_split, p),
    }
    paths = {}
    for name, fn in plot_fns.items():
        result = fn(out_dir / f"validate_{name}.png")
        if result is not None:
            paths[name] = result
            print(f"wrote {result}")
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="phase", required=True)

    p1 = sub.add_parser("phase1", help="cosine-similarity correlation matrices, exported as PNGs")
    p1.add_argument("--embeddings", required=True)
    p1.add_argument("--split", choices=["train", "val"], default="train")
    p1.add_argument("--out-dir", default=None, help="default: <embeddings dir>/plots")

    p2 = sub.add_parser("phase2", help="build the HD associative-memory trace")
    p2.add_argument("--embeddings", required=True)
    p2.add_argument("--split", choices=["train", "val"], default="train")
    p2.add_argument("--hd-dim", type=int, default=256)
    p2.add_argument("--out", default=None, help="default: <embeddings dir>/associative_memory_<split>.pt")
    p2.add_argument("--content-seed", type=int, default=0)
    p2.add_argument("--time-seed", type=int, default=42)
    p2.add_argument("--x-seed", type=int, default=1)
    p2.add_argument("--y-seed", type=int, default=2)
    p2.add_argument("--root", default=None,
                    help="dataset root containing transitions.csv; if given, per-transition "
                         "gt deltas (dx/dy/dz_world, dist_ground, dyaw_rad/deg) are loaded "
                         "and saved alongside the memory trace as 'delta_t'")

    p2b = sub.add_parser("phase2b", help="fidelity/orthogonality of time, position, heading, and delta HD encodings")
    p2b.add_argument("--embeddings", required=True)
    p2b.add_argument("--split", choices=["train", "val"], default="train")
    p2b.add_argument("--hd-dim", type=int, default=256)
    p2b.add_argument("--out-dir", default=None, help="default: <embeddings dir>/plots")
    p2b.add_argument("--time-seed", type=int, default=42)
    p2b.add_argument("--x-seed", type=int, default=1)
    p2b.add_argument("--y-seed", type=int, default=2)
    p2b.add_argument("--yaw-seed", type=int, default=3)
    p2b.add_argument("--action-seed", type=int, default=4)
    p2b.add_argument("--length-scale", type=float, default=1.0,
                     help="FPE length scale for time/position/delta encoders "
                          "(heading uses circular=True instead, no length_scale)")
    p2b.add_argument("--yaw-max-freq", type=int, default=3,
                     help="max |integer frequency| for the circular-safe heading base "
                          "(hard ceiling of 3 == floor(pi); see Phasor.__init__)")
    p2b.add_argument("--root", default=None,
                     help="dataset root containing transitions.csv; if given, also evaluates "
                          "delta_t (dx/dy/dz_world, dist_ground, dyaw) encoding fidelity")

    p3 = sub.add_parser("phase3", help="validate pose/heading/delta loading, train vs val, as PNGs")
    p3.add_argument("--embeddings", required=True)
    p3.add_argument("--out-dir", default=None, help="default: <embeddings dir>/plots")
    p3.add_argument("--root", default=None,
                    help="dataset root containing transitions.csv; if given, also validates "
                         "per-transition deltas (dx/dy/dz_world, dist_ground, dyaw_rad/deg)")

    args = parser.parse_args()
    if args.phase == "phase1":
        out_dir = args.out_dir or Path(args.embeddings).parent / "plots"
        phase1_correlations(args.embeddings, args.split, out_dir)
    elif args.phase == "phase2":
        out = args.out or Path(args.embeddings).parent / f"associative_memory_{args.split}.pt"
        phase2_build_memory(args.embeddings, args.split, args.hd_dim, out,
                            args.content_seed, args.time_seed, args.x_seed, args.y_seed,
                            root=args.root)
    elif args.phase == "phase2b":
        out_dir = args.out_dir or Path(args.embeddings).parent / "plots"
        phase2b_encoding_fidelity(args.embeddings, args.split, args.hd_dim, out_dir,
                                  args.time_seed, args.x_seed, args.y_seed, args.yaw_seed, args.action_seed,
                                  args.length_scale, args.yaw_max_freq, root=args.root)
    else:
        out_dir = args.out_dir or Path(args.embeddings).parent / "plots"
        phase3_validate_data(args.embeddings, out_dir, root=args.root)
