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

Phase 3 — validate data loading: sanity-check plots for the pose, heading,
and delta data that phases 1/2 rely on, split by train/val, so a bad
frame_t/pos_t alignment or a lopsided split shows up before it's baked into
a memory trace. Pass --root to also validate the per-transition deltas
(dx/dy/dz_world, dist_ground, dyaw_rad/deg); without it, only pose is
checked.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from vsa_cognitive_mapping.data import ACTIONS, load_deltas_by_transition
from vsa_cognitive_mapping.vsa import Phasor, random_project_to_phasor

# Fixed categorical identity: train is always blue, val is always orange,
# across every plot in this file (see dataviz skill — color follows the
# entity, never its rank). Grid/muted values match the same reference palette.
TRAIN_COLOR = "#2a78d6"
VAL_COLOR = "#eb6834"
GRID_COLOR = "#e1e0d9"
MUTED_COLOR = "#898781"


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
    else:
        out_dir = args.out_dir or Path(args.embeddings).parent / "plots"
        phase3_validate_data(args.embeddings, out_dir, root=args.root)
