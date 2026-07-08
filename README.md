# VSACognitiveMapping

## Summary

TODO

## Members

- TODO

## Setup

Create a virtual environment and install the project:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### PyTorch

PyTorch is not pinned in `pyproject.toml` because the correct build depends on your OS and hardware (CPU-only, CUDA, ROCm, etc.). Install it separately using the selector at [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) for the command that matches your system.

Install PyTorch *before* the `detection` extra (`pip install -e ".[notebook,detection]"`) — `ultralytics` depends on `torch`, and installing it afterwards will pull in a generic build rather than the one you picked for your hardware.

## Associative memory pipeline

`scripts/associative_memory.py` runs in phases over embeddings exported by
`scripts/eval.py`. First, export per-frame latents (plus ground-truth pose,
if the dataset has `transitions.csv`) for both splits:

```bash
python scripts/eval.py --config configs/jepa_sim.yaml --checkpoint checkpoints/jepa_sim/final.pt \
    --save-embeddings checkpoints/jepa_sim/embeddings.pt
```

**Phase 1 — correlations:** cosine-similarity matrices (encoder self-consistency,
predictor self-consistency, predicted-vs-actual next latent), saved as PNGs.

```bash
python scripts/associative_memory.py phase1 \
    --embeddings checkpoints/jepa_sim/embeddings.pt --split train
```

**Phase 2 — build the HD memory trace:** binds each frame's latent to its time
and position and bundles them into one associative-memory trace. Add `--root`
to also load and attach per-transition ground-truth deltas (`dx/dy/dz_world,
dist_ground, dyaw_rad/deg`) for later use.

```bash
python scripts/associative_memory.py phase2 \
    --embeddings checkpoints/jepa_sim/embeddings.pt --split train \
    --root data/dataset_vjepa_bezier
```

**Phase 3 — validate data loading:** sanity-check plots (trajectory, pose and
delta channels over time, train/val distributions, action balance) confirming
pose/heading/delta data lines up correctly before it's baked into a trace.

```bash
python scripts/associative_memory.py phase3 \
    --embeddings checkpoints/jepa_sim/embeddings.pt --root data/dataset_vjepa_bezier
```

Each phase defaults its output (`--out-dir` / `--out`) to a `plots/` folder or
file next to the embeddings; pass `--help` on any subcommand for the full
argument list.

## Acknowledgments

TODO: acknowledge the Telluride Neuromorphic AI Workshop.
