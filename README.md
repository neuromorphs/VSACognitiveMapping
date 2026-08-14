# VSACognitiveMapping

> **Looking for the GPU tasks to hand to students/collaborators?**
> → **[astm/collab_tasks/README.md](astm/collab_tasks/README.md)** — three
> ready-to-run tasks for the ConceptGraphs semantic-map comparison, pick
> any one. For everything else on this branch (the ASTM memory system
> itself), start at **[astm/README.md](astm/README.md)**.

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

## Encoder sweep

`scripts/encoder_sweep.py` compares three ways of turning a frame's JEPA
latent (`z_t`) into a VSA content vector, using the same `embeddings.pt`
export as above:

- **random-proj** — one random projection matrix, `d_in -> 2*hd_dim -> hd_dim`
  complex (the encoder phase 2 uses).
- **pca-fpe** — top-K PCA components of `z_t` (standardized to unit variance),
  each FPE-encoded with its own base phasor and bundled together. K is fixed
  (default 4, `--n-components`).
- **full-fpe** — same FPE-and-bundle scheme, but over every raw `z_t`
  dimension instead of a PCA subspace.

```bash
python scripts/encoder_sweep.py \
    --embeddings checkpoints/jepa_sim/embeddings.pt --split train
```

For each setting it computes two metrics — **fidelity** (correlation between
the encoded content's similarity matrix and the raw `z_t` cosine-similarity
matrix — does the encoding preserve which frames look alike?) and
**orthogonality** (1 − mean off-diagonal content similarity — how close the
content vectors are to mutually orthogonal, i.e. low interference when
bundled into a shared memory trace).

`--length-scale` (default 1.0) and `--n-components` (default 4) are both
fixed rather than swept: on this dataset, sweeping length_scale trades
fidelity for orthogonality along one curve in both directions (long length
scale collapses every frame to the same encoded vector; short length scale
aliases nearby latent values into near-random phase), and sweeping K showed
a non-monotonic fidelity peak around K=4–8 that falls off past it — so K=4
is the default (also the sweet spot the exploratory notebook found for the
same top-K PCA + FPE approach). Neither knob, nor bind instead of bundle,
closes the gap with random-proj's fidelity/orthogonality combination on
this data.

Output goes to separate directories per encoder under `--out-dir` (default
`<embeddings dir>/plots`): `random-proj/`, `pca-fpe/`, `full-fpe/`.

For the full derivation — Phasor algebra, all three content encoders, the
memory-trace formula, and the fidelity/orthogonality metrics — see
[docs/associative_memory_math.md](docs/associative_memory_math.md).

## Acknowledgments

TODO: acknowledge the Telluride Neuromorphic AI Workshop.
