# B3 — DINOv2-large instance keys

**Goal:** produce 1024-d DINOv2-large crop embeddings for every detection,
across all 8 Replica scenes. Smallest and most mechanical of the three
tasks — a queued rerun of code that already exists, just with a bigger
backbone.

**Why:** feeds a separate, already-scoped question — the
instance-vs-semantic problem (`vsa_cognitive_mapping/instance_recall.py`):
when two objects share a class ("chair", "chair"), what lets the memory
tell them apart? The current appearance keys use plain DINOv2 (small);
DINOv2-large keys are the queued next rung of that ladder, not part of the
ConceptGraphs comparison in B1/B2.

**Pre-registered expectation:** larger, higher-dimensional appearance keys
should improve (or at least not hurt) the multi-instance disambiguation
numbers already measured with the small backbone — see `instance_recall.py`
results in the tracker (`wiki/analysis/2026-07-29-vsa-query-layer-paper-plan.md`,
entry (bf)) for the baseline to compare against.

## What to run

This is already implemented — one command per scene, nothing to write:

```bash
python -m vsa_cognitive_mapping.encoder_comparison \
    --dataset vsa_cognitive_mapping/configs/replica_<scene>.json \
    --encoders dinov2:large --batch 64
```

Run for `room0 room1 room2 office0 office1 office2 office3 office4`.
Each scene is on the order of ~1,000–8,000 crops (see
`outputs/replica_<scene>/detections_crops.csv` row counts) — expect
minutes per scene on any modern GPU, not hours.

If you'd rather run this on Colab alongside B2 or standalone, the
`dinov2-large` stage is already present (commented out) in
`experiments/COLAB_EMBED_GPU.ipynb` — just uncomment it in the `STAGES`
list near the top of the notebook.

## Output schema (already defined by the command above)

`outputs/replica_<scene>/crop_embeddings_dinov2-large.pt` — a torch dict
with the same layout as the existing `crop_embeddings_dinov2.pt`
(`det_id`, `frame_idx`, `embedding: FloatTensor[N, 1024]`, `meta`).

## Self-check

```bash
python -c "
import torch
d = torch.load('outputs/replica_room0/crop_embeddings_dinov2-large.pt', weights_only=False)
assert d['embedding'].shape[1] == 1024, d['embedding'].shape
print('OK', d['embedding'].shape)
"
```

## What to send back

The eight `.pt` files (a few MB each). No further processing needed —
`instance_recall.py` and the scoring that follows run locally in seconds
once these land.
