# B1 — Open-vocabulary frontend swap

**Goal:** replace our YOLOv8n-COCO detector with an open-vocabulary
detector prompted with Replica's own class list, across all 8 scenes.
Everything downstream (CLIP relabelling, the 32 KB trace, the scorer) is
untouched — this task changes ONE stage only, and the script for it is
already written and tested: **`collab_tasks/scripts/embed_crops_openvocab.py`**.

**Why this first:** see `README.md` in this folder for the full derivation.
Short version — of the 24 classes ConceptGraphs' own metric scores on
room0, our detector's COCO vocabulary can only ever name 7. That caps any
downstream memory at ≈0.29 mAcc before it even starts, and we measured
that deleting the memory entirely (replacing it with an explicit instance
list on the *same* detections) does not close the gap — it scores *lower*
(0.066) than keeping the memory (0.091). A perfect frontend on the same
representation reaches 0.634. **The bottleneck is measured to be the
frontend.** If a wider-vocabulary detector raises our score, and the 32 KB
trace tracks it upward, that's the paper's strongest sentence: *the memory's
score follows its frontend, not the other way round*.

**Pre-registered expectation (state before you run it, don't retrofit
after):** mAcc under ConceptGraphs' `n_exclude 6` protocol should rise
materially above our 0.091, plausibly into the 0.2–0.4 range on room0. If
it does *not* rise, that's equally reportable — it would mean the memory
is a second, independent bottleneck we haven't yet isolated, and that's
worth knowing too.

## Setup

```bash
pip install "ultralytics>=8.1"   # YOLO-World / model.set_classes()
pip install transformers          # CLIP crop embedding (already a repo dep)
```

## Run it — one command per scene, or all at once

```bash
# from the repo root
python collab_tasks/scripts/embed_crops_openvocab.py --scene room0

# or all 8 in one go
python collab_tasks/scripts/embed_crops_openvocab.py \
    --scene room0 room1 room2 office0 office1 office2 office3 office4
```

Start with `room0` alone (ConceptGraphs' own demo scene) before committing
GPU time to the rest. If detections look sparse, try a lower
`--conf` (default 0.15 — open-vocab detectors are typically noisier than
COCO-trained YOLO).

**What it does, and why it's safe to run:** it clones each scene's dataset
config to `vsa_cognitive_mapping/configs/replica_<scene>_openvocab.json`
(same raw frames and poses, new name) and writes its output to a brand-new
`outputs/replica_<scene>_openvocab/` directory. **It never reads or writes
the existing YOLO-COCO baseline files** — those are already measured
(0.091 mAcc) and cited in the paper draft, and this script is built so
there is no way for it to overwrite them, even by mistake.

At the end it prints the exact three follow-on commands per scene — they're
also reproduced below.

## Score it (prints automatically at the end of the run above)

```bash
python -m vsa_cognitive_mapping.object_grounding \
    --dataset vsa_cognitive_mapping/configs/replica_room0_openvocab.json \
    --gt-json outputs/replica_room0/gt_instances.json --relocalize

python student_gpu_package/04_vsa_labels.py \
    --scene room0_openvocab --gt-scene room0

python student_gpu_package/05_score.py --scene room0_openvocab
```

`--gt-scene room0` tells stage 4 to score against room0's existing ground
truth (same physical scene, same GT — no need to regenerate it) while
keeping every output under the `_openvocab`-suffixed paths. Repeat with
each scene name for the other 7.

## Self-check (must pass before sending anything back)

The final command must print a `their protocol (n_exclude 6)` line with a
real, non-zero mAcc/F-mIoU — not an error, not `skipped`. A common
silent-failure sign is every prediction landing on one catch-all class;
open `student_gpu_package/handoff/room0_openvocab/scores.json` and check
the per-class breakdown isn't degenerate before sending.

## What to send back

Per scene: `outputs/replica_<scene>_openvocab/detections_crops.csv`,
`crop_embeddings_openvocab.pt`, and
`student_gpu_package/handoff/<scene>_openvocab/scores.json`. Plus: which
detector checkpoint you used, `--conf` value, and wall-clock per scene. If
a scene fails, send its console log rather than a guessed fix.
