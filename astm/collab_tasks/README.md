# Collaborator GPU tasks — ConceptGraphs head-to-head

Three independent, GPU-shaped tasks that feed the same paper section
(ICRA 2027, `paper/main.tex` §VII, semantic-map comparison). Each is
self-contained — pick any one, run it on whatever GPU you have, send back
the (small) output folder. They do not depend on each other and can run in
parallel on different machines.

## Why these three, in this order

We scored our 32 KB object memory on Replica room0 under ConceptGraphs'
own metric and their own class list (`--n_exclude 6`, excluding
other/floor/wall/ceiling/door/window — see their `scripts/eval_replica_semseg.py`).
Result: **0.091 mAcc** against their published **0.406**.

To find out *why*, we built a same-points, same-ground-truth ladder that
holds the scoring rule fixed and swaps only the source of the object list:

| rung | source of the object list | mAcc | F-mIoU |
|---|---|---|---|
| our VSA trace (32 KB) | our detector → CLIP relabel → trace → argmax | 0.091 | 0.077 |
| explicit instance list, **same detections** | our detector → CLIP relabel → cluster → centroids (memory deleted) | 0.066 | 0.050 |
| oracle instance list | ground-truth instances (perfect frontend) | 0.634 | 0.326 |
| *ConceptGraphs, published* | *their pipeline, their data, their GT — a citation, not a rung here* | *0.406* | *0.360* |

Deleting the memory (rung 2) does not recover the gap — it scores *lower*
than keeping it. The oracle (rung 3) does. **The bottleneck measured today
is the frontend** (our YOLOv8-COCO detector + box/depth/pose placement),
not the memory. That is what makes all three tasks below worth running:
they attack the frontend or complete the missing half of the comparison,
rather than tuning the memory, which the data says is not where the loss is.

Full derivation, code, and the room0 numbers: `outputs/replica_room0/`,
`student_gpu_package/handoff/room0/`, tracker entry (bj)/(bk) in
`wiki/analysis/2026-07-29-vsa-query-layer-paper-plan.md`.

## Common rules for all three

- **Scenes:** all 8 Replica scenes we already have locally —
  `room0 room1 room2 office0 office1 office2 office3 office4`.
  `room0` is ConceptGraphs' own demo scene; run it first as a smoke test
  before committing GPU time to the rest.
- **Deliverable size:** each task returns megabytes, not gigabytes, in an
  existing on-disk schema (below). If your output doesn't match the
  schema, the receiving scripts will fail loudly and say why — that's
  intentional, please send the error rather than reshaping the data to fit.
- **Self-check before sending anything back:** each task has a script
  that must print `STAGE OK` (or equivalent) before you zip and send the
  output. A partial result with a clear failure log is far more useful
  than a "finished" run nobody checked.
- **Pre-registered expectation:** each brief states what we expect to see
  *before* you run it. Report what you actually got, including if it
  disagrees — a surprising result is data, not a bug to hide.
- **No invented metrics:** every number reported must come from a
  scorer already in this repo, or verbatim from the target paper's own
  evaluation code. Do not compute a new metric to make a number look
  better or worse.

## The three tasks

1. **[B1 — open-vocabulary frontend swap](B1_open_vocab_frontend.md)**
   (start here — highest expected value, lightest install)
2. **[B2 — ConceptGraphs' own pipeline, all 8 scenes](B2_conceptgraphs_replica_run.md)**
   (heaviest install; the notebook is already hardened, see the gotchas list)
3. **[B3 — DINOv2-large instance keys](B3_dinov2_large_instance_keys.md)**
   (smallest task, ~10 min/scene, feeds a separate instance-recall question)
