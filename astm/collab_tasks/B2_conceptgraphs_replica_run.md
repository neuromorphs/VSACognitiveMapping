# B2 — ConceptGraphs' own pipeline, all 8 Replica scenes

**Goal:** run ConceptGraphs' actual code (not a reimplementation) on all
8 Replica scenes, and export **both** their per-scene result **and** their
per-object observation stream (detections + world positions before their
3D fusion step).

**Why this matters beyond "reproduce their number":** their observation
stream is what completes the comparison. Right now, one cell of the
frontend × backend grid is filled with a rough stand-in — a naive
single-link clustering *we* wrote over *our* detections, used only to show
that deleting our memory doesn't help (see `README.md` in this folder,
rung 2 of the ladder). Their real observation stream, run through our 32 KB
trace, replaces that stand-in with a real answer to: *does a scene-graph
frontend feeding OUR memory do better than it feeding THEIRS?*

**Pre-registered expectation:** their room0 mAcc/F-mIoU under their own
`--n_exclude 6` protocol should land near their paper's average (mAcc
≈0.40, F-mIoU ≈0.36 — see `gu2024conceptgraphs`, Table in their Replica
section). If it lands far from that, a config drifted somewhere in
install/run — stop and send the log rather than tuning toward the number.

## Everything you need already exists — start here, not from scratch

This is **not** a from-zero task. A complete, hardened package already
handles install, data prep, their pipeline invocation, export, and
scoring:

- **Walkthrough + troubleshooting table:** `student_gpu_package/README.md`
- **Colab notebook (no local GPU needed):**
  `experiments/COLAB_CONCEPTGRAPHS_L4.ipynb` — open via
  `colab.research.google.com/github/SynapticScotsman/Semantic-Spiking-Neural-SLAM-2023/blob/results-sites/experiments/COLAB_CONCEPTGRAPHS_L4.ipynb`
- **Six-stage local pipeline:** `student_gpu_package/00_setup.sh` through
  `05_score.py` — each stage prints `STAGE OK` on success; do not proceed
  past a failure, send the log instead

**Budget:** room0 (their demo scene) ≈ 30–90 min on an L4-class GPU
(SAM ViT-H) or a bit longer on a smaller card (MobileSAM, auto-selected if
VRAM < 10 GB). All 8 scenes ≈ 25–55 GPU-minutes-equivalent total once the
one-time environment setup (~20–35 min, dominated by a `pytorch3d` source
compile) is done. On Colab specifically, that compile result is now cached
to Drive after the first VM pays it — later VMs restore it in seconds.

## Traps already found and fixed — read before you hit them again

We ran this pipeline end-to-end on Colab and hit (and fixed) a long chain
of issues; all fixes are already in the notebook and package on the
`results-sites` branch, but if you're setting up on your own machine from
their README directly, watch for these:

1. **`torch.hub.load` interactive trust prompt** — newer torch asks
   "do you trust this repository?" and hangs forever in a non-interactive
   shell. Pass `trust_repo=True`, and note their EigenPlaces code
   internally hub-loads a second repo (CosPlace) that needs the same flag
   forced on *every* nested `torch.hub.load` call, not just your own.
2. **`supervision` API drift** — their `conceptgraph/utils/vis.py` uses
   `ColorPalette.default()` and an older annotator signature. Anything
   ≤0.16.0 won't install on Python 3.12; **0.17.1** is the oldest version
   that both installs on 3.12 and still carries their API (verified call by
   call against their actual usage, not guessed).
3. **`supervision==0.17.1` drags in a numpy-1-era `opencv-python-headless`
   build**, which breaks every `cv2` import under numpy 2 with
   `_ARRAY_API not found`. Immediately after installing supervision:
   `pip install --upgrade --no-deps opencv-python-headless`.
4. **`GSA_PATH` environment variable is required** by their
   `generate_gsa_results.py` and is not documented as mandatory anywhere
   obvious — set it to wherever you clone `Grounded-Segment-Anything`.
   Checkpoints (SAM / MobileSAM) go under `<GSA_PATH>/` and
   `<GSA_PATH>/EfficientSAM/` respectively — their loader hardcodes these
   relative paths.
5. **Their entry points are two direct scripts, not `python -m` modules:**
   `scripts/generate_gsa_results.py` (segmentation, flag-style args) then
   `slam/cfslam_pipeline_batch.py` (3D mapping, hydra `key=value` args),
   both run **from inside their cloned repo directory**. Use their README's
   exact parameter values for the class-agnostic variant
   (`class_set=none` / `gsa_variant=none`, `class_agnostic=True`,
   `skip_bg=True`) — this is the configuration that needs only SAM, no
   GroundingDINO/RAM.
6. **If you're on Colab:** keep the cloned repo and raw scene data on the
   VM's local disk (`/content/...`), not a Drive-mounted path — Drive's
   FUSE filesystem makes `git reset --hard` unreliable (can silently
   corrupt a resumed clone) and per-frame reads several times slower.
   Only small, valuable outputs (results, checkpoints cache) should live
   on Drive.

## Output schema (already defined — do not invent a new one)

Stage 3 (`student_gpu_package/03_export_cg.py`) produces, per scene, under
`student_gpu_package/handoff/<scene>/`:
- `cg_labels.npz` — their per-eval-point predicted class
- `cg_objects.json` — `[{class, x, y, z, n_detections}]`
- `cg_observations.json` — **the per-object observation stream** (the
  piece that fills the missing grid cell — please make sure this one
  exports even if you're tempted to skip it as "not needed for the score")
- `eval_points.npz` — the eval point cloud (xyz + GT label)
- `scores.json` — from stage 5, both systems under one scorer

## What to send back

The full `handoff/<scene>/` folder per scene (small — KB to low MB), plus
`environment.txt` (stage 0 writes exact package versions), plus wall-clock
per scene and which GPU/backbone (`sam_vit_h` vs `mobilesam`) you used. A
partial set — even just room0 — is useful; send the log for anything that
didn't complete rather than a guessed fix.
