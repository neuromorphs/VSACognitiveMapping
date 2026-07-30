# vsa_cognitive_mapping — standalone phasor/VSA core

A standalone port of the **phasor / VSA core** from
[`neuromorphs/VSACognitiveMapping`](https://github.com/neuromorphs/VSACognitiveMapping)
(the `jepa` branch), the Telluride Neuromorphic Workshop 2026 project, plus a
self-contained CPU demo that exercises it end to end.

This module is deliberately kept separate from `sspslam/` and the root-level
`snnhdc_*` scripts so it can be diffed directly against upstream. It does
**not** import or modify any existing project code.

## Contents

| File | What it is |
|------|-----------|
| [`vsa.py`](vsa.py) | The ported core. `Phasor` (bind / unbind / bundle / FPE / similarity, with `* / + ** %` operator sugar), `random_project_to_phasor` (real embeddings → unit-modulus I/Q phasors), and the analysis helpers `phasor_correlation_matrix`, `cosine_self_correlation`, `fidelity_score`, `orthogonality_score`, `pca_components`, `make_axis_bases`, `fpe_bundle_encode`. |
| [`test_vsa.py`](test_vsa.py) | All 18 upstream correctness tests, made pytest-free so they run with plain `python`. |
| [`demo_associative_memory.py`](demo_associative_memory.py) | A synthetic robot-walk cognitive-map demo — build a bundled memory trace, query it back by position/time, show the FPE kernel and bundling-capacity behaviour. No external data, no GPU. Prints a text report. |
| [`visualize.py`](visualize.py) | **Matplotlib** visualiser for the same run — saves PNG panels and an animated `demo.gif`, the way the upstream repo renders its results. |
| [`classroom_pipeline.py`](classroom_pipeline.py) | **Real-data** pipeline on the Spot classroom walk ([`lorinachey/spot-telluride-workshop-dataset`](https://huggingface.co/datasets/lorinachey/spot-telluride-workshop-dataset)): YOLOv8n detect+embed → per-axis position/heading/time memories → evaluate → demo gif. Ports the upstream `init-am` classroom scripts, including the circular FPE heading base. |
| [`object_map.py`](object_map.py) | **Constructive object map + live server.** Lifts each YOLO detection to a metric world location (box + D455 depth + pose), clusters into objects, binds them into one hypervector `M = Σ SEM_class ⊗ SSP(x,y)`, and serves an interactive browser map that runs the VSA unbind live per query. |
| [`astm_traces.py`](astm_traces.py) | **ASTM P0: multi-trace map state + query router.** Canonical event stream → four traces (`what⊗where`, `what⊗when`, `where⊗when`, `event`), closed-form temporal range kernels (probe-side), decode-target router with per-stage latency, vocabulary scan (`which_moved`), and Baseline-A exact event table with modal ground truth for automatic scoring (`export` / `build` / `query` / `bench`). See the ASTM plan page in `wiki/analysis/`. |

## Run it

```powershell
# from the repo root
python vsa_cognitive_mapping\test_vsa.py                 # 18/18 passed
python vsa_cognitive_mapping\demo_associative_memory.py  # prints a results report
python vsa_cognitive_mapping\demo_associative_memory.py --n-frames 32 --hd-dim 4096

# picture version -- writes PNGs + demo.gif to .\plots\ (add --show to pop windows)
python vsa_cognitive_mapping\visualize.py
python vsa_cognitive_mapping\visualize.py --show --no-gif
python vsa_cognitive_mapping\visualize.py --out-dir plots\vsa_demo --hd-dim 4096
```

Requires only `numpy` and `torch` for the core/demo; `visualize.py` also needs
`matplotlib` (+ `pillow` for the GIF), both already installed. If you install
`pytest`, `test_vsa.py` is also discoverable as a normal test module.

### Real Spot data (classroom walk)

The dataset for this part of the project is
[`lorinachey/spot-telluride-workshop-dataset`](https://huggingface.co/datasets/lorinachey/spot-telluride-workshop-dataset)
— a Boston Dynamics Spot classroom walk-through (2,478 D455 RGB frames at
640×480 + 496 LIO-SAM poses). Needs `pip install datasets huggingface_hub
ultralytics`. Raw data lands in the HuggingFace cache (outside OneDrive);
compact artifacts go to `outputs/classroom/`.

```powershell
# 1. YOLOv8n detections + 256-d embeddings per frame (slow part, CPU ~30 min at stride 2)
python vsa_cognitive_mapping\classroom_pipeline.py embed --stride 2

# 2. build three per-axis memories (position / heading / time), stride train/val split
#    PCA whitening is ON by default: raw YOLO content sits in a 0.76-1.00 similarity
#    band, which swamps content->context unbinding with crosstalk (--no-pca-whiten to ablate)
python vsa_cognitive_mapping\classroom_pipeline.py build --hd-dim 8192 --subset stride --subset-stride 3

# 3. quantitative self-recall (metres / radians / frames) -> evaluate.png
python vsa_cognitive_mapping\classroom_pipeline.py evaluate

# 4. held-out demo gif: RGB frame | position heatmap | heading polar
python vsa_cognitive_mapping\classroom_pipeline.py demo --demo-frames 48

# 5. object-centric semantic query: class name -> spatial heat field PNG
python vsa_cognitive_mapping\classroom_pipeline.py query_class --class-name "chair"

# 6. conjunctive spatiotemporal queries: give two of {class, time, place}, decode the third
python vsa_cognitive_mapping\classroom_pipeline.py query_event --class-name chair --at-time 150   # -> where?
python vsa_cognitive_mapping\classroom_pipeline.py query_event --class-name chair --at-x -1.1 --at-y -0.2  # -> when?
python vsa_cognitive_mapping\classroom_pipeline.py query_event --at-x 1.1 --at-y 1.5 --at-time 150 # -> what?
```

`build` additionally bundles a **conjunctive event trace**
`M_event = Σ conf · SEM_class ⊗ ctx_pos(x,y) ⊗ ctx_time(t)` — one triple-bound
product per detection. Because FHRR unbinding is exact, any *conditional* query
(fix all factors but one) is plain unbinding: `M_event ⊘ SEM_chair ⊘ ctx_time(150)`
leaves a pure position bundle to decode. A resonator network is only needed to
recover two or more unknowns simultaneously. Measured on the classroom walk:
time-conditioned chair positions decode to within ~0.1 m of the robot's true
pose at that time; under-unbinding (class only) leaves time bound in and
attenuates the position signal ~22× vs the `semantic_position` marginal — which
is exactly why the marginal memories are kept alongside the event trace.

`build` also bundles an **object-centric semantic memory**
`M_sem = Σ conf · SEM_class ⊗ ctx_pos(robot_x, robot_y)` from the YOLO
detections — each class label bound (confidence-weighted) to the *robot's* pose
when it saw that object. `query_class` unbinds it (`M_sem ⊘ SEM_class`) and
decodes a spatial field, so the heatmap answers **"from where were `chair`s
observed?"**. Different classes peak in different parts of the room (chair
front-centre, clock far-left corner, toilet top-right); peak strength tracks how
often a class was seen, so frequent classes (chair) read strongly and rare ones
sit near the bundling-crosstalk floor.

Two things to keep straight:
- This binds to the **robot's observation pose**, not the object's metric
  location — it's an *observation footprint*, not a "where is the object" map.
  For the object's own depth-derived location, use `object_map.py`.
- Per-class phasors are seeded with a **stable hash** (`hashlib`, not Python's
  per-process-salted `hash()`) and saved in `memory.pt`, so `build` and
  `query_class` in separate processes agree on the codebook.

Heading uses a **circular FPE base** (random integer frequencies, |k| ≤ 3) so
`ctx(ψ + 2π) = ctx(ψ)` exactly — the same construction as upstream
`Phasor(circular=True)`, which the `jepa`-branch core we ported predates.

### Constructive object map (interactive server)

Goes past recall to *building* a map: each YOLO detection is lifted to a metric
world location (box centre + median D455 depth deprojected through the camera
intrinsics, then robot pose + camera extrinsic), detections are clustered into
unique objects, and the objects are bound into one spatial hypervector
`M = Σ SEM_class ⊗ SSP(x,y)`. Needs the `depth_d455` config and the
`detections.csv` from `classroom_pipeline.py embed`.

```powershell
# 1. place detections in the world, cluster to objects -> object_map.json
python vsa_cognitive_mapping\object_map.py localize

# 2. run the live browser map (VSA unbind runs server-side per query)
python vsa_cognitive_mapping\object_map.py serve --port 8010
#    then open http://localhost:8010

# static preview instead of the server (optional)
python vsa_cognitive_mapping\object_map.py snapshot --query-class chair
```

In the browser: click a **class** (left) to run `M ⊘ SEM_class` and see its
similarity field peak at every object of that class; click **anywhere on the
map** to run `M ⊘ SSP(x,y)` and get the ranked "what's here?" class list. The
floorplan is the LIO-SAM trajectory (metrically exact); markers are objects,
sized by how many detections support them.

Caveats: camera extrinsics are approximated (a fixed forward/up body offset, no
exact mount calibration in the dataset), depth is assumed registered to colour,
and object classes are raw COCO YOLO — so an off-the-shelf detector in a
classroom occasionally emits "airplane"/"cat". The VSA map machinery is
agnostic to which detector supplies the objects.

### What `visualize.py` produces

Saved to the output folder (default `plots/`), mirroring the upstream repo's
PNG-panel + demo-video style:

| File | Panel |
|------|-------|
| `localization.png` | Reverse query one frame's appearance → similarity field over the floor, peak on the true spot. |
| `demo.gif` | Animated: step the queried frame along the walk and watch the localized peak track the robot (the upstream `demo.mp4` idea). |
| `fpe_kernel.png` | 1-D slice through the field — the fractional-power-encoding similarity kernel. |
| `capacity_sweep.png` | Exact-frame recall vs `hd_dim` (the bundling-capacity curve). |
| `evaluate.png` | Self-recall: true → recalled position, and per-frame error. |
| `correlations.png` | Content self-similarity matrix + the trajectory. |

## What the demo shows

A robot visits N frames along a trajectory; each frame has a position `(x, y)`,
a time `t`, and an appearance embedding (random 256-d vectors standing in for
YOLO/JEPA detector features). The appearance is projected to a phasor "content"
vector, then two associative memories are built by binding content to an
FPE-encoded context and bundling:

```
M_pos  = mean_n  content_n  ⊗  ctx_pos(x_n, y_n)      ctx_pos(x, y) = Bx**(x/l) ⊗ By**(y/l)
M_time = mean_n  content_n  ⊗  ctx_time(t_n)          ctx_time(t)   = Bt**(t/l)
```

Querying unbinds a memory by a candidate context and cleans up against the
stored content codebook — `M_pos ⊘ ctx_pos(x, y) → argmax similarity` answers
"what did I see at `(x, y)`?". This is the same mechanism as the upstream
`associative_memory.py` / classroom pipeline, in miniature. Keeping one memory
per axis (rather than one trace bound to all axes) follows the `init-am`
classroom design, so each memory only has to discriminate along its own
dimension.

Reference numbers (defaults, `seed=0`): position self-recall ≈ 0.96 exact-frame
with ~0 m mean error; the position similarity kernel peaks at the true `x`; and
exact-frame accuracy rises with `hd_dim` (256 → 2048), the bundling-capacity
tradeoff noted in the upstream docs.

## Provenance & local changes

- Ported **2026-07-14** from `VSACognitiveMapping` `jepa` branch,
  `src/vsa_cognitive_mapping/vsa.py` and `tests/test_vsa.py`.
- `vsa.py` is byte-for-byte upstream except for one added line,
  `from __future__ import annotations`, so the `int | None` style hints parse
  on this repo's **Python 3.9** (they would otherwise raise `TypeError` at
  import). Behaviour is identical.
- `test_vsa.py` drops the hard `pytest` dependency (this env has no pytest) via
  a tiny `approx` shim and a `__main__` runner; the assertions are unchanged.
- `demo_associative_memory.py` is new — it is our own minimal driver, written
  to depend only on the ported core.

## How this relates to the rest of the repo

The upstream `Phasor` is an **FHRR** representation (unit-modulus complex
vectors, elementwise-multiply binding) — the same VSA backend the local
`snnhdc_fhrr_*` scripts and the LeJEPA→FHRR canonical plan target. The main
differences worth noting when comparing:

- This core stores phasors as **raw complex** vectors and reads out with a
  `Re(⟨a, b⟩)/D` similarity — matching `snnhdc_fhrr_common.fhrr_similarity`.
- It adds **FPE** (fractional power encoding) for continuous position/time,
  which is the SSP-style fractional-binding idea from `sspslam`'s
  `HexagonalSSPSpace`, here applied to a learned-feature content channel.
