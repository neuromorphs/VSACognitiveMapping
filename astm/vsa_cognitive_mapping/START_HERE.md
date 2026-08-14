# Start here

A hypervector memory for robot observations. Every object detection is bound
into a **fixed-size** vector as `what ⊗ where ⊗ when`, and questions are
answered by algebra — divide out what you know, correlate what is left against a
grid to read off what you don't.

The open research question, and what most of this code measures, is:
**learned image features are badly shaped for that algebra, and it is not
obvious what to do about it.** See [CONCEPTS.md](docs/CONCEPTS.md) for the idea
and [RESULTS_SO_FAR.md](docs/RESULTS_SO_FAR.md) for what is already known —
including several things that turned out to be wrong.

---

## 1. Install

```bash
git clone -b astm https://github.com/neuromorphs/VSACognitiveMapping.git
cd VSACognitiveMapping
pip install torch torchvision transformers datasets ultralytics \
            numpy pandas pillow matplotlib
```

**Python version matters.** Use **3.10 or newer** for everything in `astm/`.
Some of the wider repo (and Meta's DINOv2 hub code) uses `str | Path`
annotations that are a syntax error on 3.9. If you are stuck on 3.9 you will hit
`TypeError: unsupported operand type(s) for |`.

**GPU.** Nothing here needs one, but the encoder passes are the slow part and
they scale straight onto a GPU — see §4.

Quick check:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## 2. Point it at data

Datasets are described by a small JSON file, so trying new data is
configuration rather than code editing. Full guide: [DATASETS.md](DATASETS.md).

**Always validate first:**

```bash
python -m vsa_cognitive_mapping.sequences validate \
    --dataset vsa_cognitive_mapping/configs/classroom_seq.json
```

This decodes a few frames and checks the pose stream is physically plausible.
It exists because a diverged SLAM solution is *silent* — it produces
plausible-looking metres and poisons every pose-dependent result. One of the
bundled sequences fails it on purpose:

```
school_run1: 11211 frames, 455.0 s
  verdict: UNUSABLE
    - 874 consecutive jumps over 5 m
    - peak implied speed 1594.8 m/s exceeds 5.0 m/s
```

**You do not need poses for most of the work.** Crop features, the isotropy
ladder, the rogue-dimension test, the encoder comparison and class retrieval all
run on images alone. Only crosstalk, place recall and the memory build need
poses. That widens your dataset options a lot.

---

## 3. The pipeline, in order

```bash
# (a) per-detection appearance features -- everything else needs these
python -m vsa_cognitive_mapping.classroom_pipeline embed-crops \
    --dataset vsa_cognitive_mapping/configs/classroom_seq.json --stride 1

# (b) the same crops through other backbones
python -m vsa_cognitive_mapping.encoder_comparison \
    --dataset vsa_cognitive_mapping/configs/classroom_seq.json \
    --encoders resnet50 dinov2 resnet50-untrained

# (c) analyses -- these read --out-dir, no dataset config needed
python -m vsa_cognitive_mapping.isotropy_ladder --out-dir outputs/classroom
python -m vsa_cognitive_mapping.encoder_report  --out-dir outputs/classroom
python -m vsa_cognitive_mapping.heldout_eval    --out-dir outputs/classroom

# (d) pose-dependent -- only if `validate` said YES
python -m vsa_cognitive_mapping.crosstalk_scaling --out-dir outputs/classroom \
    --content crop_embeddings_dinov2.pt --tag "classroom dinov2"
python -m vsa_cognitive_mapping.vsa_kalman --out-dir outputs/classroom --sweep

# (e) the playable page
python tools/export_video_page.py --scene classroom
python tools/build_video_html.py  --scene classroom
```

Step (a) is YOLOv8n over every frame; on CPU the 2,478-frame classroom takes
roughly 25 minutes. Step (b) is the one that wants a GPU.

---

## 4. Using your GPU, and a better DINO

`encoder_comparison` takes `--device` and `--batch`, and DINOv2 comes in four
sizes:

| `--encoders` value | model | output dim |
|---|---|---|
| `dinov2` *(= `dinov2:small`)* | `facebook/dinov2-small` | 384 |
| `dinov2:base` | `facebook/dinov2-base` | 768 |
| `dinov2:large` | `facebook/dinov2-large` | 1024 |
| `dinov2:giant` | `facebook/dinov2-giant` | 1536 |

```bash
python -m vsa_cognitive_mapping.encoder_comparison \
    --dataset vsa_cognitive_mapping/configs/classroom_seq.json \
    --encoders dinov2:base dinov2:large \
    --device cuda --batch 192
```

Each variant writes its own file (`crop_embeddings_dinov2-base.pt`), so they
coexist rather than overwrite. `--device auto` (the default) picks CUDA when it
is available.

**Two things to keep fixed unless you mean to change them.** `--img-size`
defaults to 224 and must be a multiple of DINOv2's patch size 14; every result
in this repo was measured at 224, so changing it makes your numbers
incomparable. And preprocessing is a **square bilinear resize**, deliberately
*not* the HuggingFace processor's shortest-edge-then-centre-crop — on an object
box with an extreme aspect ratio a centre crop silently throws away most of the
object.

**A good first experiment for you:** does a larger backbone change the
conclusions? Every result so far used `dinov2:small`, and the most interesting
finding is that the *whitened* crosstalk floor is encoder-independent. Does that
still hold for `large` and `giant`? That is a real open question, it needs a
GPU, and the answer is publishable either way.

---

## 5. Read the results before trusting any number

Several published-looking numbers in older docs on this branch are **wrong or
superseded**. [RESULTS_SO_FAR.md](docs/RESULTS_SO_FAR.md) lists them. The three
that will bite you fastest:

1. **Never quote a whitening multiplier without its encoder.** It is 19.9× on
   YOLOv8n and 8.1× on DINOv2 — measured on the same frames.
2. **A random train/test split is invalid here.** At 15 fps the neighbours of a
   held-out frame are still in the memory. Use `--modes blocked`.
3. **Always compute a chance baseline.** In a 6.9 m room, chance was 5.24 m and
   the best achievable 2.02 m, so "3.6 m error" alone means nothing.

---

## 6. Where to go next

| Document | What it is |
|---|---|
| [CONCEPTS.md](docs/CONCEPTS.md) | The ideas: binding, bundling, isotropy, crosstalk — no prior VSA knowledge assumed |
| [RESULTS_SO_FAR.md](docs/RESULTS_SO_FAR.md) | Every measured result, the corrections, and the open questions |
| [ADVANCED_SIGREG_VSA.md](docs/ADVANCED_SIGREG_VSA.md) | The research direction: training an encoder whose output is *made* to suit the algebra |
| [DATASETS.md](DATASETS.md) | Adding your own data |
| [docs/README.md](docs/README.md) | The playable pages and briefings |

---

## 7. GPU task: ConceptGraphs head-to-head on Replica room0

Context: we are comparing the VSA object memory against ConceptGraphs
(https://concept-graphs.github.io/, ICRA 2024) on the SAME data — Replica
room0, the demo scene their own repo ships configured for. The CPU half
(our pipeline + the grounding battery in
`vsa_cognitive_mapping/object_grounding.py`) is done on Paul's machine;
your half needs a GPU. Tracker entry (ay) in
`wiki/analysis/2026-07-29-vsa-query-layer-paper-plan.md` has the full
scoping — read it first.

Deliverables (three small files back to Paul; do NOT commit the data):

1. **Their object map on room0.** Run the official pipeline
   (https://github.com/concept-graphs/concept-graphs) on the NICE-SLAM
   room0 render (their README's demo path). Export each mapped object as
   JSON: `[{"class": str, "x": float, "y": float, "z": float}]` (object
   centroid, world frame of traj.txt).
2. **GT instance centroids.** Their eval tooling reads the Replica
   semantic mesh; export `[{"class": str, "x": ..., "y": ..., "z": ...}]`
   for every GT instance in room0. This becomes `--gt-json` for BOTH
   systems, so the two are scored by the identical script.
3. **Systems numbers.** Their map's size on disk (bytes), wall-clock build
   time, and GPU used — one line of JSON. These fill the systems columns
   (our trace is 32 KB; the interesting contrast is capability, not a
   gotcha: their graph supports relational queries ours cannot).

Notes: keep 2D scoring in mind — we project to the floor plane (the two
largest-variance axes of traj.txt translations; the script prints which).
If their pipeline versions drift (SAM/CLIP checkpoints), record exact
versions — the comparison must state them.

**Deliverable 4 (added 2026-08-05) — their OBSERVATION STREAM, not just the
final map.** Export ConceptGraphs' per-frame object observations as JSON:
`[{"frame": int, "class": str, "x": ..., "y": ..., "z": ..., "conf": ...}]`
(one row per per-frame object detection they associate, world frame). Why:
the fair comparison is same-frontend-different-backend. With their stream we
run OUR VSA backend on THEIR perception (and our `instances` backend already
mimics their object layer on our perception), completing the 2×2:
frontend (YOLO+depth | SAM+CLIP) × backend (instance graph | VSA trace).
`vsa_cognitive_mapping/object_grounding.py` consumes this stream directly
(same schema as `outputs/replica_room0/object_points.json`).

**Deliverable 2 update:** Paul's side now extracts GT instance centroids
from the vMAP renders CPU-side (`tools/replica_gt_from_renders.py`), so your
semantic-mesh export becomes a CROSS-CHECK of that GT rather than the only
source — still wanted, lower priority than deliverables 1 and 4.
