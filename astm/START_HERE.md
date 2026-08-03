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
cd VSACognitiveMapping/astm          # <- run everything below from here
pip install -r requirements.txt
```

Two dependencies are optional and marked as such in the file. `datasets` is
only needed for the HuggingFace-hosted sequences — a folder of your own images
does not need it, because `sequences.py` imports it lazily. If you are bringing
your own data, you can skip it.

**Run from `astm/`.** The package is `astm/vsa_cognitive_mapping/`, and the
modules add that directory to `sys.path` themselves, so `python -m
vsa_cognitive_mapping.<module>` works from `astm/` with no install step. Run it
from the repo root instead and you get
`ModuleNotFoundError: No module named 'vsa_cognitive_mapping'`. Every path in
this document is relative to `astm/`.

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

## 1b. Run these two first — no data, no GPU, under a minute

Before downloading anything, confirm the algebra works on your machine:

```bash
python -m vsa_cognitive_mapping.test_vsa                  # 18 correctness tests
python -m vsa_cognitive_mapping.demo_associative_memory   # synthetic walk demo
```

`test_vsa` checks bind/unbind/bundle, the FPE homomorphism, unit modulus and
projection shapes — expect `18/18 passed`. It needs only `numpy` and `torch`.

`demo_associative_memory` builds a small memory from a synthetic robot walk,
queries it back by position and time, prints the similarity kernel as ASCII, and
sweeps bundling capacity against dimensionality. Its last table is the whole
capacity story in miniature:

```
    hd_dim | exact acc | mean pos err (m)
    -------+-----------+-----------------
       256 |   0.917   |   0.0228
       512 |   0.958   |   0.0000
      1024 |   0.958   |   0.0000
```

If those two run, everything else is a data problem rather than an install
problem.

### The playable pages

Rendered results you can open in a browser without running anything — see
[docs/README.md](docs/README.md) for all of them with one-click links. The two
worth opening first are the **classroom video** (three frame-by-frame players:
the walk, the memory recalling position under four different encoders, and what
each encoder retrieves for a query crop) and the **interactive demo**
(free-text queries, time queries, memory merge, then the mechanism explained).

Note the video pages' recall figures predate the held-out protocol, so they are
closed-set numbers — see [RESULTS_SO_FAR.md](docs/RESULTS_SO_FAR.md).

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
