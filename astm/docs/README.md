# ASTM documentation — read these in the browser

GitHub renders `.html` as **source code**, not as a page. Use the "open
rendered" links below — they need no repo configuration; both services just
proxy this branch's raw files. Every page is **self-contained**: figures, data
and frames are inlined, so nothing is fetched at view time and they work
offline once loaded.

---

## ▶ Start here — the three playable pages

| Page | Open it | What you get |
|---|---|---|
| **1. Classroom video** <br>*best first look* | [open rendered ▸](https://raw.githack.com/neuromorphs/VSACognitiveMapping/astm/astm/docs/video_classroom.html) · [mirror](https://htmlpreview.github.io/?https://github.com/neuromorphs/VSACognitiveMapping/blob/astm/astm/docs/video_classroom.html) | Three frame-by-frame players over a 2,478-frame Spot walk. **Flythrough**: camera + detections beside the walked path. **Memory recall**: the similarity field the map decodes each frame, with an encoder toggle so the *same* query is answered by four different memories, and a "worst frame" button. **Encoder comparison**: a query crop and the nearest neighbour each backbone returns. ~4.8 MB. |
| **2. School run video** | [open rendered ▸](https://raw.githack.com/neuromorphs/VSACognitiveMapping/astm/astm/docs/video_school_run1.html) · [mirror](https://htmlpreview.github.io/?https://github.com/neuromorphs/VSACognitiveMapping/blob/astm/astm/docs/video_school_run1.html) | The same three players over the 11,211-frame scale-up sequence (41,697 detections, three encoders). **Read the red banner**: this sequence's LIO-SAM odometry has diverged, so its recall panel demonstrates *failure*, not performance. The flythrough and encoder comparison use no poses and are unaffected. ~4.8 MB. |
| **3. Interactive demo** | [open rendered ▸](https://raw.githack.com/neuromorphs/VSACognitiveMapping/astm/astm/docs/demo_page.html) · [mirror](https://htmlpreview.github.io/?https://github.com/neuromorphs/VSACognitiveMapping/blob/astm/astm/docs/demo_page.html) | Demos first, mechanism afterwards: replay of the memory localising a walking Spot (median 0.150 m, with a jump-to-worst-frame button for the 6.85 m failure), free-text queries ("cold storage for food" → the fridge, 0.24 m), time-conditioned queries, and the memory merge. Then a live explainer of the algebra, then a limits section stating what the page does *not* show. 2.4 MB. |

---

## Results and analysis

| Document | Open it | What it is |
|---|---|---|
| **Anisotropy results** <br>*the current scientific finding* | [open rendered ▸](https://raw.githack.com/neuromorphs/VSACognitiveMapping/astm/astm/docs/results_anisotropy.html) · [mirror](https://htmlpreview.github.io/?https://github.com/neuromorphs/VSACognitiveMapping/blob/astm/astm/docs/results_anisotropy.html) | Four independent experiments showing **isotropy predicts bundling capacity, not key quality**. The rogue-dimension test (ours is spectral, not the NLP pathology), the encoder four-way (Liang replicated, Godey inverted), the isotropisation ladder, and the two-scene comparison — including one falsified hypothesis and one retraction. |
| **Results briefing** <br>*for a SLAM audience* | [open rendered ▸](https://raw.githack.com/neuromorphs/VSACognitiveMapping/astm/astm/docs/results_robotics_share.html) · [mirror](https://htmlpreview.github.io/?https://github.com/neuromorphs/VSACognitiveMapping/blob/astm/astm/docs/results_robotics_share.html) | Every measured result written for semantic-mapping readers. TL;DR box, plain-language explainers, 9 figures, and a positioned related-work review across five themes with 29 references. |
| **Provenance audit** | [read on GitHub ▸](provenance-audit.md) | What is ported from this repo's own `jepa`/`init-am` branches versus what was added here. Read before attributing anything. |

## Explainers

| Document | Open it | What it is |
|---|---|---|
| **Concept explainer** | [open rendered ▸](https://raw.githack.com/neuromorphs/VSACognitiveMapping/astm/astm/docs/demo_explainer.html) · [mirror](https://htmlpreview.github.io/?https://github.com/neuromorphs/VSACognitiveMapping/blob/astm/astm/docs/demo_explainer.html) | What each demo panel shows and how to read it — the screen-share companion. |
| **Code explainer** | [open rendered ▸](https://raw.githack.com/neuromorphs/VSACognitiveMapping/astm/astm/docs/code_explainer.html) · [mirror](https://htmlpreview.github.io/?https://github.com/neuromorphs/VSACognitiveMapping/blob/astm/astm/docs/code_explainer.html) | Every module, when it was added, which vector answers which question, and the λ-decay deep-dive. |
| **Working document v2** | [read on GitHub ▸](working-document-v2.md) | Objectives scored, architecture as built, original open questions with measured answers. Renders natively. |
| **Demo explainer (markdown)** | [read on GitHub ▸](demo-explainer.md) | The concept explainer in markdown, with the limitations ledger. |

`results_robotics.html` is the same briefing with figures as **relative links**
to `../figures/` — better after a clone, worse through a proxy. Prefer
`results_robotics_share.html` online.

---

## Read this before quoting a number

Three corrections that supersede older text in this branch:

1. **The whitening multiplier is encoder-dependent.** Measured at matched
   N = 2,429 on identical boxes: **19.9×** on YOLOv8n crops, **8.9×** on
   ResNet-50, **8.1×** on DINOv2 (growth-law metric χ). The published 365×/22×
   figures came from YOLOv8n's penultimate detection feature — the most
   anisotropic representation available. **Never quote a bare multiplier; quote
   it with its encoder.** What *is* encoder-independent is the floor: whitened
   χ spans only 9.1–11.0 across all four backbones, including a randomly
   initialised one. Whitening lands every encoder in the same place; only the
   distance travelled differs.
2. **`school_run1` poses are unusable.** Its LIO-SAM solution reports 23,124 m
   of path in 455 s, 209 jumps over 5 m, and a peak implied speed of 1,595 m/s
   (a Spot walks at ~1.6 m/s). Anything binding position on this sequence is
   void. Pose-*free* work on it — crop isotropy, effective rank, class
   retrieval, the encoder comparison — is unaffected and stands.
3. **Nothing is held out anywhere.** Every replayed frame is in the stored set,
   so recall figures measure faithful storage, not generalisation. Relatedly, a
   randomly initialised ResNet-50 reaches the *same* median recall as DINOv2
   (0.160 vs 0.156 m) and separates only in the tail — closed-set self-recall
   barely tests descriptor quality at all.

## Figures

All in [`../figures/`](../figures/), rendering natively on GitHub.

- `crosstalk_scaling_yolov8n_crops.png`, `..._resnet50_crops.png`,
  `..._dinov2_crops.png` — the same crosstalk metric on three backbones at
  matched N; this is the encoder-dependence above, drawn
- `crosstalk_scaling.png` — the original frame-level YOLO measurement
  (raw slope 0.97 vs whitened 0.86)
- `crossover_analysis.png` — the honest bytes/latency economics against an
  exact event table
- `two_loop_experiment.png` — revisit battery: bimodal occupancy, cross-loop
  control, loop-closure recall
- `semantic_query_chair_memory_object.png` — "where are the chairs?" over
  depth-derived object positions
- `working_memory_person.png` — decaying working memory tracking a person,
  then honestly forgetting
- `language_query.png` — free-text query localised through the memory's own
  algebra

## Reproducing the pages

The `.py` files in [`../`](..) and [`../tools/`](../tools) are **flat snapshots
for reading and diffing**, mirroring `README_MODULE.md`'s intent. They keep
their original `vsa_cognitive_mapping.*` imports, so they do **not** execute
from this layout — they are here so you can read exactly what produced each
figure. The commands below run in the working repository, where the package
sits at `vsa_cognitive_mapping/`:

```
python tools/export_video_page.py  --scene classroom     # -> video_<scene>.json
python tools/build_video_html.py   --scene classroom     # inlines it into the player
python -m vsa_cognitive_mapping.isotropy_ladder          # ladder + Timkey decomposition
python -m vsa_cognitive_mapping.encoder_comparison --encoders resnet50 dinov2
python -m vsa_cognitive_mapping.encoder_report           # the four-way table
python -m vsa_cognitive_mapping.room_diversity           # two-scene comparison
python -m vsa_cognitive_mapping.crosstalk_scaling --content crop_embeddings_dinov2.pt
```

Per-detection crop features come first — without them every "partial cue" is a
38-class COCO label:

```
python -m vsa_cognitive_mapping.classroom_pipeline embed-crops --stride 1 \
    --rgb-config school_run1_rgb_d455 --out-dir outputs/school_run1
```

`--out-dir` alone does **not** select a sequence; pass `--rgb-config` too or you
will silently re-crop the classroom.

The exporter runs a pose sanity check and refuses to present recall as a result
when the SLAM solution has diverged — that is what produces the banner on the
school-run page.
