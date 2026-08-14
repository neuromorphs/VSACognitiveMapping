# Running on a new dataset

Everything used to be wired to `lorinachey/spot-telluride-workshop-dataset`.
A sequence is now a **JSON config**, so trying new data is configuration rather
than code editing.

## 1. Write a config

Copy [`configs/TEMPLATE_new_dataset.json`](configs/TEMPLATE_new_dataset.json).
Two shapes:

**A folder of images** (poses optional):

```json
{
  "name": "my_room",
  "kind": "folder",
  "images": "data/my_room/rgb/*.png",
  "fps": 30,
  "out_dir": "outputs/my_room",
  "poses": {"kind": "tum", "path": "data/my_room/groundtruth.txt", "axes": "xy"}
}
```

**A HuggingFace dataset**:

```json
{
  "name": "some_run",
  "kind": "huggingface",
  "repo": "org/dataset-name",
  "rgb_config": "rgb",
  "odom_config": "odometry",
  "out_dir": "outputs/some_run"
}
```

Pose formats supported:

| `poses.kind` | What it reads |
|---|---|
| `tum` | `timestamp tx ty tz qx qy qz qw`, `#` comments ignored. Covers TUM RGB-D, ICL-NUIM (`*.gt.freiburg`), and most SLAM benchmarks. Use `axes` to pick the floor plane — `"xy"` for most robot data, `"xz"` for ICL-NUIM's living room. |
| `csv` | Any CSV; you name the columns (`time`, `x`, `y`, `yaw`, or `qx/qy/qz/qw`). Omit `time` if rows are already one-per-frame in order. |
| *(omitted)* | No poses. Still fully usable — see below. |

## 2. Validate it — always do this first

```bash
python -m vsa_cognitive_mapping.sequences validate --dataset configs/my_room.json
```

It decodes a few frames, checks the pose stream is physically plausible, and
prints which analyses your data supports. Example of it doing its job:

```
school_run1: 11211 frames, 455.0 s (24.6 fps)
  poses 11211, 455.0 s, path 21859.5 m, extent 1896.9 x 2398.8 m
  step median 1.1 cm, max 218.85 m, jumps>5 m 874, peak speed 1594.8 m/s
  verdict: UNUSABLE
    - 874 consecutive jumps over 5 m
    - peak implied speed 1594.8 m/s exceeds 5.0 m/s
```

That check exists because a diverged SLAM solution is **silent**: it produces
plausible-looking metres and poisons every pose-dependent result. It cost a day
before the check existed. Do not skip it.

## 3. You do not need poses for most of the work

This is worth knowing before you go hunting for a dataset with ground truth.

| Analysis | Needs poses? |
|---|---|
| Per-detection crop features (`embed-crops`) | no |
| Isotropy ladder, Timkey decomposition (`isotropy_ladder`) | no |
| Encoder comparison (`encoder_comparison`, `encoder_report`) | no |
| Class retrieval, scene diversity (`room_diversity`) | no |
| Crosstalk vs N (`crosstalk_scaling`) | **yes** |
| Place recall, memory build, video recall pages | **yes** |

The strongest results in this project so far — the isotropy/retrieval
dissociation, the rogue-dimension test, the encoder four-way — are all in the
first group. A pose-free image sequence is a perfectly good test set for them.

## 4. Run the pipeline

```bash
# per-detection appearance features (the input everything else needs)
python -m vsa_cognitive_mapping.classroom_pipeline embed-crops \
    --dataset configs/my_room.json --stride 1

# optional: the same crops through other backbones
python -m vsa_cognitive_mapping.encoder_comparison \
    --dataset configs/my_room.json --encoders resnet50 dinov2

# analyses (these read --out-dir, so they need no dataset config)
python -m vsa_cognitive_mapping.isotropy_ladder  --out-dir outputs/my_room
python -m vsa_cognitive_mapping.encoder_report   --out-dir outputs/my_room
python -m vsa_cognitive_mapping.room_diversity   --scenes outputs/my_room outputs/classroom

# pose-dependent, only if validate said YES
python -m vsa_cognitive_mapping.crosstalk_scaling --out-dir outputs/my_room \
    --content crop_embeddings_dinov2.pt --tag "my_room dinov2"

# the playable video page
python tools/export_video_page.py --scene my_room
python tools/build_video_html.py  --scene my_room
```

## 5. Two things to check on any new dataset

**Detections.** `embed-crops` prints how many boxes YOLO found and how many
were flagged tiny. A rendered or unusual-looking scene can produce very few
COCO detections, which makes every downstream "semantic" number thin. If the
crop count is small relative to frames, say so in any result.

**A chance baseline.** Every retrieval number needs one. Class retrieval's
chance is the sum of squared class frequencies; place recall's chance is the
median distance to a random eligible frame, and its ceiling is the distance to
the *nearest* eligible frame. Without both, "3.6 m error" is uninterpretable —
in a 6.9 m room, chance was 5.24 m and the ceiling 2.02 m.

## Worked examples

- [`configs/classroom_seq.json`](configs/classroom_seq.json) — HuggingFace, poses OK
- [`configs/school_run1_seq.json`](configs/school_run1_seq.json) — HuggingFace, poses **rejected**
- [`configs/spot_run1_local_seq.json`](configs/spot_run1_local_seq.json) — local JPEGs + CSV poses

The last two are the same physical walk read through different paths, and both
readers independently report a ~24 m path across a ~6.9 × 6.7 m room — a useful
cross-check that a new reader is wired up correctly.

## Benchmark recipe: 7-Scenes (cross-traverse protocol)

The one-command path to a literature-comparable relocalisation table
(memory = official train traverses, queries = a physically separate walk):

```bash
python tools/prepare_7scenes.py --scene chess     # download, strip to color+pose, convert, write configs
python -m vsa_cognitive_mapping.vpr_frontend --dataset vsa_cognitive_mapping/configs/7scenes_chess_seq01.json   # x6
python -m vsa_cognitive_mapping.cross_recall \
    --mem-datasets   vsa_cognitive_mapping/configs/7scenes_chess_seq0{1,2}.json \
                     vsa_cognitive_mapping/configs/7scenes_chess_seq0{4,6}.json \
    --query-datasets vsa_cognitive_mapping/configs/7scenes_chess_seq0{3,5}.json \
    --encoder crop_embeddings_eigenplaces.pt --merge-check
```

`cross_recall` differs from `heldout_eval` on purpose: no blocked splits
(held-out-ness comes from real separate traverses), treatment statistics are
fit on MEMORY frames only (no query leak), and it reports the retrieved
frame's full **3D** translation error alongside the 2D grid error — only the
3D number may sit next to published baselines (e.g. DenseVLAD 0.21 m median
on chess). Every row carries a bytes column. First measured result:
tracker entry (ax), `outputs/cross_recall_7scenes_chess.json`.
