#!/usr/bin/env bash
# Stage 1a: collect EVERYTHING for one scene on THIS machine — no files from
# Paul needed. Fetches the rendered sequence + GT, then builds our
# observation stream (YOLO + depth placement). Usage:
#     bash 01_get_data.sh room0
# Repeat per scene. Safe to re-run (every step resumes/skips when done).
set -e
trap 'echo "STAGE FAILED (01_get_data) — network steps resume on re-run; send the last lines above if it keeps failing"' ERR
SCENE=${1:-room0}
eval "$(conda shell.bash hook)"; conda activate cgraphs
cd "$(dirname "$0")/.."          # repo root

pip -q install numpy pillow ultralytics 2>/dev/null || true

echo "== [1/4] RGB-D + poses ($SCENE, ~5 GB range-fetch, resumes) =="
test -f "data/replica/$SCENE/poses.csv" || python tools/prepare_replica.py --scene "$SCENE"

echo "== [2/4] GT semantic renders ($SCENE, ~0.4 GB) =="
VS=$(python -c "import re;print(re.sub(r'(\D)(\d+)$', r'\1_\2', '$SCENE'))")
test -f "data/replica/${VS}_vmap/info_semantic.json" || \
  python tools/replica_gt_from_renders.py --scene "$SCENE"

echo "== [3/4] our detections (YOLO over 2000 frames; fast on GPU) =="
test -f "outputs/replica_$SCENE/detections_crops.csv" || \
  python -m vsa_cognitive_mapping.classroom_pipeline embed-crops \
    --dataset "vsa_cognitive_mapping/configs/replica_$SCENE.json"

echo "== [4/4] world placement (depth + pose -> object_points.json) =="
test -f "outputs/replica_$SCENE/object_points.json" || \
  python -m vsa_cognitive_mapping.object_grounding \
    --dataset "vsa_cognitive_mapping/configs/replica_$SCENE.json" \
    --gt-json "outputs/replica_$SCENE/gt_instances.json" || true
test -f "outputs/replica_$SCENE/object_points.json" || {
  echo "FAIL: object_points.json was not produced — send the log above"; exit 1; }

cd student_gpu_package
python 01_check_data.py --scene "$SCENE"
echo "STAGE OK (01_get_data $SCENE)"
