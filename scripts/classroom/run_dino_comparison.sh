#!/usr/bin/env bash
# Run the classroom pipeline once with --embedding-model yolo and once with
# --embedding-model dino, writing both backends' outputs side by side into
# one directory for comparison (default: outputs/classroom_dino_comparison).
#
#   scripts/classroom/run_dino_comparison.sh
#   scripts/classroom/run_dino_comparison.sh outputs/my_comparison_dir
#
# Steps (see docs/classroom_associative_memory_math.md for the full pipeline
# writeup): detect+embed (YOLO also writes detections.csv, shared by both
# backends since detection is embedding-model-independent) -> diagnostics
# (embedding correlation/uncertainty) -> a --subset all memory build +
# evaluate + a point query -> a --subset stride memory build + the
# held-out "have I seen this before?" demo video.
#
# Every command below is idempotent (skips regenerating a file that already
# exists unless --force is passed), so this script is safe to re-run --
# e.g. after it fails partway through, or to add --visualize video renders
# on a later pass without redoing detection/embedding.
#
# Query defaults to --query-x -2.0 --query-y 3.0 in step 3; edit QUERY_ARGS
# below to query a different position, or use --query-yaw/--query-time
# instead (see classroom_associative_memory.py query --help).
#
# HD_DIM sets the phasor/VSA dimensionality used everywhere it applies
# (diagnostics' projection in step 2, and the associative memory builds in
# steps 3 and 4) -- one consistent value end to end, rather than each
# script's own default (256 for the diagnostics scripts, 2048 for build).
#
# Both memory builds (steps 3 and 4) also PCA-whiten the raw embeddings
# before projecting to phasor content (--pca-whiten): on this dataset, raw
# embedding cosine similarity sits in a tight 0.70-1.00 band, and whitening
# spreads out the discriminative structure that band otherwise swamps --
# see docs/classroom_associative_memory_math.md §3/§8's worked example.
#
# Both memory builds also pass --trim-stationary, excluding a stationary
# prefix/suffix (the robot hasn't moved yet, or not anymore) from whichever
# --subset gets bundled -- these are otherwise near-duplicate frames that
# waste bundling capacity. Detected automatically from odometry displacement
# (see classroom_associative_memory.py build --help for the threshold
# knobs); tune TRIM_DISTANCE_THRESHOLD below if it over/under-trims for a
# different recording.
#
# Step 4's demo also passes --kalman-filter, opting into the temporal
# pseudo-Kalman position filter (blends the previous filtered position,
# bound with the ground-truth odometry delta since then, against this
# frame's raw VSA recall by a fixed weight) -- adds a third marker to the
# position heatmap panel showing the smoothed estimate next to the
# existing raw per-frame one. Tune KALMAN_WEIGHT below if it over/under-
# smooths for a different recording.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

OUT="${1:-outputs/classroom_dino_comparison}"
HD_DIM=8192
TRIM_DISTANCE_THRESHOLD=0.1
QUERY_ARGS=(--query-x -2.0 --query-y 3.0)
KALMAN_WEIGHT=0.5

echo "=== writing outputs to $OUT ==="

echo "=== 1. detect + embed (yolo, dino) ==="
python scripts/classroom/detect_and_embed_classroom.py --out-dir "$OUT" --embedding-model yolo
python scripts/classroom/detect_and_embed_classroom.py --out-dir "$OUT" --embedding-model dino

echo "=== 2. diagnostics (embedding correlation, embedding uncertainty) ==="
for MODEL in yolo dino; do
  SUFFIX=""; [ "$MODEL" = "dino" ] && SUFFIX="_dino"
  python scripts/classroom/plot_embedding_correlation.py --embedding-model "$MODEL" \
    --embeddings "$OUT/embeddings${SUFFIX}.pt" --hd-dim "$HD_DIM" \
    --out-path "$OUT/embedding_correlation${SUFFIX}.png" \
    --phasor-out-path "$OUT/embedding_phasor_fidelity${SUFFIX}.png"

  python scripts/classroom/plot_embedding_uncertainty.py --embedding-model "$MODEL" \
    --embeddings "$OUT/embeddings${SUFFIX}.pt" --hd-dim "$HD_DIM" \
    --out-path "$OUT/embedding_uncertainty${SUFFIX}.png" \
    --phasor-out-path "$OUT/embedding_uncertainty_phasor${SUFFIX}.png"
done

echo "=== 3. build (--subset all) + evaluate + point query ==="
for MODEL in yolo dino; do
  SUFFIX=""; [ "$MODEL" = "dino" ] && SUFFIX="_dino"
  python scripts/classroom/classroom_associative_memory.py build --embedding-model "$MODEL" --subset all \
    --embeddings "$OUT/embeddings${SUFFIX}.pt" --hd-dim "$HD_DIM" --pca-whiten \
    --trim-stationary --trim-distance-threshold "$TRIM_DISTANCE_THRESHOLD" \
    --out "$OUT/associative_memory_all${SUFFIX}.pt"

  python scripts/classroom/classroom_associative_memory.py evaluate --embedding-model "$MODEL" \
    --memory "$OUT/associative_memory_all${SUFFIX}.pt" --out-path "$OUT/evaluate_result${SUFFIX}.png"

  python scripts/classroom/classroom_associative_memory.py query --embedding-model "$MODEL" \
    --memory "$OUT/associative_memory_all${SUFFIX}.pt" --detections "$OUT/detections.csv" \
    "${QUERY_ARGS[@]}" --out-path "$OUT/query_result${SUFFIX}.png"
done

echo "=== 4. build (--subset stride) + held-out demo video ==="
for MODEL in yolo dino; do
  SUFFIX=""; [ "$MODEL" = "dino" ] && SUFFIX="_dino"
  python scripts/classroom/classroom_associative_memory.py build --embedding-model "$MODEL" \
    --subset stride --stride 3 --hd-dim "$HD_DIM" --pca-whiten \
    --trim-stationary --trim-distance-threshold "$TRIM_DISTANCE_THRESHOLD" \
    --embeddings "$OUT/embeddings${SUFFIX}.pt" --out "$OUT/associative_memory_stride${SUFFIX}.pt"

  python scripts/classroom/classroom_associative_memory.py demo --embedding-model "$MODEL" \
    --memory "$OUT/associative_memory_stride${SUFFIX}.pt" --embeddings "$OUT/embeddings${SUFFIX}.pt" \
    --detections "$OUT/detections.csv" --kalman-filter --kalman-weight "$KALMAN_WEIGHT" \
    --out-path "$OUT/demo${SUFFIX}.mp4"
done

echo "=== done: yolo and dino outputs written side by side under $OUT ==="
