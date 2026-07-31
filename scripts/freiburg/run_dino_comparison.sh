#!/usr/bin/env bash
# Freiburg (ICL-NUIM "living room traj0") counterpart of
# scripts/classroom/run_dino_comparison.sh: run the pipeline once with
# --embedding-model yolo and once with --embedding-model dino, writing both
# backends' outputs side by side into one directory for comparison (default:
# outputs/freiburg_dino_comparison).
#
#   scripts/freiburg/run_dino_comparison.sh
#   scripts/freiburg/run_dino_comparison.sh outputs/my_comparison_dir
#
# Steps: detect+embed (YOLO also writes detections.csv, shared by both
# backends) -> diagnostics (embedding correlation/uncertainty, reusing
# scripts/classroom's plot scripts -- they're fully embedding-source-
# agnostic, operating only on an embeddings.pt {frame_idx, embedding} dict)
# -> a --subset all memory build + evaluate + a point query -> a --subset
# stride memory build + the held-out "have I seen this before?" demo video.
#
# Unlike the classroom recording, this dataset has no stationary prefix/
# suffix (the ground-truth trajectory moves continuously from the first
# frame to the last), so neither memory build here passes --trim-stationary.
#
# QUERY_ARGS defaults to the trajectory's mean position (--query-x 0.0
# --query-y -1.5, in the ground-truth x-z floor plane -- see
# freiburg_associative_memory.py's docstring for why z, not y, is this
# dataset's second position axis); edit below to query a different position.
#
# HD_DIM matches scripts/classroom/run_dino_comparison.sh's value, for a
# like-for-like comparison across datasets.
#
# Every command below is idempotent (skips regenerating a file that already
# exists unless --force is passed), so this script is safe to re-run.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

OUT="${1:-outputs/freiburg_dino_comparison}"
HD_DIM=8192
QUERY_ARGS=(--query-x 0.0 --query-y -1.5)

echo "=== writing outputs to $OUT ==="

echo "=== 1. detect + embed (yolo, dino) ==="
python scripts/freiburg/detect_and_embed_freiburg.py --out-dir "$OUT" --embedding-model yolo
python scripts/freiburg/detect_and_embed_freiburg.py --out-dir "$OUT" --embedding-model dino

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
  python scripts/freiburg/freiburg_associative_memory.py build --embedding-model "$MODEL" --subset all \
    --embeddings "$OUT/embeddings${SUFFIX}.pt" --hd-dim "$HD_DIM" --pca-whiten \
    --out "$OUT/associative_memory_all${SUFFIX}.pt"

  python scripts/freiburg/freiburg_associative_memory.py evaluate --embedding-model "$MODEL" \
    --memory "$OUT/associative_memory_all${SUFFIX}.pt" --out-path "$OUT/evaluate_result${SUFFIX}.png"

  python scripts/freiburg/freiburg_associative_memory.py query --embedding-model "$MODEL" \
    --memory "$OUT/associative_memory_all${SUFFIX}.pt" --detections "$OUT/detections.csv" \
    "${QUERY_ARGS[@]}" --out-path "$OUT/query_result${SUFFIX}.png"
done

echo "=== 4. build (--subset stride) + held-out demo video ==="
for MODEL in yolo dino; do
  SUFFIX=""; [ "$MODEL" = "dino" ] && SUFFIX="_dino"
  python scripts/freiburg/freiburg_associative_memory.py build --embedding-model "$MODEL" \
    --subset stride --stride 3 --hd-dim "$HD_DIM" --pca-whiten \
    --embeddings "$OUT/embeddings${SUFFIX}.pt" --out "$OUT/associative_memory_stride${SUFFIX}.pt"

  python scripts/freiburg/freiburg_associative_memory.py demo --embedding-model "$MODEL" \
    --memory "$OUT/associative_memory_stride${SUFFIX}.pt" --embeddings "$OUT/embeddings${SUFFIX}.pt" \
    --detections "$OUT/detections.csv" --out-path "$OUT/demo${SUFFIX}.mp4"
done

echo "=== done: yolo and dino outputs written side by side under $OUT ==="
