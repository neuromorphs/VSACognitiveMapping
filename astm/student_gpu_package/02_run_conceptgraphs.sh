#!/usr/bin/env bash
# Stage 2: run the OFFICIAL ConceptGraphs pipeline on one Replica scene.
# Usage: bash 02_run_conceptgraphs.sh room0
# Do not tune their configs; the point is the comparison, not the score.
set -e
SCENE=${1:-room0}

# ---- the only two lines you may need to edit ------------------------------
CG_REPO="$(pwd)/concept-graphs"
CG_DATA="$(pwd)/cg_dataset"      # built by 01_check_data.py (their layout)
# ---------------------------------------------------------------------------

eval "$(conda shell.bash hook)"; conda activate cgraphs
OUT="$(pwd)/cg_out/$SCENE"; mkdir -p "$OUT"

echo "== ConceptGraphs on $SCENE =="
test -d "$CG_DATA/$SCENE/results" || {
  echo "FAIL: $CG_DATA/$SCENE/results missing — run: python 01_check_data.py --scene $SCENE"; exit 1; }

# Their entry points have moved between releases. Try the known candidates in
# order; the first that exists as a module is used. If ALL fail, follow their
# README 'Usage' section and put the working module name FIRST in this list.
CANDIDATES=(
  "conceptgraph.slam.cfslam_pipeline_batch"
  "conceptgraph.slam.rerun_realtime_mapping"
  "conceptgraph.scripts.run_slam"
)
cd "$CG_REPO"
ENTRY=""
for m in "${CANDIDATES[@]}"; do
  python -c "import importlib,sys; importlib.import_module('$m')" 2>/dev/null \
    && { ENTRY="$m"; break; }
done
[ -n "$ENTRY" ] || {
  echo "---------------------------------------------------------------"
  echo "FAIL: none of the candidate entry points import. Their layout:"
  find conceptgraph -name "*.py" | xargs grep -l "__main__" 2>/dev/null | head -20
  echo "Open their README 'Usage', find the Replica run command, add its"
  echo "module name to CANDIDATES at the top of this script, re-run."
  exit 1
}
echo "entry point: $ENTRY"

# VRAM-appropriate backbone chosen by 00_setup (sam_vit_h >=10 GB VRAM,
# mobilesam otherwise — an OFFICIALLY supported ConceptGraphs variant).
GSA=$(cat ../gsa_choice.txt 2>/dev/null || echo sam_vit_h)
echo "segmentation backbone: $GSA  (from gsa_choice.txt)"
EXTRA=""
[ "$GSA" = "mobilesam" ] && EXTRA="sam_variant=mobilesam"

SECONDS=0
python -m "$ENTRY" \
    dataset_root="$CG_DATA" scene_id="$SCENE" $EXTRA \
    save_dir="$OUT" 2>&1 | tee "$OUT/run.log" || {
  echo "---------------------------------------------------------------"
  echo "The run itself failed. Last 30 log lines:"
  tail -30 "$OUT/run.log"
  echo "Common causes + fixes are in README 'Troubleshooting' (CUDA OOM ->"
  echo "SAM vit_b; hydra config name -> their README's exact CLI; missing"
  echo "cam params -> cg_dataset/$SCENE/cam_params.json exists, point their"
  echo "dataset config at it)."
  exit 1
}
echo "wall-clock: ${SECONDS}s  (record this — it goes in the systems table)"
ls -lh "$OUT"
echo "STAGE OK (02_run_conceptgraphs $SCENE)"
