#!/usr/bin/env bash
# Stage 0: environment for the ConceptGraphs head-to-head. Run once.
# Idempotent: safe to re-run after a partial failure.
set -e
trap 'echo "STAGE FAILED (00_setup) — see README Troubleshooting, item matching the last line above"' ERR

command -v conda >/dev/null || {
  echo "FAIL: conda not on PATH. Install miniconda, or run:  source ~/miniconda3/etc/profile.d/conda.sh"; exit 1; }

echo "== creating conda env 'cgraphs' (skipped if it exists) =="
conda env list | grep -q "^cgraphs " || conda create -y -n cgraphs python=3.10
eval "$(conda shell.bash hook)"
conda activate cgraphs

echo "== cloning official ConceptGraphs =="
if [ ! -d concept-graphs ]; then
  git clone https://github.com/concept-graphs/concept-graphs.git
fi
cd concept-graphs
# Their README's install section is authoritative; this covers the usual path.
pip install -e . || echo "WARN: editable install failed — follow THEIR README install section, then re-run this script (it will skip done steps)"

# CUDA wheel must match the driver. Pick automatically, fall back to CPU.
if command -v nvidia-smi >/dev/null; then
  CUDA_MAJ=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+" | head -1 || echo 12)
  if [ "$CUDA_MAJ" -ge 12 ]; then IDX=cu121; else IDX=cu118; fi
  echo "== torch for CUDA $CUDA_MAJ ($IDX) =="
  pip install torch torchvision --index-url https://download.pytorch.org/whl/$IDX
else
  echo "WARN: no GPU visible — installing CPU torch (stage 2 will be very slow)"
  pip install torch torchvision
fi
pip install open_clip_torch supervision open3d scipy

# ---- VRAM-aware segmentation backbone choice ------------------------------
# ConceptGraphs supports light variants (MobileSAM / FastSAM) officially.
# Pick from measured VRAM; record the choice for stage 2 and the report.
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0)
if [ "$VRAM_MB" -ge 10000 ]; then GSA_CHOICE="sam_vit_h"
elif [ "$VRAM_MB" -ge 4000 ]; then GSA_CHOICE="mobilesam"
else GSA_CHOICE="mobilesam"; echo "WARN: <4 GB VRAM — stage 2 may still OOM; see README VRAM tiers"; fi
echo "VRAM ${VRAM_MB} MB -> segmentation backbone: $GSA_CHOICE"
echo "$GSA_CHOICE" > ../gsa_choice.txt

dl(){ [ -f "$2" ] || { if command -v wget >/dev/null; then wget -O "$2" "$1"; else curl -L -o "$2" "$1"; fi; }; }
mkdir -p checkpoints
# MobileSAM is tiny — always fetch (it is also the fallback if vit_h OOMs)
dl https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt \
   checkpoints/mobile_sam.pt
if [ "$GSA_CHOICE" = "sam_vit_h" ]; then
  dl https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth \
     checkpoints/sam_vit_h_4b8939.pth
fi
# Their SEMANTIC pipeline (Grounded-SAM) also wants these two; big on disk
# (~6 GB), modest in VRAM. Skip with SKIP_GROUNDED=1 if disk is tight — the
# class-agnostic mode (gsa_variant=none) then still works.
if [ "${SKIP_GROUNDED:-0}" != "1" ]; then
  dl https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth \
     checkpoints/groundingdino_swint_ogc.pth
  dl https://huggingface.co/spaces/xinyu1205/recognize-anything/resolve/main/ram_swin_large_14m.pth \
     checkpoints/ram_swin_large_14m.pth
fi
cd ..

pip install ultralytics transformers pillow numpy   # for stage 4 (our side)

echo "== recording environment =="
{ python -V; pip freeze; nvidia-smi -L 2>/dev/null || echo "no GPU visible"; } \
  > environment.txt
echo "STAGE OK (00_setup)"
