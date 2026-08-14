"""Stage 1: verify everything the later stages need, and build the dataset
layout ConceptGraphs expects. RUN THIS BEFORE STAGE 2 — it catches every
known data-side failure while the fix is still one copy command.

Checks (per scene):
  data side   frame*.jpg + depth*.png counts, traj.txt row count matches,
              poses.csv present
  our side    outputs/replica_<scene>/object_points.json + detections_crops.csv
  GT side     data/replica/<scene'>_vmap/ info_semantic.json + semantic_class
              + depth renders (needed by stage 4's eval points)
  CG layout   their Replica dataloader wants <root>/<scene>/results/frame%06d.jpg
              — ours is flat, so this script builds cg_dataset/<scene>/results/
              as symlinks (falls back to copies if symlinks are unavailable).

    python 01_check_data.py --scene room0
Exit code 0 + "STAGE OK" means stages 2-5 have everything they need.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import sys

PKG = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(PKG)

ok = True


def check(cond, label, fix):
    global ok
    mark = "OK  " if cond else "FAIL"
    print(f"[{mark}] {label}")
    if not cond:
        print(f"       fix: {fix}")
        ok = False
    return cond


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="room0")
    ap.add_argument("--no-heal", action="store_true",
                    help="report missing GT instead of auto-fetching it")
    args = ap.parse_args()
    s = args.scene
    vs = re.sub(r"(\D)(\d+)$", r"\1_\2", s)
    sd = os.path.join(ROOT, "data", "replica", s)

    print(f"== stage 1 checks: {s} ==")
    rgb = sorted(glob.glob(os.path.join(sd, "frame*.jpg")))
    dep = sorted(glob.glob(os.path.join(sd, "depth*.png")))
    check(len(rgb) == 2000, f"RGB frames ({len(rgb)}/2000)",
          f"copy data/replica/{s}/ from Paul, or run "
          f"'python tools/prepare_replica.py --scene {s}' in the main repo")
    check(len(dep) == 2000, f"depth frames ({len(dep)}/2000)", "same as above")
    traj = os.path.join(sd, "traj.txt")
    n_traj = sum(1 for _ in open(traj)) if os.path.exists(traj) else 0
    check(n_traj == len(rgb) and n_traj > 0,
          f"traj.txt rows ({n_traj}) == frames", "same as above")
    check(os.path.exists(os.path.join(sd, "poses.csv")), "poses.csv",
          f"python tools/prepare_replica.py --scene {s} --skip-fetch")

    od = os.path.join(ROOT, "outputs", f"replica_{s}")
    check(os.path.exists(os.path.join(od, "object_points.json")),
          "our observation stream (object_points.json)",
          "python -m vsa_cognitive_mapping.object_grounding --dataset "
          f"vsa_cognitive_mapping/configs/replica_{s}.json --gt-json "
          f"outputs/replica_{s}/gt_instances.json --relocalize  (in main repo)")
    check(os.path.exists(os.path.join(od, "detections_crops.csv")),
          "our detections csv",
          "python -m vsa_cognitive_mapping.classroom_pipeline embed-crops "
          f"--dataset vsa_cognitive_mapping/configs/replica_{s}.json")

    gd = os.path.join(ROOT, "data", "replica", f"{vs}_vmap")
    info = os.path.join(gd, "info_semantic.json")

    def gt_counts():
        return (len(glob.glob(os.path.join(gd, "semantic_class_*.png"))),
                len(glob.glob(os.path.join(gd, "depth_*.png"))))

    n_sem, n_gdep = gt_counts()
    if (not os.path.exists(info) or n_sem <= 50) and not args.no_heal:
        # On Colab the renders live on the ephemeral VM disk while their
        # product (gt_instances.json) persists in Drive — so a fresh VM has
        # the product but not the source this check (and stage 4) needs.
        # Run the exact fix command this check would print (resumable,
        # ~0.4 GB range-pull).
        print("GT renders missing — fetching them now "
              "(tools/replica_gt_from_renders.py, resumable, ~0.4 GB)...",
              flush=True)
        import subprocess
        subprocess.run([sys.executable,
                        os.path.join(ROOT, "tools",
                                     "replica_gt_from_renders.py"),
                        "--scene", s], cwd=ROOT)
        n_sem, n_gdep = gt_counts()
    check(os.path.exists(info), "GT vocabulary (info_semantic.json)",
          f"python tools/replica_gt_from_renders.py --scene {s}  (main repo; "
          "range-pulls only the GT members)")
    check(n_sem > 50 and n_gdep > 50,
          f"GT renders (semantic {n_sem}, depth {n_gdep})", "same as above")

    # ---- build the layout their dataloader expects ------------------------
    # On Colab the repo lives on Drive-FUSE, where stat-ing 4000 layout
    # entries costs ~40 min and their dataloader would read every frame
    # through Drive during the run — build on the fast local VM disk there.
    if os.path.isdir("/content"):
        cg_base = "/content/cg_dataset"
        old = os.path.join(PKG, "cg_dataset")
        if os.path.isdir(old) and not os.path.islink(old):
            try:
                os.rename(old, old + "_drive_old")
                print(f"[note] moved slow Drive-side cg_dataset aside "
                      f"-> {old}_drive_old (delete it from drive.google.com "
                      "to reclaim space)")
            except OSError:
                pass
    else:
        cg_base = os.path.join(PKG, "cg_dataset")
    cg = os.path.join(cg_base, s, "results")
    os.makedirs(cg, exist_ok=True)
    made = 0
    if rgb and n_traj:
        # frame000000.jpg / depth000000.png naming, traj.txt beside results/
        def place(src, dst):
            if os.path.exists(dst):
                return 0
            try:
                os.symlink(os.path.abspath(src), dst)
            except (OSError, NotImplementedError):
                shutil.copy2(src, dst)
            return 1
        for p in rgb:
            i = int(re.search(r"(\d+)", os.path.basename(p)).group(1))
            made += place(p, os.path.join(cg, f"frame{i:06d}.jpg"))
        for p in dep:
            i = int(re.search(r"(\d+)", os.path.basename(p)).group(1))
            made += place(p, os.path.join(cg, f"depth{i:06d}.png"))
        t_dst = os.path.join(cg_base, s, "traj.txt")
        if not os.path.exists(t_dst):
            shutil.copy2(traj, t_dst)
        print(f"[OK  ] {cg_base}/{s}/results/ layout ready "
              f"({made} links created this run)")
        # their configs also want camera params; write the render intrinsics
        cam = os.path.join(cg_base, s, "cam_params.json")
        if not os.path.exists(cam):
            with open(cam, "w") as f:
                json.dump({"camera": {"w": 1200, "h": 680, "fx": 600.0,
                                      "fy": 600.0, "cx": 599.5, "cy": 339.5,
                                      "png_depth_scale": 6553.5}}, f, indent=2)
            print("[OK  ] cam_params.json written (NICE-SLAM render camera)")

    print()
    if ok:
        print("STAGE OK (01_check_data) — stages 2-5 have everything they need")
    else:
        print("STAGE FAILED — fix the FAIL lines above before stage 2")
        sys.exit(1)


if __name__ == "__main__":
    main()
