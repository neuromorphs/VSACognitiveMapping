"""One-command pipeline: dataset config -> embed -> build -> live demo.

Functionalises the whole demo so a NEW recording is just a config file:

    python vsa_cognitive_mapping/run_demo_pipeline.py --config configs/classroom.json
    python vsa_cognitive_mapping/run_demo_pipeline.py --config configs/school_run1.json
    python vsa_cognitive_mapping/run_demo_pipeline.py --config configs/school_run1.json --steps embed,build
    python vsa_cognitive_mapping/run_demo_pipeline.py --config ... --force   # redo existing steps

Stages (each skipped when its output already exists, unless --force):
    embed   workshop detect_and_embed_classroom.py (YOLOv8n) -> detections.csv + embeddings.pt
    build   workshop classroom_associative_memory.py build   -> associative_memory_stride.pt
    serve   workshop_demo.py on the built artifacts (foreground)

Config JSON (see vsa_cognitive_mapping/configs/*.json):
    {
      "name": "school_run1",                     // labels the run
      "repo": "lorinachey/spot-telluride-workshop-dataset",
      "rgb_config": "school_run1_rgb_d455",      // any HF config with frame_idx/timestamp_ns/image
      "odom_config": "school_run1_odometry_lio_sam",
      "out_dir": "outputs/school_run1",          // all artifacts land here (repo-relative)
      "embed": {"device": "cpu", "batch_size": 16, "limit": null},
      "build": {"subset": "stride", "stride": 3, "hd_dim": 8192, "pca_whiten": true},
      "serve": {"port": 8020, "grid_resolution": 56}
    }

New sensors entirely (not on HF): produce the same two artifacts yourself —
`embeddings.pt` = {"frame_idx": LongTensor[N], "timestamp_ns": LongTensor[N],
"embedding": FloatTensor[N, D]} in timestamp order covering every frame, plus
`detections.csv` (frame_idx,timestamp_ns,class_id,class_name,confidence,
x1,y1,x2,y2) — then run build+serve against any HF-style rgb/odom source
(or adapt load_frame_data). The VSA stages never care where features came from.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKSHOP = os.path.join(REPO_ROOT, "external", "VSACognitiveMapping-init-am")


def sh(cmd, cwd, env_extra=None):
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    print("\n$ " + " ".join(cmd) + (f"   (cwd={cwd})" if cwd else ""))
    r = subprocess.run(cmd, cwd=cwd, env=env)
    if r.returncode != 0:
        raise SystemExit(f"step failed with exit code {r.returncode}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", default="embed,build,serve",
                    help="comma list from: embed,build,serve")
    ap.add_argument("--force", action="store_true", help="redo steps whose outputs exist")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    out_dir = os.path.join(REPO_ROOT, cfg["out_dir"])
    os.makedirs(out_dir, exist_ok=True)
    detections = os.path.join(out_dir, "detections.csv")
    embeddings = os.path.join(out_dir, "embeddings.pt")
    memory = os.path.join(out_dir, f"associative_memory_{cfg['build'].get('subset', 'stride')}.pt")
    ws_env = {"PYTHONPATH": os.path.join(WORKSHOP, "src")}
    py = sys.executable

    print(f"run '{cfg['name']}': repo={cfg['repo']} rgb={cfg['rgb_config']} "
          f"odom={cfg['odom_config']} -> {out_dir}")

    if "embed" in steps:
        if os.path.exists(embeddings) and not args.force:
            print(f"[embed] SKIP - {embeddings} exists (use --force to redo)")
        else:
            e = cfg.get("embed", {})
            cmd = [py, os.path.join(WORKSHOP, "scripts", "classroom",
                                    "detect_and_embed_classroom.py"),
                   "--repo", cfg["repo"], "--d455-rgb-config", cfg["rgb_config"],
                   "--out-dir", out_dir,
                   "--device", e.get("device", "cpu"),
                   "--batch-size", str(e.get("batch_size", 16))]
            if e.get("limit"):
                cmd += ["--limit", str(e["limit"])]
            sh(cmd, cwd=WORKSHOP, env_extra=ws_env)

    if "build" in steps:
        if os.path.exists(memory) and not args.force:
            print(f"[build] SKIP - {memory} exists (use --force to redo)")
        else:
            b = cfg.get("build", {})
            cmd = [py, os.path.join(WORKSHOP, "scripts", "classroom",
                                    "classroom_associative_memory.py"), "build",
                   "--repo", cfg["repo"], "--d455-rgb-config", cfg["rgb_config"],
                   "--odom-config", cfg["odom_config"],
                   "--embeddings", embeddings, "--out", memory,
                   "--subset", b.get("subset", "stride"),
                   "--stride", str(b.get("stride", 3)),
                   "--hd-dim", str(b.get("hd_dim", 8192))]
            if b.get("pca_whiten", True):
                cmd.append("--pca-whiten")
            sh(cmd, cwd=WORKSHOP, env_extra=ws_env)

    if "serve" in steps:
        s = cfg.get("serve", {})
        cmd = [py, os.path.join(REPO_ROOT, "vsa_cognitive_mapping", "workshop_demo.py"),
               "--memory", memory, "--embeddings", embeddings,
               "--detections", detections,
               "--repo", cfg["repo"], "--rgb-config", cfg["rgb_config"],
               "--odom-config", cfg["odom_config"],
               "--port", str(s.get("port", 8020)),
               "--grid-resolution", str(s.get("grid_resolution", 56))]
        sh(cmd, cwd=REPO_ROOT)


if __name__ == "__main__":
    main()
