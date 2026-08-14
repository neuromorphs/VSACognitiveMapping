"""B1 — open-vocabulary frontend swap.

Replaces the YOLOv8n-COCO detector in vsa_cognitive_mapping/classroom_pipeline.py's
`embed-crops` stage with an open-vocabulary detector (YOLO-World) prompted
with Replica's OWN class list, and CLIP crop embeddings on top of it.

Writes to a brand-new output directory (outputs/replica_<scene>_openvocab/)
and a brand-new sequence config (vsa_cognitive_mapping/configs/replica_<scene>_openvocab.json)
so the existing YOLO-COCO baseline artifacts (already measured: 0.091 mAcc,
cited in the paper draft) are never touched, read, or overwritten.

Requires (on top of this repo's existing requirements):
    pip install "ultralytics>=8.1"   # YOLO-World / model.set_classes()
    pip install transformers          # CLIP crop embedding (already a dep
                                       # if you've run the main pipeline)

Usage — one scene:
    python collab_tasks/scripts/embed_crops_openvocab.py --scene room0

All 8 scenes:
    python collab_tasks/scripts/embed_crops_openvocab.py --scene room0 room1 room2 office0 office1 office2 office3 office4

Prints the exact follow-on commands to score the result — see
collab_tasks/B1_open_vocab_frontend.md for the full walkthrough and the
self-check to run before sending anything back.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from vsa_cognitive_mapping.sequences import load_sequence  # noqa: E402


def replica_vocab(scene):
    """Read the scored Replica class list straight from this scene's GT
    instances — the same file object_grounding.py / 05_score.py score
    against — so the detector is prompted with exactly what will be judged,
    not a list assembled by hand."""
    path = f"outputs/replica_{scene}/gt_instances.json"
    if not os.path.exists(path):
        raise SystemExit(f"missing {path} — run, from the main repo:\n"
                         f"  python tools/replica_gt_from_renders.py --scene {scene}")
    inst = json.load(open(path))["instances"]
    classes = sorted({i["cls"] for i in inst
                      if not str(i["cls"]).startswith("class_-1")})
    if not classes:
        raise SystemExit(f"{path} has no usable classes")
    return classes


def make_openvocab_config(scene):
    """Clone this scene's sequence config with a distinct name/out_dir,
    pointing at the SAME raw frames and poses (the physical scene and its
    ground truth are unchanged; only the detector differs)."""
    src = f"vsa_cognitive_mapping/configs/replica_{scene}.json"
    if not os.path.exists(src):
        raise SystemExit(f"missing {src} — has this scene been fetched? "
                         f"(tools/prepare_replica.py --scene {scene})")
    cfg = json.load(open(src))
    cfg["name"] = f"{cfg['name']}_openvocab"
    cfg["out_dir"] = f"{cfg['out_dir']}_openvocab"
    dst = f"vsa_cognitive_mapping/configs/replica_{scene}_openvocab.json"
    json.dump(cfg, open(dst, "w"), indent=2)
    return dst, cfg["out_dir"]


def load_detector(classes):
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit('pip install "ultralytics>=8.1"  (needed for '
                         "YOLO-World / model.set_classes)")
    model = YOLO("yolov8s-worldv2.pt")           # open-vocab checkpoint, downloads once
    if not hasattr(model, "set_classes"):
        raise SystemExit("this ultralytics version has no model.set_classes "
                         "— pip install -U ultralytics")
    model.set_classes(classes)
    return model


def load_clip():
    from transformers import CLIPModel, CLIPProcessor

    from vsa_cognitive_mapping.clip_compat import image_features
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().to(dev)
    print(f"CLIP crop embedding on {dev}")
    return model, proc, dev, image_features


def run_scene(scene, pad_frac=0.08, min_side=16.0, conf=0.15,
             crops_name="crop_embeddings_openvocab.pt"):
    classes = replica_vocab(scene)
    print(f"\n=========== {scene} ===========")
    print(f"prompting detector with {len(classes)} Replica classes: "
          f"{', '.join(classes[:8])}{', ...' if len(classes) > 8 else ''}")

    cfg_path, out_dir = make_openvocab_config(scene)
    os.makedirs(out_dir, exist_ok=True)
    print(f"wrote {cfg_path}  ->  outputs go to {out_dir}/ "
         f"(the YOLO-COCO baseline directory is not touched)")

    model = load_detector(classes)
    clip_model, clip_proc, dev, image_features = load_clip()

    seq = load_sequence(cfg_path)
    print(f"{seq.name}: {len(seq)} frames")

    rows, crop_embs = [], []
    t0 = time.time()
    frame_ids = seq.frame_ids()
    ts = (seq.timestamps_s() * 1e9).astype(np.int64)
    for i in range(len(seq)):
        img = seq.image(i)
        W, H = img.size
        res = model.predict(img, conf=conf, verbose=False)[0]
        crops, meta = [], []
        for b in res.boxes:
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
            bw, bh = x2 - x1, y2 - y1
            pad = pad_frac * max(bw, bh)
            cx1 = max(0, int(round(x1 - pad)))
            cy1 = max(0, int(round(y1 - pad)))
            cx2 = min(W, int(round(x2 + pad)))
            cy2 = min(H, int(round(y2 + pad)))
            if cx2 - cx1 < 2 or cy2 - cy1 < 2:
                continue
            tiny = min(bw, bh) < min_side
            cid = int(b.cls)
            cls_name = classes[cid] if 0 <= cid < len(classes) else str(model.names.get(cid, cid))
            crops.append(img.crop((cx1, cy1, cx2, cy2)))
            meta.append([int(frame_ids[i]), int(ts[i]), cid, cls_name,
                        float(b.conf), x1, y1, x2, y2,
                        bw * bh / float(W * H), int(tiny)])
        if crops:
            with torch.no_grad():
                inputs = clip_proc(images=crops, return_tensors="pt").to(dev)
                emb = image_features(clip_model, inputs).cpu().float()
            crop_embs.extend(list(emb))
            rows.extend(meta)
        if i % 25 == 0:
            print(f"  frame {i}/{len(seq)}  crops_so_far={len(rows)}", flush=True)

    if not rows:
        raise SystemExit(f"{scene}: no detections produced any usable crop "
                         "— try a lower --conf, or check the class list "
                         "printed above isn't empty/garbled")

    C = torch.stack(crop_embs)
    out = {"det_id": torch.arange(len(rows)),
           "frame_idx": torch.tensor([r[0] for r in rows]),
           "timestamp_ns": torch.tensor([r[1] for r in rows]),
           "class_id": torch.tensor([r[2] for r in rows]),
           "confidence": torch.tensor([r[4] for r in rows], dtype=torch.float32),
           "area_frac": torch.tensor([r[9] for r in rows], dtype=torch.float32),
           "tiny": torch.tensor([r[10] for r in rows]),
           "embedding": C,
           "meta": {"model": "yolov8s-worldv2 (open-vocab)", "conf": conf,
                    "vocab_size": len(classes)}}
    torch.save(out, os.path.join(out_dir, crops_name))

    with open(os.path.join(out_dir, "detections_crops.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["det_id", "frame_idx", "timestamp_ns", "class_id",
                    "class_name", "confidence", "x1", "y1", "x2", "y2",
                    "area_frac", "tiny"])
        for i, r in enumerate(rows):
            w.writerow([i] + r)

    dt = (time.time() - t0) / 60
    print(f"{scene}: {len(rows)} crops from {len(seq)} frames in {dt:.1f} min "
         f"-> {out_dir}/detections_crops.csv + {crops_name}")
    return out_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", nargs="+", required=True,
                    help="one or more scene ids, e.g. room0 room1 office0")
    ap.add_argument("--conf", type=float, default=0.15,
                    help="detector confidence threshold (open-vocab models "
                         "are typically noisier than COCO-trained YOLO — "
                         "lower than the default 0.25 to start")
    args = ap.parse_args()

    for scene in args.scene:
        out_dir = run_scene(scene, conf=args.conf)

    print("\nSTAGE OK — next, for EACH scene run:\n")
    for scene in args.scene:
        print(f"# --- {scene} ---")
        print(f"python -m vsa_cognitive_mapping.object_grounding \\")
        print(f"    --dataset vsa_cognitive_mapping/configs/replica_{scene}_openvocab.json \\")
        print(f"    --gt-json outputs/replica_{scene}/gt_instances.json --relocalize")
        print(f"python student_gpu_package/04_vsa_labels.py "
             f"--scene {scene}_openvocab --gt-scene {scene}")
        print(f"python student_gpu_package/05_score.py --scene {scene}_openvocab")
        print()
    print("The last command's printed 'their protocol (n_exclude 6)' line "
         "is the number to compare against the 0.091 baseline (--scene "
         "<scene>, no suffix) and ConceptGraphs' published 0.406.")


if __name__ == "__main__":
    main()
