"""Stage 4: our VSA memory, made to speak ConceptGraphs' eval language.

Produces per-point class labels from the 32 KB trace so THEIR scorer can
judge OUR system:
  1. loads our observation stream (object_points.json — the same YOLO+depth
     detections Paul's runs use);
  2. re-labels every detection crop with CLIP against the REPLICA class
     list (their vocabulary + prompt style — closes the COCO vocab gap);
  3. builds the trace (class ⊗ position, bounded per-class insertion);
  4. builds handoff/<scene>/eval_points.npz — GT-labelled points
     backprojected from the vMAP semantic renders (the canonical eval set
     for BOTH systems) — unless stage 3 already produced one;
  5. labels every eval point: argmax over class fields at the point's
     floor coordinates -> handoff/<scene>/vsa_labels.npz.

Runs on CPU (~30-60 min, CLIP pass dominates; GPU cuts it to minutes).
Run from the MAIN repo root (needs vsa_cognitive_mapping/ + data/replica/):

    python student_gpu_package/04_vsa_labels.py --scene room0
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import zipfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from vsa_cognitive_mapping.classroom_pipeline import ClassroomEncoders  # noqa: E402
from vsa_cognitive_mapping.object_grounding import (  # noqa: E402
    class_phasors, build_trace, cap_per_class)
from vsa_cognitive_mapping.sequences import load_sequence  # noqa: E402

HD, LS = 4096, 0.6


def replica_class_list(scene):
    """Class names from the scene's info_semantic.json (their vocabulary)."""
    vscene = re.sub(r"(\D)(\d+)$", r"\1_\2", scene)
    for cand in (f"data/replica/{vscene}_vmap", f"data/replica/{scene}_vmap"):
        p = os.path.join(cand, "info_semantic.json")
        if os.path.exists(p):
            meta = json.load(open(p))
            return sorted({c["name"] for c in meta.get("classes", [])}), meta
    raise SystemExit("FAIL: info_semantic.json not found — run "
                     "tools/replica_gt_from_renders.py --scene "
                     f"{scene} first (it extracts it)")


def clip_relabel(scene, classes, batch=32):
    """CLIP labels for every detection crop against the Replica class list.
    Cached; identical prompt template to ConceptGraphs ('an image of a X')."""
    import torch
    from transformers import CLIPModel, CLIPProcessor

    from vsa_cognitive_mapping.clip_compat import image_features, text_features

    out_dir = f"outputs/replica_{scene}"
    cache = os.path.join(out_dir, "crop_clip_replica.pt")
    if os.path.exists(cache):
        return torch.load(cache, weights_only=False)
    rows = list(csv.DictReader(open(os.path.join(out_dir,
                                                 "detections_crops.csv"))))
    seq = load_sequence(f"vsa_cognitive_mapping/configs/replica_{scene}.json")
    pos = {int(f): i for i, f in enumerate(seq.frame_ids())}
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().to(dev)
    with torch.no_grad():
        t = proc(text=[f"an image of a {c}" for c in classes],
                 return_tensors="pt", padding=True).to(dev)
        T = text_features(model, t)
        T = T / T.norm(dim=-1, keepdim=True)
    lab = {}
    with torch.no_grad():
        for s in range(0, len(rows), batch):
            chunk = rows[s:s + batch]
            imgs = []
            for r in chunk:
                img = seq.image(pos[int(r["frame_idx"])])
                imgs.append(img.crop((max(0, float(r["x1"])),
                                      max(0, float(r["y1"])),
                                      min(img.size[0], float(r["x2"])),
                                      min(img.size[1], float(r["y2"])))))
            v = proc(images=imgs, return_tensors="pt").to(dev)
            V = image_features(model, v)
            V = V / V.norm(dim=-1, keepdim=True)
            best = (V @ T.T).argmax(1).cpu().numpy()
            for r, b in zip(chunk, best):
                lab[int(r["det_id"])] = classes[int(b)]
            if (s // batch) % 25 == 0:
                print(f"  CLIP {s}/{len(rows)}", flush=True)
    torch.save(lab, cache)
    return lab


def build_eval_points(scene, n_pts=200_000, stride=20, gt_scene=None):
    """GT-labelled eval points backprojected from the vMAP semantic renders —
    the canonical point set both systems are scored on. `gt_scene` (if the
    run is scored against a different physical scene's ground truth than its
    own name implies, e.g. an alternate-frontend run) is used for the vMAP
    lookup; the eval points are otherwise identical for any run against the
    same physical scene, so an existing handoff/<gt_scene>/eval_points.npz is
    reused rather than recomputed (same GT, same deterministic sampling)."""
    gt_scene = gt_scene or scene
    out = f"student_gpu_package/handoff/{scene}/eval_points.npz"
    if os.path.exists(out):
        return out
    shared = f"student_gpu_package/handoff/{gt_scene}/eval_points.npz"
    if gt_scene != scene and os.path.exists(shared):
        print(f"reusing {shared} (same physical scene, same GT)")
        return shared
    # reuse the GT extractor's machinery via its saved per-scene assets
    from tools.replica_gt_from_renders import scene_paths  # noqa
    vscene = re.sub(r"(\D)(\d+)$", r"\1_\2", gt_scene)
    ex_dir = f"data/replica/{vscene}_vmap"
    from PIL import Image
    inst = sorted(f for f in os.listdir(ex_dir)
                  if re.fullmatch(r"semantic_class_\d+\.png", f))
    dep = sorted(f for f in os.listdir(ex_dir)
                 if re.fullmatch(r"depth_\d+\.png", f))
    traj = np.loadtxt(os.path.join(ex_dir, "traj_w_c.txt")).reshape(-1, 4, 4)
    meta = json.load(open(os.path.join(ex_dir, "info_semantic.json")))
    id2name = {int(c["id"]): c["name"] for c in meta.get("classes", [])}
    if not inst:
        raise SystemExit(f"FAIL: no semantic_class renders in {ex_dir} — "
                         f"rerun tools/replica_gt_from_renders.py --scene {scene}")
    idx = lambda n: int(re.search(r"(\d+)\.png$", n).group(1))
    dep_by = {idx(n): n for n in dep}
    P, L = [], []
    for n in inst[::stride]:
        fi = idx(n)
        if fi not in dep_by:
            continue
        C = np.asarray(Image.open(os.path.join(ex_dir, n)))
        D = np.asarray(Image.open(os.path.join(ex_dir, dep_by[fi])),
                       np.float32) / 1000.0
        H, W = D.shape
        fx = 600.0 * W / 1200.0
        cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
        vv, uu = np.where((D > 0.1) & (D < 10.0))
        sel = np.random.RandomState(fi).choice(
            len(vv), min(3000, len(vv)), replace=False)
        vv, uu = vv[sel], uu[sel]
        z = D[vv, uu]
        pc = np.stack([(uu - cx) / fx * z, (vv - cy) / fx * z, z])
        M = traj[fi]
        pw = (M[:3, :3] @ pc + M[:3, 3:4]).T
        P.append(pw)
        L.extend(id2name.get(int(c), "unknown") for c in C[vv, uu])
    P = np.concatenate(P)
    L = np.array(L)
    if len(P) > n_pts:
        sel = np.random.RandomState(0).choice(len(P), n_pts, replace=False)
        P, L = P[sel], L[sel]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, xyz=P.astype(np.float32), gt_class=L)
    print(f"eval points: {len(P)} ({out})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="room0")
    ap.add_argument("--gt-scene", default=None,
                    help="physical scene to read GT/vmap renders from, if "
                         "different from --scene (e.g. an alternate-frontend "
                         "run named 'room0_openvocab' still shares room0's "
                         "ground truth). Defaults to --scene.")
    ap.add_argument("--max-per-class", type=int, default=60)
    ap.add_argument("--grid", type=int, default=96)
    args = ap.parse_args()
    scene = args.scene
    gt_scene = args.gt_scene or scene

    classes, _ = replica_class_list(gt_scene)
    print(f"{len(classes)} Replica classes (their vocabulary)")
    lab = clip_relabel(scene, classes)

    pts = json.load(open(f"outputs/replica_{scene}/object_points.json"))["points"]
    n0 = len(pts)
    if any(p.get("det") is None for p in pts):
        # older object_points.json lacks det ids: rebuild the join by
        # sweeping the csv's (frame, class) queues in order
        from collections import defaultdict, deque
        q = defaultdict(deque)
        for r in csv.DictReader(open(f"outputs/replica_{scene}/"
                                     "detections_crops.csv")):
            q[(int(r["frame_idx"]), r["class_name"])].append(int(r["det_id"]))
        for p in pts:
            key = (p["frame"], p["cls"])
            p["det"] = q[key].popleft() if q[key] else None
    for p in pts:
        p["cls"] = lab.get(p.get("det"), None) if p.get("det") is not None else None
    pts = [p for p in pts if p["cls"]]
    print(f"observations relabelled to Replica vocab: {len(pts)}/{n0}")
    if not pts:
        raise SystemExit("FAIL: relabel join produced zero observations — "
                         "send this log to Paul")

    ep = build_eval_points(scene, gt_scene=gt_scene)
    E = np.load(ep, allow_pickle=True)
    xyz, gt = E["xyz"], E["gt_class"]

    # floor-plane axes: same rule as everywhere (largest-variance axes)
    var = xyz.var(0)
    a, b = sorted(np.argsort(var)[-2:])
    enc = ClassroomEncoders(HD, 0, LS, 20.0)
    sem = class_phasors(sorted({p["cls"] for p in pts}), HD)
    trace = build_trace(cap_per_class(pts, args.max_per_class), enc, sem, HD)
    trace /= max(np.abs(trace).max(), 1e-12)

    # dense labels: argmax over class fields on a grid, then nearest-cell
    xs, ys = xyz[:, a], xyz[:, b]
    gx = np.linspace(xs.min(), xs.max(), args.grid)
    gy = np.linspace(ys.min(), ys.max(), args.grid)
    G = np.empty((args.grid ** 2, HD), np.complex64)
    k = 0
    for yy in gy:
        for xx in gx:
            G[k] = enc.ctx_pos(float(xx), float(yy)).values.astype(np.complex64)
            k += 1
    names = sorted(sem)
    F = np.stack([((trace / sem[c])[None, :] @ np.conj(G).T).real[0]
                  for c in names])                       # (C, grid^2)
    winner = F.argmax(0)                                 # class per cell
    ix = np.clip(np.searchsorted(gx, xs), 0, args.grid - 1)
    iy = np.clip(np.searchsorted(gy, ys), 0, args.grid - 1)
    pred = np.array([names[winner[iy_ * args.grid + ix_]]
                     for ix_, iy_ in zip(ix, iy)])
    out = f"student_gpu_package/handoff/{scene}/vsa_labels.npz"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, xyz=xyz, pred_class=pred)
    agree = float(np.mean(pred == gt))
    print(f"wrote {out}; raw point-label agreement {agree:.1%} "
          f"(the real score comes from 05_score.py)")
    print("STAGE OK (04_vsa_labels)")


if __name__ == "__main__":
    main()
