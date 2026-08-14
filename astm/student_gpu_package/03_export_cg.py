"""Stage 3: export ConceptGraphs' outputs to the handoff schema.

Reads their saved result (pkl/pkl.gz under cg_out/<scene>/), writes:
  handoff/<scene>/cg_objects.json       [{class, x, y, z, n_detections}]
  handoff/<scene>/cg_observations.json  per-frame observation stream
  handoff/<scene>/eval_points.npz       the eval point cloud (xyz + gt label)
  handoff/<scene>/cg_labels.npz         their per-point predicted labels

Their result format has varied between releases; this script handles the
documented MapObjectList pickle. If loading fails it prints what it found —
send that log to Paul rather than guessing.

    python 03_export_cg.py --scene room0
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import pickle

import numpy as np

# anchor to this package dir like stages 4-5, so cg_out/handoff resolve the
# same regardless of the caller's working directory (notebook runs from the
# repo root, the README runs from student_gpu_package/)
PKG = os.path.dirname(os.path.abspath(__file__))


def load_their_result(out_dir):
    cands = (glob.glob(os.path.join(out_dir, "**", "*.pkl.gz"), recursive=True)
             + glob.glob(os.path.join(out_dir, "**", "*.pkl"), recursive=True))
    if not cands:
        raise SystemExit(f"FAIL: no pkl under {out_dir}. Contents:\n"
                         + "\n".join(glob.glob(os.path.join(out_dir, "*"))))
    path = max(cands, key=os.path.getsize)
    print(f"loading {path}")
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rb") as f:
        return pickle.load(f), path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="room0")
    ap.add_argument("--eval-points", type=int, default=200_000,
                    help="subsample of the fused cloud used as eval points")
    ap.add_argument("--cg-out", default=None,
                    help="dir holding their pkl.gz (default <pkg>/cg_out/<scene>)")
    args = ap.parse_args()

    res, src = load_their_result(args.cg_out
                                 or os.path.join(PKG, "cg_out", args.scene))
    # Documented layout: dict with 'objects' (MapObjectList) or the list itself
    objects = res.get("objects", res) if isinstance(res, dict) else res
    out_dir = os.path.join(PKG, "handoff", args.scene)
    os.makedirs(out_dir, exist_ok=True)

    def sniff(ob, keys, default):
        """Try many key spellings; works for dicts and attribute objects."""
        for k in keys:
            if hasattr(ob, "get") and ob.get(k) is not None:
                return ob.get(k)
            if hasattr(ob, k) and getattr(ob, k) is not None:
                return getattr(ob, k)
        return default

    obj_rows, obs_rows = [], []
    pts_all, lab_all = [], []
    for oid, ob in enumerate(objects):
        # schema drifts between releases — sniff the known spellings and,
        # on total failure, print what IS there instead of guessing
        raw_pts = sniff(ob, ["pcd_np", "points", "pcd_points"], None)
        if raw_pts is None:
            pcd = sniff(ob, ["pcd"], None)
            raw_pts = getattr(pcd, "points", None)
        pts = np.asarray(raw_pts) if raw_pts is not None else np.empty((0, 3))
        cname = sniff(ob, ["class_name", "most_common_class", "object_tag",
                           "caption", "class_id"], "unknown")
        frames = sniff(ob, ["image_idx", "frame_idx", "obs_frames"], [])
        if oid == 0:
            avail = (list(ob.keys()) if hasattr(ob, "keys")
                     else [a for a in dir(ob) if not a.startswith("_")])
            print(f"first object's available keys/attrs: {avail[:30]}")
        if pts.size == 0 and oid == 0:
            raise SystemExit("FAIL: cannot find point data on their objects — "
                             "send the printed key list above to Paul")
        if isinstance(cname, (list, np.ndarray)):
            cname = max(set(map(str, cname)), key=list(map(str, cname)).count)
        if len(pts) == 0:
            continue
        c = pts.mean(0)
        obj_rows.append(dict(id=oid, cls=str(cname), x=float(c[0]),
                             y=float(c[1]), z=float(c[2]),
                             n_points=int(len(pts))))
        for fr in list(frames)[:2000]:
            obs_rows.append(dict(frame=int(fr), obj=oid, cls=str(cname),
                                 x=float(c[0]), y=float(c[1]), z=float(c[2])))
        pts_all.append(pts)
        lab_all.append(np.full(len(pts), oid))

    with open(os.path.join(out_dir, "cg_objects.json"), "w") as f:
        json.dump(obj_rows, f, indent=1)
    with open(os.path.join(out_dir, "cg_observations.json"), "w") as f:
        json.dump(obs_rows, f)

    P = np.concatenate(pts_all)
    L = np.concatenate(lab_all)
    if len(P) > args.eval_points:
        sel = np.random.RandomState(0).choice(len(P), args.eval_points,
                                              replace=False)
        P, L = P[sel], L[sel]
    cls_of = {o["id"]: o["cls"] for o in obj_rows}
    names = np.array([cls_of[int(i)] for i in L])
    np.savez_compressed(os.path.join(out_dir, "cg_labels.npz"),
                        xyz=P.astype(np.float32), pred_class=names)
    print(f"{len(obj_rows)} objects, {len(obs_rows)} observations, "
          f"{len(P)} labelled points")
    print("NOTE: eval_points.npz (GT labels) is built by 04_vsa_labels.py "
          "from the semantic renders so BOTH systems are scored on the "
          "identical point set.")
    print("STAGE OK (03_export_cg)")


if __name__ == "__main__":
    main()
