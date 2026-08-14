"""Instance vs semantic recall: does the memory find THIS chair, or A chair?

Paul's diagnosis (2026-08-06): a class-keyed trace (class ⊗ position) has no
associative binding between a specific instance's identity and its location —
all instances of a class share one key, so recall can be class-correct but
instance-wrong. This experiment measures that failure and its VSA-native fix.

Three memory designs, identical observation stream:
  class   sem[class] ⊗ S(x,y)          (the current object map)
  app     φ(appearance) ⊗ S(x,y)       (each detection's own crop embedding
                                        as the key — instance-associative)
  hybrid  φ(appearance) ⊗ sem[class] ⊗ S(x,y)

Protocol: detections are assigned to GT instances (same mapped class,
nearest centroid within --assign-radius). Each instance's detections are
split into memory half / query half by frame order. A query is a HELD-OUT
view of a known instance; each design decodes a position; the answer is
scored against the GT instance set of that class:
  instance-correct        peak within radius of THE queried instance
  class-correct-wrong-i   peak within radius of a DIFFERENT instance, same class
  wrong                   neither
Multi-instance classes (chairs x8-10, vases x5) are the diagnostic rows;
singletons cannot show the confusion and are reported separately.

    python -m vsa_cognitive_mapping.instance_recall \
        --scenes room2 office4 room0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vsa_cognitive_mapping.classroom_pipeline import ClassroomEncoders  # noqa: E402
from vsa_cognitive_mapping.object_grounding import (  # noqa: E402
    REPLICA2COCO, class_phasors)
from vsa_cognitive_mapping.vsa import random_project_to_phasor  # noqa: E402

HD, GRID, LS = 4096, 64, 0.6


def load_scene(scene):
    d = f"outputs/replica_{scene}"
    pts = json.load(open(os.path.join(d, "object_points.json")))["points"]
    emb = torch.load(os.path.join(d, "crop_embeddings_dinov2.pt")
                     if os.path.exists(os.path.join(
                         d, "crop_embeddings_dinov2.pt"))
                     else os.path.join(d, "crop_embeddings.pt"),
                     weights_only=False)["embedding"].numpy().astype(np.float64)
    gt = json.load(open(os.path.join(d, "gt_instances.json")))["instances"]
    inst = [dict(cls=REPLICA2COCO.get(g["cls"]), x=g["x"], y=g["y"], id=i)
            for i, g in enumerate(gt) if REPLICA2COCO.get(g["cls"])]
    # det join (object_points rows may predate the det field)
    if any(p.get("det") is None for p in pts):
        import csv
        from collections import deque
        q = defaultdict(deque)
        for r in csv.DictReader(open(os.path.join(d, "detections_crops.csv"))):
            q[(int(r["frame_idx"]), r["class_name"])].append(int(r["det_id"]))
        for p in pts:
            key = (p["frame"], p["cls"])
            p["det"] = q[key].popleft() if q[key] else None
    pts = [p for p in pts if p.get("det") is not None and p["det"] < len(emb)]
    return pts, emb, inst


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", nargs="+",
                    default=["room2", "office4", "room0"])
    ap.add_argument("--radius", type=float, default=0.75,
                    help="instance-correct radius (m); instances closer than "
                         "2x this are merged for scoring, not guessed apart")
    ap.add_argument("--assign-radius", type=float, default=1.0)
    ap.add_argument("--min-views", type=int, default=6,
                    help="instance needs this many memory views to be queried")
    ap.add_argument("--max-per-instance", type=int, default=40)
    args = ap.parse_args()

    enc = ClassroomEncoders(HD, 0, LS, 20.0)
    agg = {k: defaultdict(int) for k in ("class", "app", "hybrid")}
    agg_multi = {k: defaultdict(int) for k in ("class", "app", "hybrid")}

    for scene in args.scenes:
        pts, emb, inst = load_scene(scene)
        by_cls_inst = defaultdict(list)
        for g in inst:
            by_cls_inst[g["cls"]].append(g)

        # assign detections to GT instances
        assigned = []
        for p in pts:
            cands = by_cls_inst.get(p["cls"], [])
            if not cands:
                continue
            dd = [np.hypot(p["x"] - g["x"], p["y"] - g["y"]) for g in cands]
            j = int(np.argmin(dd))
            if dd[j] <= args.assign_radius:
                assigned.append((p, cands[j]["id"]))
        det_ids = [p["det"] for p, _ in assigned]
        E = emb[det_ids]
        mu, sd = E.mean(0), E.std(0) + 1e-8
        z, _ = random_project_to_phasor(
            torch.from_numpy(np.ascontiguousarray((E - mu) / sd)).float(),
            d=HD, seed=0)
        APP = z.numpy().astype(np.complex128)

        # split per instance by frame order
        per_inst = defaultdict(list)
        for k, (p, iid) in enumerate(assigned):
            per_inst[iid].append(k)
        mem_idx, qry_idx = [], []
        for iid, ks in per_inst.items():
            ks = sorted(ks, key=lambda k: assigned[k][0]["frame"])
            half = len(ks) // 2
            mem_idx.extend(ks[:half][:args.max_per_instance])
            if len(ks[:half]) >= args.min_views:
                qry_idx.extend(ks[half:][:20])

        classes = sorted({p["cls"] for p, _ in assigned})
        sem = class_phasors(classes, HD)
        xs = [p["x"] for p, _ in assigned]
        ys = [p["y"] for p, _ in assigned]
        ext = [min(xs) - 1, max(xs) + 1, min(ys) - 1, max(ys) + 1]
        gx = np.linspace(ext[0], ext[1], GRID)
        gy = np.linspace(ext[2], ext[3], GRID)
        G = np.empty((GRID * GRID, HD), np.complex64)
        k2 = 0
        for yy in gy:
            for xx in gx:
                G[k2] = enc.ctx_pos(float(xx), float(yy)).values.astype(
                    np.complex64)
                k2 += 1

        P = {k: enc.ctx_pos(float(assigned[k][0]["x"]),
                            float(assigned[k][0]["y"])).values
             for k in mem_idx}
        M = dict(cls=np.zeros(HD, np.complex128),
                 app=np.zeros(HD, np.complex128),
                 hyb=np.zeros(HD, np.complex128))
        for k in mem_idx:
            p, _ = assigned[k]
            M["cls"] += sem[p["cls"]] * P[k]
            M["app"] += APP[k] * P[k]
            M["hyb"] += APP[k] * sem[p["cls"]] * P[k]
        for m in M.values():
            m /= max(np.abs(m).max(), 1e-12)

        gt_pos = {g["id"]: (g["x"], g["y"]) for g in inst}
        n_multi = {c: len(v) for c, v in by_cls_inst.items()}

        def score(design, key_vec, p, iid):
            F = ((M[design] / key_vec)[None, :] @ np.conj(G).T).real[0]
            j = int(F.argmax())
            px, py = gx[j % GRID], gy[j // GRID]
            d_true = np.hypot(px - gt_pos[iid][0], py - gt_pos[iid][1])
            others = [np.hypot(px - gt_pos[g["id"]][0], py - gt_pos[g["id"]][1])
                      for g in by_cls_inst[p["cls"]] if g["id"] != iid]
            d_other = min(others) if others else np.inf
            if d_true <= args.radius and d_true <= d_other:
                return "instance"
            if d_other <= args.radius:
                return "wrong_instance"
            return "wrong"

        design_key = dict(cls=("class", lambda p, k: sem[p["cls"]]),
                          app=("app", lambda p, k: APP[k]),
                          hyb=("hybrid", lambda p, k: APP[k] * sem[p["cls"]]))
        n_q = 0
        for k in qry_idx:
            p, iid = assigned[k]
            multi = n_multi[p["cls"]] > 1
            n_q += 1
            for dkey, (name, fn) in design_key.items():
                r = score(dkey, fn(p, k), p, iid)
                agg[name][r] += 1
                if multi:
                    agg_multi[name][r] += 1
        print(f"{scene}: {len(assigned)} assigned detections, "
              f"{len(per_inst)} instances, {n_q} held-out queries "
              f"({sum(1 for k in qry_idx if n_multi[assigned[k][0]['cls']] > 1)}"
              f" on multi-instance classes)")

    def table(a, label):
        print(f"\n=== {label} ===")
        hdr = (f"{'memory design':<26}{'instance-correct':>17}"
               f"{'class-ok wrong-inst':>20}{'wrong':>8}")
        print(hdr)
        print("-" * len(hdr))
        for name in ("class", "app", "hybrid"):
            tot = sum(a[name].values()) or 1
            print(f"{name:<26}{a[name]['instance'] / tot:>17.0%}"
                  f"{a[name]['wrong_instance'] / tot:>20.0%}"
                  f"{a[name]['wrong'] / tot:>8.0%}")

    table(agg, "all queried instances")
    table(agg_multi, "MULTI-instance classes only (the diagnostic)")
    out = "outputs/instance_recall.json"
    with open(out, "w") as f:
        json.dump(dict(all={k: dict(v) for k, v in agg.items()},
                       multi={k: dict(v) for k, v in agg_multi.items()},
                       scenes=args.scenes, radius=args.radius), f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
