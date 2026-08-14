"""Relational recall: "the {class} nearest the {anchor}" from field algebra.

Paul's idea (plan: wiki/analysis/2026-08-06-relational-recall-plan.md):
relations need not be stored — position codes contain them implicitly, and
query-time field products apply them. "The chair near the lamp" = unbind
CHAIR (field over all chairs), unbind LAMP (field over lamps), multiply:
the chair whose mass sits near lamp mass wins. One 32 KB class-keyed trace,
two unbinds, one product. No vision model, no relation database.

Deterministic battery from GT (no hand-picking): for every GT instance t of
a MULTI-instance class c, the anchor is the nearest different-class GT
instance a; the query "the {c} nearest the {cls(a)}" is kept iff t is the
unique correct answer with >= --margin over the runner-up c-instance
(computed from GT; the count of discarded ambiguous queries is printed).
Scoring: the established instance-correct protocol (radius 0.75 m, as
instance_recall.py). Proximity kernel: the FPE similarity kernel already in
the encoders (length-scale 0.6 m) — no new parameters anywhere.

Designs (same observation stream):
  class        F_c argmax                      (the ~25% baseline)
  relational   (relu F_c · relu K⊛F_a) argmax  (this idea)
  app          appearance-key field argmax      (the 43% reference; query =
                                                held-out view of t)
  app+rel      appearance field · anchor proximity

    python -m vsa_cognitive_mapping.relational_recall \
        --scenes room2 office4 room0 office2 office3
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
    class_phasors, field_peaks)
from vsa_cognitive_mapping.instance_recall import load_scene  # noqa: E402
from vsa_cognitive_mapping.vsa import random_project_to_phasor  # noqa: E402

HD, GRID, LS = 4096, 64, 0.6


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", nargs="+",
                    default=["room2", "office4", "room0", "office2", "office3"])
    ap.add_argument("--radius", type=float, default=0.75)
    ap.add_argument("--margin", type=float, default=0.5,
                    help="query kept iff target beats runner-up by this")
    ap.add_argument("--assign-radius", type=float, default=1.0)
    ap.add_argument("--min-views", type=int, default=6)
    ap.add_argument("--max-per-class", type=int, default=60)
    args = ap.parse_args()

    enc = ClassroomEncoders(HD, 0, LS, 20.0)
    agg = {k: defaultdict(int) for k in ("class", "product", "selector",
                                         "app", "app+product",
                                         "app+selector")}
    n_amb = 0
    per_scene = []

    for scene in args.scenes:
        pts, emb, inst = load_scene(scene)
        by_cls_inst = defaultdict(list)
        for g in inst:
            by_cls_inst[g["cls"]].append(g)

        # detection -> GT instance assignment (as instance_recall)
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

        per_i = defaultdict(list)
        for k, (p, iid) in enumerate(assigned):
            per_i[iid].append(k)
        mem_idx, qview = [], {}
        for iid, ks in per_i.items():
            ks = sorted(ks, key=lambda k: assigned[k][0]["frame"])
            half = len(ks) // 2
            mem_idx.extend(ks[:half][:args.max_per_class])
            if len(ks[:half]) >= args.min_views and ks[half:]:
                qview[iid] = ks[half:][:10]      # held-out views for app modes

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

        M = np.zeros(HD, np.complex128)
        Mapp = np.zeros(HD, np.complex128)
        for k in mem_idx:
            p, _ = assigned[k]
            P = enc.ctx_pos(float(p["x"]), float(p["y"])).values
            M += sem[p["cls"]] * P
            Mapp += APP[k] * P
        M /= max(np.abs(M).max(), 1e-12)
        Mapp /= max(np.abs(Mapp).max(), 1e-12)

        def field(trace, key):
            F = ((trace / key)[None, :] @ np.conj(G).T).real[0]
            return np.maximum(F, 0.0)

        gt_pos = {g["id"]: (g["x"], g["y"]) for g in inst}

        # ---- deterministic battery from GT --------------------------------
        # For every multi-instance target class c and EVERY other class ac,
        # the query "the {c} nearest the {ac}" is kept iff exactly one
        # c-instance is nearest to ac (min over ac's instances) with margin.
        queries = []
        anchor_classes = [a for a in by_cls_inst if a in sem]
        for c, insts in by_cls_inst.items():
            if len(insts) < 2 or c not in sem:
                continue
            for ac in anchor_classes:
                if ac == c:
                    continue
                d_to_a = {g["id"]: min(
                    np.hypot(g["x"] - b["x"], g["y"] - b["y"])
                    for b in by_cls_inst[ac])
                    for g in insts}
                order = sorted(insts, key=lambda g: d_to_a[g["id"]])
                if (d_to_a[order[1]["id"]] - d_to_a[order[0]["id"]]
                        < args.margin):
                    n_amb += 1
                    continue
                queries.append((c, ac, order[0]["id"]))

        def judge(px, py, tid, cname):
            d_true = np.hypot(px - gt_pos[tid][0], py - gt_pos[tid][1])
            others = [np.hypot(px - gt_pos[g["id"]][0],
                               py - gt_pos[g["id"]][1])
                      for g in by_cls_inst[cname] if g["id"] != tid]
            d_other = min(others) if others else np.inf
            if d_true <= args.radius and d_true <= d_other:
                return "instance"
            if d_other <= args.radius:
                return "wrong_instance"
            return "wrong"

        def score_argmax(F, tid, cname):
            j = int(F.argmax())
            return judge(gx[j % GRID], gy[j // GRID], tid, cname)

        def score_selected(Ftarget, Fanchor, tid, cname):
            """Decode matching the sentence's semantics: among the TARGET
            field's modes, select the one where the anchor field is
            strongest. (A joint product's argmax drifts BETWEEN target and
            anchor — the product of two kernels peaks in the gap.)"""
            k = min(max(len(by_cls_inst[cname]), 2), 5)
            pks = field_peaks(Ftarget, gx, gy, GRID, k=k)
            def aval(pk):
                ix = int(np.clip(np.searchsorted(gx, pk[0]), 0, GRID - 1))
                iy = int(np.clip(np.searchsorted(gy, pk[1]), 0, GRID - 1))
                return Fanchor[iy * GRID + ix]
            best = max(pks, key=aval)
            return judge(best[0], best[1], tid, cname)

        n_sc = 0
        for c, ac, tid in queries:
            Fc = field(M, sem[c])
            Fa = field(M, sem[ac])
            agg["class"][score_argmax(Fc, tid, c)] += 1
            # BOTH relational decodes, as separate named designs, so every
            # quoted number has a reproducing artifact (skeptic-panel fix:
            # the product row had been edited out after being recorded)
            agg["product"][score_argmax(Fc * Fa, tid, c)] += 1
            agg["selector"][score_selected(Fc, Fa, tid, c)] += 1
            if tid in qview:
                for k in qview[tid][:3]:
                    Fapp = field(Mapp, APP[k])
                    agg["app"][score_argmax(Fapp, tid, c)] += 1
                    agg["app+product"][score_argmax(Fapp * Fa, tid, c)] += 1
                    agg["app+selector"][score_selected(Fapp, Fa, tid, c)] += 1
            n_sc += 1
        per_scene.append(dict(scene=scene, queries=n_sc))
        print(f"{scene}: {n_sc} unambiguous relational queries "
              f"({len(queries)} kept)")

    print(f"\nambiguous queries discarded across scenes (margin "
          f"{args.margin} m): {n_amb}")
    hdr = (f"{'design':<14}{'n':>6}{'instance-correct':>17}"
           f"{'class-ok wrong-inst':>20}{'wrong':>8}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for name in ("class", "product", "selector", "app", "app+product",
                 "app+selector"):
        tot = sum(agg[name].values()) or 1
        print(f"{name:<14}{tot:>6}{agg[name]['instance'] / tot:>17.0%}"
              f"{agg[name]['wrong_instance'] / tot:>20.0%}"
              f"{agg[name]['wrong'] / tot:>8.0%}")
    out = "outputs/relational_recall.json"
    with open(out, "w") as f:
        json.dump(dict(designs={k: dict(v) for k, v in agg.items()},
                       per_scene=per_scene, discarded_ambiguous=n_amb,
                       radius=args.radius, margin=args.margin), f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
