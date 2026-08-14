"""R2: VIEW+STATEMENT battery — a held-out VIEW of a specific instance PLUS
a verbal spatial anchor ("this thing I'm looking at, the one near the lamp").

Measures whether the verbal anchor rescues appearance keys where they fail,
and whether it interferes where they succeed. Reuses relational_recall.py's
machinery exactly: same scenes, GT assignment, per-instance frame-order
memory/query split, class/appearance phasors, grid, deterministic
GT-anchored query battery (all target-class x anchor-class pairs,
unique-with-margin filter) and instance-correct scoring protocol.
No protocol constants changed; no new tunable parameters.

Designs (per query (c, anchor, target) x held-out view k of the target,
up to 3 views, as R1's app rows):
  V    appearance field argmax                       (reference, ~72%)
  V+A  (appearance field x anchor proximity field) argmax   (product)
  V|A  conditional rescue: use the appearance argmax UNLESS the anchor
       field's value at that argmax is below the anchor field's own median
       over the grid; in that case fall back to the product argmax.
       The median is the field's own scale-free order statistic — this is
       a decision rule, not a tuned threshold.

Every design is stratified by whether the plain appearance answer (V) was
instance-correct: rescued-when-wrong vs damaged-when-right counts.

    python -m vsa_cognitive_mapping.relational_view_battery \
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
from vsa_cognitive_mapping.object_grounding import class_phasors  # noqa: E402
from vsa_cognitive_mapping.instance_recall import load_scene  # noqa: E402
from vsa_cognitive_mapping.vsa import random_project_to_phasor  # noqa: E402

HD, GRID, LS = 4096, 64, 0.6

DESIGNS = ("V", "V+A", "V|A")
RULE = ("V|A rule: answer = appearance-field argmax UNLESS the anchor "
        "field's value at that argmax is below the anchor field's own "
        "median over the grid; then answer = product argmax. The median "
        "is scale-free - a decision rule, not a tuned threshold.")


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
    agg = {k: defaultdict(int) for k in DESIGNS}
    # per design: 2x2 vs the plain appearance answer (V)
    strat = {k: defaultdict(int) for k in DESIGNS}
    n_amb = 0
    n_fallback = 0            # V|A: how often the rule fell back to product
    n_views_total = 0
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

        # ---- deterministic battery from GT (identical to R1) --------------
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

        def judge_at(j, tid, cname):
            return judge(gx[j % GRID], gy[j // GRID], tid, cname)

        n_sc = 0
        for c, ac, tid in queries:
            if tid not in qview:
                continue
            Fa = field(M, sem[ac])          # anchor proximity field
            med_a = float(np.median(Fa))    # its own scale-free median
            for k in qview[tid][:3]:
                Fapp = field(Mapp, APP[k])
                j_app = int(Fapp.argmax())
                j_prod = int((Fapp * Fa).argmax())

                res = {}
                res["V"] = judge_at(j_app, tid, c)
                res["V+A"] = judge_at(j_prod, tid, c)
                if Fa[j_app] >= med_a:
                    res["V|A"] = res["V"]
                else:
                    res["V|A"] = res["V+A"]
                    n_fallback += 1

                app_ok = res["V"] == "instance"
                for name in DESIGNS:
                    agg[name][res[name]] += 1
                    d_ok = res[name] == "instance"
                    cell = ("kept_right" if app_ok and d_ok else
                            "damaged_when_right" if app_ok else
                            "rescued_when_wrong" if d_ok else
                            "still_wrong")
                    strat[name][cell] += 1
                n_views_total += 1
            n_sc += 1
        per_scene.append(dict(scene=scene, queries_with_views=n_sc))
        print(f"{scene}: {n_sc} unambiguous relational queries with "
              f"held-out views ({len(queries)} in battery)")

    print(f"\nambiguous queries discarded across scenes (margin "
          f"{args.margin} m): {n_amb}")
    print(RULE)
    print(f"V|A fell back to the product argmax on {n_fallback} of "
          f"{n_views_total} views")

    hdr = (f"{'design':<8}{'n':>6}{'instance-correct':>17}"
           f"{'class-ok wrong-inst':>20}{'wrong':>8}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for name in DESIGNS:
        tot = sum(agg[name].values()) or 1
        print(f"{name:<8}{tot:>6}{agg[name]['instance'] / tot:>17.0%}"
              f"{agg[name]['wrong_instance'] / tot:>20.0%}"
              f"{agg[name]['wrong'] / tot:>8.0%}")

    hdr2 = (f"{'design':<8}{'kept-right':>12}{'damaged-when-right':>20}"
            f"{'rescued-when-wrong':>20}{'still-wrong':>13}")
    print("\nstratified by whether the plain appearance answer (V) was "
          "instance-correct:")
    print(hdr2)
    print("-" * len(hdr2))
    for name in DESIGNS:
        s = strat[name]
        print(f"{name:<8}{s['kept_right']:>12}{s['damaged_when_right']:>20}"
              f"{s['rescued_when_wrong']:>20}{s['still_wrong']:>13}")

    out = "outputs/relational_view_battery.json"
    with open(out, "w") as f:
        json.dump(dict(designs={k: dict(v) for k, v in agg.items()},
                       rescue_damage={k: dict(v) for k, v in strat.items()},
                       rule=RULE,
                       fallback_views=n_fallback, total_views=n_views_total,
                       per_scene=per_scene, discarded_ambiguous=n_amb,
                       radius=args.radius, margin=args.margin), f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
