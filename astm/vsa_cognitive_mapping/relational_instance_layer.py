"""Path C scout: relational selection at the INSTANCE layer.

Same battery as relational_recall.py (identical load_scene, GT assignment,
memory split, deterministic query generator, judge scoring, constants), two
new mechanisms answering "the {c} nearest the {a}":

  cluster   Explicit instance layer: cluster the MEMORY-half observations
            per class (clusters_of, radius 0.6 / min_count 4 — the
            established object_grounding constants). Among c-clusters pick
            the centroid minimising distance to the nearest a-cluster
            centroid. Pure symbol selection — concedes the algebra,
            mirrors FARM's object layer.
  pursuit   Matching pursuit ON THE FIELD: decode F_c from the 32 KB trace;
            iteratively take argmax, record candidate p_i, subtract the
            model response lambda_i * K(x - p_i) (K = the FPE similarity
            kernel on the grid, lambda_i = the residual value at p_i);
            k = number of GT instances of c. Candidates are enumerated
            from the trace ALONE; selection = the candidate maximising the
            anchor field value at the candidate (R1's selector semantics).
            pursuit_k+1 reruns with k+1 iterations (robustness to an
            over-counted cache).

A rerun of the R1 class baseline validates that the battery is bit-identical
to the recorded relational_recall.py run. Enumeration quality (fraction of
GT instances of the queried class with a pursuit candidate within the
scoring radius) isolates enumeration from selection as the limiter.

    python -m vsa_cognitive_mapping.relational_instance_layer \
        --scenes room2 office4 room0 office2 office3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vsa_cognitive_mapping.classroom_pipeline import ClassroomEncoders  # noqa: E402
from vsa_cognitive_mapping.object_grounding import (  # noqa: E402
    class_phasors, clusters_of)
from vsa_cognitive_mapping.instance_recall import load_scene  # noqa: E402

HD, GRID, LS = 4096, 64, 0.6
CLUSTER_RADIUS, MIN_COUNT = 0.6, 4        # established object_grounding values

# Recorded R1 field-product numbers (outputs/relational_recall.json +
# the logged product variant), quoted for comparison — not recomputed here
# except the class baseline, which validates battery identity.
R1_RECORDED = {
    "R1 class":    dict(instance=29, wrong_instance=71, wrong=0),
    "R1 product":  dict(instance=46, wrong_instance=21, wrong=33),
    "R1 selector": dict(instance=29, wrong_instance=58, wrong=12),
}

DESIGNS = ("class_rerun", "cluster", "pursuit", "pursuit_k+1",
           "pursuit_euclid")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", nargs="+",
                    default=["room2", "office4", "room0", "office2", "office3"])
    ap.add_argument("--radius", type=float, default=0.75)
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--assign-radius", type=float, default=1.0)
    ap.add_argument("--min-views", type=int, default=6)
    ap.add_argument("--max-per-class", type=int, default=60)
    args = ap.parse_args()

    enc = ClassroomEncoders(HD, 0, LS, 20.0)
    agg = {k: defaultdict(int) for k in DESIGNS}
    per_scene = []
    n_amb = 0
    enum_cov, enum_tot = 0, 0                 # pursuit enumeration quality
    enum_by_scene = {}
    n_no_target_cluster, n_no_anchor_cluster = 0, 0

    for scene in args.scenes:
        pts, _emb, inst = load_scene(scene)
        by_cls_inst = defaultdict(list)
        for g in inst:
            by_cls_inst[g["cls"]].append(g)

        # detection -> GT instance assignment (identical to relational_recall)
        assigned = []
        for p in pts:
            cands = by_cls_inst.get(p["cls"], [])
            if not cands:
                continue
            dd = [np.hypot(p["x"] - g["x"], p["y"] - g["y"]) for g in cands]
            j = int(np.argmin(dd))
            if dd[j] <= args.assign_radius:
                assigned.append((p, cands[j]["id"]))

        # memory half (identical split; query views only feed app modes,
        # which do not exist here)
        per_i = defaultdict(list)
        for k, (p, iid) in enumerate(assigned):
            per_i[iid].append(k)
        mem_idx = []
        for iid, ks in per_i.items():
            ks = sorted(ks, key=lambda k: assigned[k][0]["frame"])
            half = len(ks) // 2
            mem_idx.extend(ks[:half][:args.max_per_class])

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
        for k in mem_idx:
            p, _ = assigned[k]
            M += sem[p["cls"]] * enc.ctx_pos(float(p["x"]),
                                             float(p["y"])).values
        M /= max(np.abs(M).max(), 1e-12)

        def field(key):
            F = ((M / key)[None, :] @ np.conj(G).T).real[0]
            return np.maximum(F, 0.0)

        gt_pos = {g["id"]: (g["x"], g["y"]) for g in inst}

        # ---- deterministic battery (identical to relational_recall) -------
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

        # ---- design CLUSTER: explicit instance layer from memory half -----
        mem_obs = [assigned[k][0] for k in mem_idx]
        clus_by = defaultdict(list)
        for cl in clusters_of(mem_obs, CLUSTER_RADIUS, MIN_COUNT):
            clus_by[cl["class"]].append((cl["x"], cl["y"]))

        def answer_cluster(c, ac):
            tc, aa = clus_by.get(c), clus_by.get(ac)
            if not tc:
                return None, "no_target_cluster"
            if not aa:
                return None, "no_anchor_cluster"
            best = min(tc, key=lambda t: min(
                np.hypot(t[0] - a[0], t[1] - a[1]) for a in aa))
            return best, None

        # ---- design PURSUIT: matching pursuit on the decoded field --------
        fields = {}                            # class -> relu field (cached)

        def get_field(c):
            if c not in fields:
                fields[c] = field(sem[c])
            return fields[c]

        pursuit_cache = {}

        def pursuit(c, n):
            """n matching-pursuit candidates from F_c; K normalised so
            K(0)=1, lambda_i = residual value at the pick."""
            key = (c, n)
            if key in pursuit_cache:
                return pursuit_cache[key]
            R = get_field(c).astype(np.float64).copy()
            cands = []
            for _ in range(n):
                j = int(R.argmax())
                lam = float(R[j])
                if lam <= 0.0:
                    break
                K = (G[j][None, :] @ np.conj(G).T).real[0] / HD
                R -= lam * K
                cands.append((float(gx[j % GRID]), float(gy[j // GRID]), j))
            pursuit_cache[key] = cands
            return cands

        def select_by_anchor(cands, ac):
            Fa = get_field(ac)
            return max(cands, key=lambda t: Fa[t[2]])

        sc = {k: defaultdict(int) for k in DESIGNS}
        enum_seen = set()
        sc_cov, sc_tot = 0, 0
        for c, ac, tid in queries:
            # R1 class baseline rerun (battery-identity check)
            Fc = get_field(c)
            j = int(Fc.argmax())
            sc["class_rerun"][judge(gx[j % GRID], gy[j // GRID], tid, c)] += 1

            # cluster
            ans, fail = answer_cluster(c, ac)
            if ans is None:
                sc["cluster"]["wrong"] += 1
                if fail == "no_target_cluster":
                    n_no_target_cluster += 1
                else:
                    n_no_anchor_cluster += 1
            else:
                sc["cluster"][judge(ans[0], ans[1], tid, c)] += 1

            # pursuit at k and k+1 (k = GT instance count, known to battery)
            k_gt = len(by_cls_inst[c])
            for name, n_it in (("pursuit", k_gt), ("pursuit_k+1", k_gt + 1)):
                cands = pursuit(c, n_it)
                if not cands:
                    sc[name]["wrong"] += 1
                    continue
                bx, by, _ = select_by_anchor(cands, ac)
                sc[name][judge(bx, by, tid, c)] += 1

            # PURSUIT-EUCLID: the closing test — CLUSTER's Euclidean rule
            # over PURSUIT's candidates. Both target and anchor instance
            # sets are enumerated FROM THE TRACE (anchor k = GT count of
            # the anchor class, same battery convention as pursuit; at
            # deployment a count cache or stopping rule replaces it).
            tcands = pursuit(c, k_gt)
            acands = pursuit(ac, len(by_cls_inst[ac]))
            if not tcands or not acands:
                sc["pursuit_euclid"]["wrong"] += 1
            else:
                bx, by, _ = min(tcands, key=lambda t: min(
                    np.hypot(t[0] - a[0], t[1] - a[1]) for a in acands))
                sc["pursuit_euclid"][judge(bx, by, tid, c)] += 1

            # enumeration quality: once per queried class per scene
            if c not in enum_seen:
                enum_seen.add(c)
                cands = pursuit(c, k_gt)
                for g in by_cls_inst[c]:
                    sc_tot += 1
                    if any(np.hypot(g["x"] - cx, g["y"] - cy) <= args.radius
                           for cx, cy, _ in cands):
                        sc_cov += 1

        enum_cov += sc_cov
        enum_tot += sc_tot
        enum_by_scene[scene] = dict(covered=sc_cov, total=sc_tot)
        for name in DESIGNS:
            for r, v in sc[name].items():
                agg[name][r] += v
        per_scene.append(dict(scene=scene, queries=len(queries),
                              results={k: dict(v) for k, v in sc.items()}))
        print(f"{scene}: {len(queries)} queries | " + " | ".join(
            f"{n} {sc[n]['instance']}/{sc[n]['wrong_instance']}"
            f"/{sc[n]['wrong']}" for n in DESIGNS)
            + f" | enum {sc_cov}/{sc_tot}")

    print(f"\nambiguous queries discarded (margin {args.margin} m): {n_amb}")
    if n_no_target_cluster or n_no_anchor_cluster:
        print(f"cluster failures counted as wrong: {n_no_target_cluster} no "
              f"target-class cluster, {n_no_anchor_cluster} no anchor cluster")

    hdr = (f"{'design':<14}{'n':>6}{'instance-correct':>17}"
           f"{'class-ok wrong-inst':>20}{'wrong':>8}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for name, v in R1_RECORDED.items():
        print(f"{name:<14}{'':>6}{v['instance']:>16}%"
              f"{v['wrong_instance']:>19}%{v['wrong']:>7}%   (recorded)")
    for name in DESIGNS:
        tot = sum(agg[name].values()) or 1
        print(f"{name:<14}{tot:>6}{agg[name]['instance'] / tot:>17.0%}"
              f"{agg[name]['wrong_instance'] / tot:>20.0%}"
              f"{agg[name]['wrong'] / tot:>8.0%}")
    print(f"\npursuit enumeration quality (GT instances of queried classes "
          f"with a candidate within {args.radius} m): "
          f"{enum_cov}/{enum_tot} = {enum_cov / max(enum_tot, 1):.0%}")

    out = "outputs/relational_instance_layer.json"
    with open(out, "w") as f:
        json.dump(dict(designs={k: dict(v) for k, v in agg.items()},
                       r1_recorded=R1_RECORDED,
                       per_scene=per_scene,
                       enumeration=dict(covered=enum_cov, total=enum_tot,
                                        by_scene=enum_by_scene),
                       cluster_failures=dict(
                           no_target_cluster=n_no_target_cluster,
                           no_anchor_cluster=n_no_anchor_cluster),
                       cluster_radius=CLUSTER_RADIUS, min_count=MIN_COUNT,
                       discarded_ambiguous=n_amb,
                       radius=args.radius, margin=args.margin), f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
