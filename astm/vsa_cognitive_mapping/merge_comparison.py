"""Map JOINING head-to-head: VSA addition vs the alternatives, same data.

K robots each see a disjoint slice of a walk (contiguous frame blocks — the
realistic multi-session case). Each builds its map alone; the maps are then
joined and the joined map answers the grounding battery. Every method
consumes the identical observation stream (same-frontend rule).

Methods (what actually gets transmitted and how joining works):
  vsa        one 32 KB trace per robot; join = element-wise addition.
             No association, no optimisation. Exactness measured against
             the jointly-built trace.
  instances  per-robot instance list (cluster centroids); join = greedy
             cross-robot association (merge instances of the same class
             within assoc-radius). The scene-graph-style merge. We COUNT
             association decisions and errors vs clustering the pooled
             observations (the offline optimum).
  gridmap    per-robot per-class count grid (the occupancy-map analogue);
             join = cell-wise addition (also exact, but bytes scale with
             grid area x classes, and positions are quantised).
  concat     raw observation union — exact, trivially, at linear bytes.

All joins assume a SHARED WORLD FRAME (stated on output): alignment is
orthogonal to this comparison and is not solved by any of these joins.

    python -m vsa_cognitive_mapping.merge_comparison \
        --dataset vsa_cognitive_mapping/configs/replica_room0.json \
        --gt-json outputs/replica_room0/gt_instances.json --robots 4
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vsa_cognitive_mapping.classroom_pipeline import ClassroomEncoders  # noqa: E402
from vsa_cognitive_mapping.object_grounding import (  # noqa: E402
    REPLICA2COCO, class_phasors, build_trace, field_peaks, clusters_of,
    cap_per_class)


def split_robots(pts, k):
    frames = sorted({p["frame"] for p in pts})
    cuts = np.array_split(np.array(frames), k)
    sets = []
    for c in cuts:
        lo, hi = c[0], c[-1]
        sets.append([p for p in pts if lo <= p["frame"] <= hi])
    return sets


def assoc_merge(lists, radius):
    """Greedy cross-robot instance association: same class within radius ->
    merged (count-weighted centroid). Returns merged list + decision count."""
    merged = []
    decisions = 0
    for lst in lists:
        for inst in lst:
            decisions += 1
            hit = None
            for m in merged:
                if (m["class"] == inst["class"]
                        and np.hypot(m["x"] - inst["x"],
                                     m["y"] - inst["y"]) <= radius):
                    hit = m
                    break
            if hit is None:
                merged.append(dict(inst))
            else:
                w = hit["count"] + inst["count"]
                hit["x"] = (hit["x"] * hit["count"] + inst["x"] * inst["count"]) / w
                hit["y"] = (hit["y"] * hit["count"] + inst["y"] * inst["count"]) / w
                hit["count"] = w
    return merged, decisions


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--gt-json", required=True)
    ap.add_argument("--robots", type=int, default=4)
    ap.add_argument("--radius", type=float, default=1.0)
    ap.add_argument("--cluster-radius", type=float, default=0.6)
    ap.add_argument("--assoc-radius", type=float, default=0.75)
    ap.add_argument("--min-count", type=int, default=4)
    ap.add_argument("--hd-dim", type=int, default=4096)
    ap.add_argument("--grid", type=int, default=64)
    ap.add_argument("--length-scale", type=float, default=0.6)
    ap.add_argument("--room-margin", type=float, default=1.5)
    ap.add_argument("--max-per-class", type=int, default=60)
    args = ap.parse_args()

    cfg = json.load(open(args.dataset))
    out_dir = cfg["out_dir"]
    pts = json.load(open(os.path.join(out_dir, "object_points.json")))["points"]
    from vsa_cognitive_mapping.sequences import load_sequence
    seq = load_sequence(args.dataset)
    px, py, _ = seq.poses()
    lo_x, hi_x = px.min() - args.room_margin, px.max() + args.room_margin
    lo_y, hi_y = py.min() - args.room_margin, py.max() + args.room_margin
    pts = [p for p in pts if lo_x <= p["x"] <= hi_x and lo_y <= p["y"] <= hi_y]

    raw = json.load(open(args.gt_json))
    gt_by = {}
    for g in (raw["instances"] if isinstance(raw, dict) else raw):
        name = REPLICA2COCO.get(g.get("cls") or g.get("class"))
        if name:
            gt_by.setdefault(name, []).append((float(g["x"]), float(g["y"])))

    robots = split_robots(pts, args.robots)
    print(f"{len(pts)} observations -> {args.robots} robots "
          f"({[len(r) for r in robots]} each); shared world frame ASSUMED "
          f"(alignment is orthogonal to every join below)\n")

    xs = [p["x"] for p in pts]
    ys = [p["y"] for p in pts]
    ext = [min(xs) - 1, max(xs) + 1, min(ys) - 1, max(ys) + 1]
    enc = ClassroomEncoders(args.hd_dim, 0, args.length_scale, 20.0)
    gx = np.linspace(ext[0], ext[1], args.grid)
    gy = np.linspace(ext[2], ext[3], args.grid)
    G = np.empty((args.grid ** 2, args.hd_dim), np.complex64)
    k = 0
    for yy in gy:
        for xx in gx:
            G[k] = enc.ctx_pos(float(xx), float(yy)).values.astype(np.complex64)
            k += 1
    classes = sorted({p["cls"] for p in pts})
    sem = class_phasors(classes, args.hd_dim)

    def score(get_answers):
        r1 = []
        for cname in sorted(gt_by):
            targets = np.array(gt_by[cname])
            # a GT class never detected anywhere is a MISS for every
            # method, not a skip — skipping would flatter R@1
            peaks = get_answers(cname) if cname in sem else []
            if not peaks:
                r1.append(False)
                continue
            d1 = float(np.min(np.hypot(*(targets - np.array(peaks[0])).T)))
            r1.append(d1 <= args.radius)
        return float(np.mean(r1))

    rows = []

    # ---- vsa: per-robot capped traces, join by addition -------------------
    parts = [build_trace(cap_per_class(r, args.max_per_class, seed=11 + i),
                         enc, sem, args.hd_dim)
             for i, r in enumerate(robots)]
    M_join = sum(parts)
    M_joint = build_trace(sum([cap_per_class(r, args.max_per_class, seed=11 + i)
                               for i, r in enumerate(robots)], []),
                          enc, sem, args.hd_dim)
    exact = float(np.abs(M_join - M_joint).max())
    Mn = M_join / max(np.abs(M_join).max(), 1e-12)

    def vsa_ans(cname):
        F = ((Mn / sem[cname])[None, :] @ np.conj(G).T).real[0]
        return field_peaks(F, gx, gy, args.grid)

    rows.append(dict(method="VSA trace + addition",
                     bytes_per_robot=args.hd_dim * 8,
                     join_op="element-wise add",
                     assoc_decisions=0,
                     exact_vs_joint=f"{exact:.1e}",
                     truth_r1=score(vsa_ans)))

    # ---- instances: per-robot lists, associative merge --------------------
    lists = [clusters_of(r, args.cluster_radius, args.min_count)
             for r in robots]
    merged, decisions = assoc_merge(lists, args.assoc_radius)
    pooled = clusters_of(pts, args.cluster_radius, args.min_count)
    # association quality: instance count drift vs pooled-offline clustering
    drift = {}
    for src, tag in ((merged, "merged"), (pooled, "pooled")):
        for m in src:
            key = m["class"]
            drift.setdefault(key, [0, 0])
            drift[key][0 if tag == "merged" else 1] += 1
    n_mismatch = sum(1 for v in drift.values() if v[0] != v[1])
    by_m = {}
    for m in merged:
        by_m.setdefault(m["class"], []).append(m)
    inst_bytes = int(np.mean([len(l) for l in lists])) * 16

    def inst_ans(cname):
        v = sorted(by_m.get(cname, []), key=lambda m: -m["count"])[:3]
        return [(m["x"], m["y"]) for m in v]

    rows.append(dict(method="instance lists + association",
                     bytes_per_robot=inst_bytes,
                     join_op=f"greedy assoc (r={args.assoc_radius})",
                     assoc_decisions=decisions,
                     exact_vs_joint=f"{n_mismatch} classes drift vs pooled",
                     truth_r1=score(inst_ans)))

    # ---- gridmap: per-class count grids, join by addition -----------------
    gsz = 48
    ggx = np.linspace(ext[0], ext[1], gsz)
    ggy = np.linspace(ext[2], ext[3], gsz)
    grids = []
    for r in robots:
        g = {}
        for p in r:
            ix = int(np.clip(np.searchsorted(ggx, p["x"]), 0, gsz - 1))
            iy = int(np.clip(np.searchsorted(ggy, p["y"]), 0, gsz - 1))
            g.setdefault(p["cls"], np.zeros((gsz, gsz), np.float32))
            g[p["cls"]][iy, ix] += 1
        grids.append(g)
    joined = {}
    for g in grids:
        for c, arr in g.items():
            joined[c] = joined.get(c, 0) + arr
    grid_bytes = len(classes) * gsz * gsz * 2      # uint16 counts

    def grid_ans(cname):
        if cname not in joined:
            return []
        arr = joined[cname]
        flat = arr.ravel()
        order = np.argsort(flat)[::-1][:3]
        return [(float(ggx[o % gsz]), float(ggy[o // gsz]))
                for o in order if flat[o] > 0]

    rows.append(dict(method="per-class count grids + addition",
                     bytes_per_robot=grid_bytes,
                     join_op="cell-wise add",
                     assoc_decisions=0,
                     exact_vs_joint="exact (by construction)",
                     truth_r1=score(grid_ans)))

    # ---- concat: raw observations ----------------------------------------
    by_c = {}
    for p in pts:
        by_c.setdefault(p["cls"], []).append(p)

    def concat_ans(cname):
        ps = sorted(by_c.get(cname, []), key=lambda p: -p["conf"])
        picks = []
        for p in ps:
            if all(np.hypot(p["x"] - q[0], p["y"] - q[1]) > 0.5 for q in picks):
                picks.append((p["x"], p["y"]))
            if len(picks) == 3:
                break
        return picks

    rows.append(dict(method="raw observation union",
                     bytes_per_robot=int(np.mean([len(r) for r in robots])) * 12,
                     join_op="concatenate",
                     assoc_decisions=0,
                     exact_vs_joint="exact (trivially)",
                     truth_r1=score(concat_ans)))

    hdr = (f"{'method':<32}{'bytes/robot':>12}{'join':>24}"
           f"{'assoc':>7}{'truth-R@1':>11}   exact-vs-joint")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['method']:<32}{r['bytes_per_robot']/1024:>10.1f}KB"
              f"{r['join_op']:>24}{r['assoc_decisions']:>7}"
              f"{r['truth_r1']:>11.0%}   {r['exact_vs_joint']}")

    out = os.path.join(out_dir, f"merge_comparison_k{args.robots}.json")
    with open(out, "w") as f:
        json.dump(dict(robots=args.robots, rows=rows,
                       n_obs=len(pts), n_classes_gt=len(gt_by)), f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
