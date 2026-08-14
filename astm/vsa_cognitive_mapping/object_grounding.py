"""Object-grounding battery on a rendered RGB-D sequence (Replica room0).

The ConceptGraphs-comparable protocol scoped in tracker (ay): class queries
("where is the chair?") answered from a VSA object memory, scored at INSTANCE
level — success = a returned location within r metres of a ground-truth
instance of that class, reported as R@1/R@3 and success rate. No millimetres
anywhere: this is the precision class both scene graphs and we actually
occupy.

Pipeline (all CPU):
  1. localize   — YOLO detections (detections_crops.csv) + depth PNGs +
                  full 4x4 poses (traj.txt) -> world points per detection,
                  clustered per class into detected instances.
                  Replica render intrinsics fx=fy=600, cx=599.5, cy=339.5,
                  depth = png/6553.5 m (asserted against image size).
  2. map        — VSA object memory: SEM_class (x) S(x,y) bundled over
                  detections into ONE trace; queries unbind the class and
                  read the position field's top-k well-separated peaks.
  3. score      — vs --gt-json ([{class,x,y}] in world floor coords) when
                  available (student exports it from the Replica semantic
                  mesh); otherwise the two-pseudo-session protocol: memory
                  from the first half of the walk, evaluated against
                  instances clustered from the second half (labelled
                  provisional). --merge-check adds the split-build-merge
                  equality test.

    python -m vsa_cognitive_mapping.object_grounding \
        --dataset vsa_cognitive_mapping/configs/replica_room0.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vsa_cognitive_mapping.classroom_pipeline import ClassroomEncoders  # noqa: E402
from vsa_cognitive_mapping.object_map import _cluster  # noqa: E402

FX = FY = 600.0
CX, CY = 599.5, 339.5
DEPTH_SCALE = 6553.5
IMG_WH = (1200, 680)

# Replica semantic classes -> the COCO vocabulary YOLO can emit. GT classes
# with no COCO counterpart (walls, blinds, rug, ...) are excluded from the
# YOLO-comparable battery and listed. stool->chair and table->dining table
# are judgement calls, stated openly (YOLO labels stools as chairs).
REPLICA2COCO = {
    "sofa": "couch", "chair": "chair", "stool": "chair",
    "table": "dining table", "desk": "dining table",
    "vase": "vase", "plant": "potted plant", "indoor-plant": "potted plant",
    "book": "book", "bowl": "bowl", "cup": "cup", "bottle": "bottle",
    "tv-screen": "tv", "tv_monitor": "tv", "monitor": "tv",
    "clock": "clock", "sink": "sink", "bike": "bicycle", "bench": "bench",
}


# ------------------------------------------------------------- localisation

def localize(cfg, out_dir):
    from PIL import Image

    img_dir = os.path.dirname(cfg["images"])
    traj = np.loadtxt(os.path.join(img_dir, "traj.txt")).reshape(-1, 4, 4)
    T = traj[:, :3, 3]
    var = T.var(axis=0)
    a, b = sorted(np.argsort(var)[-2:])

    dets_path = os.path.join(out_dir, "detections_crops.csv")
    rows = list(csv.DictReader(open(dets_path, newline="")))
    print(f"localizing {len(rows)} detections (floor axes {'xyz'[a]},{'xyz'[b]})")

    pts, depth_cache = [], {}
    n_ok = 0
    for r in rows:
        fi = int(r["frame_idx"])
        if fi not in depth_cache:
            dp = os.path.join(img_dir, f"depth{fi:06d}.png")
            if not os.path.exists(dp):
                continue
            depth_cache.clear()                    # one frame resident at a time
            depth_cache[fi] = np.asarray(Image.open(dp), np.float32) / DEPTH_SCALE
        d = depth_cache[fi]
        assert d.shape[1] == IMG_WH[0] and d.shape[0] == IMG_WH[1], \
            f"unexpected render size {d.shape} — intrinsics constants are wrong"
        x1, y1 = float(r["x1"]), float(r["y1"])
        x2, y2 = float(r["x2"]), float(r["y2"])
        u1, u2 = max(0, int(x1)), min(d.shape[1], int(x2))
        v1, v2 = max(0, int(y1)), min(d.shape[0], int(y2))
        if u2 <= u1 or v2 <= v1:
            continue
        patch = d[v1:v2, u1:u2]
        valid = patch[(patch > 0.1) & (patch < 10.0)]
        if valid.size < max(10, 0.02 * patch.size):
            continue
        z = float(np.median(valid))
        u, v = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        p_cam = np.array([(u - CX) / FX * z, (v - CY) / FY * z, z])
        M = traj[fi]
        p_world = M[:3, :3] @ p_cam + M[:3, 3]
        pts.append(dict(frame=fi, cls=r["class_name"],
                        conf=float(r["confidence"]), det=int(r["det_id"]),
                        x=float(p_world[a]), y=float(p_world[b])))
        n_ok += 1
    print(f"placed {n_ok}/{len(rows)} detections in world coordinates")
    out = os.path.join(out_dir, "object_points.json")
    with open(out, "w") as f:
        json.dump(dict(axes=["xyz"[a], "xyz"[b]], points=pts), f)
    print(f"wrote {out}")
    return pts


def clusters_of(pts, radius, min_count):
    per_class = {}
    for p in pts:
        per_class.setdefault(p["cls"], []).append((p["x"], p["y"], p["conf"]))
    out = []
    for cname, ps in per_class.items():
        for c in _cluster(ps, radius, min_count):
            c["class"] = cname
            out.append(c)
    return out


# ------------------------------------------------------------------ VSA map

def class_phasors(classes, hd, seed=7):
    rng = np.random.RandomState(seed)
    return {c: np.exp(1j * rng.uniform(-np.pi, np.pi, hd)) for c in classes}


def cap_per_class(pts, k, seed=11):
    """Bounded per-class insertion (reservoir-style): at most k detections of
    any one class enter the trace. Kills dominant-class crosstalk (couch at
    1150 detections drowns book at 5 — the chi law at class level) while
    keeping ONE trace and exact mergeability: each session caps its own
    stream, and the merged trace is the sum of the capped streams."""
    if k is None or k <= 0:
        return pts
    rng = np.random.RandomState(seed)
    by = {}
    for p in pts:
        by.setdefault(p["cls"], []).append(p)
    out = []
    for ps in by.values():
        if len(ps) > k:
            ps = [ps[i] for i in rng.choice(len(ps), k, replace=False)]
        out.extend(ps)
    return out


def build_trace(pts, enc, sem, hd):
    M = np.zeros(hd, np.complex128)
    for p in pts:
        M += sem[p["cls"]] * enc.ctx_pos(p["x"], p["y"]).values
    return M


def field_peaks(F, gx, gy, grid, k=3, sep=4):
    idx = np.argsort(F)[::-1]
    picks, taken = [], []
    for j in idx:
        c = np.array([j % grid, j // grid])
        if all(np.abs(c - t).max() > sep for t in taken):
            taken.append(c)
            picks.append((float(gx[j % grid]), float(gy[j // grid])))
        if len(picks) == k:
            break
    return picks


# ---------------------------------------- CLIP crop verification ---------

def verify_crops(cfg, out_dir, pts, agree_k=3, batch=32):
    """Independent second labeler: CLIP ViT-B/32 zero-shot over the classes
    present. A detection survives only if CLIP's top-k classes for its crop
    include YOLO's label — two independent labelers must agree before an
    observation enters the memory or the target set. Kills consistent
    hallucinations (the armchair-'toilet'). Cached per det_id."""
    import csv as _csv

    import torch
    from vsa_cognitive_mapping.sequences import load_sequence

    cache_path = os.path.join(out_dir, "crop_clip_verify.pt")
    rows = {int(r["det_id"]): r for r in _csv.DictReader(
        open(os.path.join(out_dir, "detections_crops.csv"), newline=""))}
    if os.path.exists(cache_path):
        cache = torch.load(cache_path, weights_only=False)
    else:
        from transformers import CLIPModel, CLIPProcessor
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model.eval().to(dev)
        print(f"CLIP verification on {dev}")
        classes = sorted({r["class_name"] for r in rows.values()})
        with torch.no_grad():
            t = proc(text=[f"a photo of a {c}" for c in classes],
                     return_tensors="pt", padding=True).to(dev)
            from vsa_cognitive_mapping.clip_compat import (image_features,
                                                           text_features)
            T = text_features(model, t)
            T = T / T.norm(dim=-1, keepdim=True)
        seq = load_sequence(cfg)
        pos = {int(f): i for i, f in enumerate(seq.frame_ids())}
        ids = sorted(rows)
        top = {}
        with torch.no_grad():
            for s in range(0, len(ids), batch):
                chunk = ids[s:s + batch]
                imgs = []
                for di in chunk:
                    r = rows[di]
                    img = seq.image(pos[int(r["frame_idx"])])
                    x1, y1 = float(r["x1"]), float(r["y1"])
                    x2, y2 = float(r["x2"]), float(r["y2"])
                    imgs.append(img.crop((max(0, x1), max(0, y1),
                                          min(img.size[0], x2),
                                          min(img.size[1], y2))))
                v = proc(images=imgs, return_tensors="pt").to(dev)
                V = image_features(model, v)
                V = V / V.norm(dim=-1, keepdim=True)
                sim = (V @ T.T).cpu().numpy()
                for j, di in enumerate(chunk):
                    order = np.argsort(sim[j])[::-1]
                    top[di] = [classes[o] for o in order[:5]]
                if (s // batch) % 20 == 0:
                    print(f"  CLIP verify {s}/{len(ids)}", flush=True)
        cache = dict(classes=classes, top=top)
        torch.save(cache, cache_path)
        print(f"cached CLIP verification -> {cache_path}")
    top = cache["top"]
    if any(p.get("det") is None for p in pts):
        # object_points.json predates the det field: rebuild the join by
        # sweeping the csv's (frame, class) queues in order
        from collections import defaultdict, deque
        q = defaultdict(deque)
        for di in sorted(rows):
            r = rows[di]
            q[(int(r["frame_idx"]), r["class_name"])].append(di)
        for p in pts:
            key = (p["frame"], p["cls"])
            p["det"] = q[key].popleft() if q[key] else None
    kept = [p for p in pts
            if p.get("det") is not None and p["det"] in top
            and p["cls"] in top[p["det"]][:agree_k]]
    print(f"CLIP verification (agree@top-{agree_k}): kept {len(kept)}/{len(pts)} "
          f"observations (dropped = YOLO label CLIP disagrees with)")
    return kept


# ------------------------- same-frontend backends (the fair comparison) --
# Every backend consumes the IDENTICAL observation stream, so differences
# between rows are attributable to the memory structure alone.

def backend_instances(mem_pts, cluster_radius, min_count):
    """Explicit instance list (cluster + lookup) — the object layer of a
    ConceptGraphs-style scene graph, minus their perception; also the
    frontend's ceiling. Answers = instance centroids, largest first."""
    by = {}
    for c in clusters_of(mem_pts, cluster_radius, min_count):
        by.setdefault(c["class"], []).append(c)
    ans = {c: [(i["x"], i["y"]) for i in
               sorted(v, key=lambda i: -i["count"])[:3]]
           for c, v in by.items()}
    n_inst = sum(len(v) for v in by.values())
    return ans, n_inst * 16                       # class id + x + y + count


def backend_knn_obs(mem_pts, sep=0.5):
    """Raw observation store: keep every placed detection; answers = the
    highest-confidence, mutually separated observations of the class."""
    by = {}
    for p in mem_pts:
        by.setdefault(p["cls"], []).append(p)
    ans = {}
    for c, ps in by.items():
        picks = []
        for p in sorted(ps, key=lambda p: -p["conf"]):
            if all(np.hypot(p["x"] - q[0], p["y"] - q[1]) > sep
                   for q in picks):
                picks.append((p["x"], p["y"]))
            if len(picks) == 3:
                break
        ans[c] = picks
    return ans, len(mem_pts) * 12                 # class id + x + y + conf


# --------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--gt-json", default=None,
                    help="[{class,x,y}] world floor coords; without it the "
                         "two-pseudo-session protocol is used (provisional)")
    ap.add_argument("--radius", type=float, default=1.0,
                    help="grounding success radius (metres)")
    ap.add_argument("--cluster-radius", type=float, default=0.6)
    ap.add_argument("--min-count", type=int, default=4)
    ap.add_argument("--hd-dim", type=int, default=4096)
    ap.add_argument("--grid", type=int, default=64)
    ap.add_argument("--length-scale", type=float, default=0.6)
    ap.add_argument("--merge-check", action="store_true")
    ap.add_argument("--relocalize", action="store_true")
    ap.add_argument("--room-margin", type=float, default=1.5,
                    help="drop points this many metres beyond the walked "
                         "extent (depth-through-glass phantoms); pass -1 "
                         "to disable")
    ap.add_argument("--max-per-class", type=int, default=0,
                    help="bounded per-class insertion: cap each class's "
                         "detections entering the trace (0 = no cap)")
    ap.add_argument("--verify-crops", action="store_true",
                    help="require CLIP (independent labeler) to agree with "
                         "YOLO's label before an observation is used")
    ap.add_argument("--verify-agree", type=int, default=3,
                    help="YOLO label must be in CLIP's top-k for the crop")
    args = ap.parse_args()

    cfg = json.load(open(args.dataset))
    out_dir = cfg["out_dir"]
    pts_path = os.path.join(out_dir, "object_points.json")
    if args.relocalize or not os.path.exists(pts_path):
        pts = localize(cfg, out_dir)
    else:
        pts = json.load(open(pts_path))["points"]
        print(f"loaded {len(pts)} placed detections from {pts_path}")

    # Depth through windows/mirrors places phantom points far outside the
    # room; drop anything > room-margin beyond the camera path's extent.
    if args.room_margin is not None:
        from vsa_cognitive_mapping.sequences import load_sequence
        seq = load_sequence(args.dataset)
        px, py, _ = seq.poses()
        lo_x, hi_x = px.min() - args.room_margin, px.max() + args.room_margin
        lo_y, hi_y = py.min() - args.room_margin, py.max() + args.room_margin
        n0 = len(pts)
        pts = [p for p in pts
               if lo_x <= p["x"] <= hi_x and lo_y <= p["y"] <= hi_y]
        print(f"room-margin filter (±{args.room_margin} m beyond the walked "
              f"extent): kept {len(pts)}/{n0} points")

    if args.verify_crops:
        pts = verify_crops(args.dataset, out_dir, pts, args.verify_agree)

    frames = sorted({p["frame"] for p in pts})
    half = frames[len(frames) // 2]
    sess_a = [p for p in pts if p["frame"] <= half]
    sess_b = [p for p in pts if p["frame"] > half]

    if args.gt_json:
        raw = json.load(open(args.gt_json))
        rows = raw["instances"] if isinstance(raw, dict) else raw
        gt, skipped = [], {}
        for g in rows:
            name = g.get("cls") or g.get("class")
            coco = REPLICA2COCO.get(name, name)
            det_classes = {p["cls"] for p in pts}
            if coco in det_classes or coco in REPLICA2COCO.values():
                gt.append(dict(cls=coco, x=g["x"], y=g["y"]))
            else:
                skipped[name] = skipped.get(name, 0) + 1
        print(f"GT: {len(gt)} COCO-comparable instances; excluded "
              f"(no COCO class): "
              f"{', '.join(f'{k}×{v}' for k, v in sorted(skipped.items()))}")
        mem_pts, gt_tag = pts, f"ground truth ({raw.get('source', 'gt json')})"
    else:
        gt = [dict(cls=c["class"], x=c["x"], y=c["y"])
              for c in clusters_of(sess_b, args.cluster_radius, args.min_count)]
        mem_pts, gt_tag = sess_a, ("PROVISIONAL: session-B clusters "
                                   "(first-half memory, second-half targets)")
    gt_by_class = {}
    for g in gt:
        gt_by_class.setdefault(g.get("cls") or g.get("class"), []).append(
            (float(g["x"]), float(g["y"])))

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
    mem_pts_raw = list(mem_pts)          # the shared observation stream
    mem_pts = cap_per_class(mem_pts, args.max_per_class)
    if args.max_per_class:
        print(f"bounded per-class insertion: cap {args.max_per_class} -> "
              f"{len(mem_pts)} detections in the trace")
    M = build_trace(mem_pts, enc, sem, args.hd_dim)
    M /= max(np.abs(M).max(), 1e-12)
    trace_bytes = args.hd_dim * 8

    mem_count = {}
    for p in mem_pts:
        mem_count[p["cls"]] = mem_count.get(p["cls"], 0) + 1

    # What the score MEANS depends on the target source. Without real GT the
    # protocol measures CONSISTENCY (does the memory ground the label where
    # the detector re-finds it?) — a consistent MISlabel (an armchair seen
    # as "toilet" in both halves) scores as consistent. Truth needs --gt-json.
    metric = "truth" if args.gt_json else "consistency"

    # ---- same-frontend backends: identical observation stream ------------
    def vsa_answers(cname):
        F = ((M / sem[cname])[None, :] @ np.conj(G).T).real[0]
        return field_peaks(F, gx, gy, args.grid)

    ans_inst, bytes_inst = backend_instances(mem_pts_raw, args.cluster_radius,
                                             args.min_count)
    ans_knn, bytes_knn = backend_knn_obs(mem_pts_raw)
    backends = [
        ("VSA trace (capped)", vsa_answers, args.hd_dim * 8,
         "O(1) add", "vector addition (exact)"),
        ("instance list [ceiling]", lambda c: ans_inst.get(c, []), bytes_inst,
         "re-cluster", "association problem"),
        ("raw observations", lambda c: ans_knn.get(c, []), bytes_knn,
         "append", "concatenate (grows)"),
    ]

    def score(get_answers):
        r1s, r3s, seen1, seen3, per = [], [], [], [], []
        for cname in sorted(gt_by_class):
            targets = np.array(gt_by_class[cname])
            # never-detected GT class = automatic miss (skipping flatters R@1)
            peaks = get_answers(cname) if cname in sem else []
            if peaks:
                d1 = float(np.min(np.hypot(*(targets - np.array(peaks[0])).T)))
                hit1 = d1 <= args.radius
                hit3 = any(np.min(np.hypot(*(targets - np.array(pk)).T))
                           <= args.radius for pk in peaks)
            else:
                d1, hit1, hit3 = float("inf"), False, False
            r1s.append(hit1)
            r3s.append(hit3)
            if mem_count.get(cname, 0) >= args.min_count:
                seen1.append(hit1)
                seen3.append(hit3)
            per.append(dict(cls=cname, n_gt=len(targets),
                            n_mem=mem_count.get(cname, 0),
                            r1=bool(hit1), r3=bool(hit3),
                            d1=(round(d1, 2) if np.isfinite(d1) else None),
                            peaks=[[round(v, 2) for v in pk] for pk in peaks]))
        return (float(np.mean(r1s)), float(np.mean(r3s)),
                float(np.mean(seen1)) if seen1 else None,
                float(np.mean(seen3)) if seen3 else None, per)

    print(f"\n=== grounding battery: {len(gt_by_class)} classes, success "
          f"radius {args.radius} m — METRIC: {metric.upper()}-R@k ===")
    print(f"targets: {gt_tag}")
    if metric == "consistency":
        print("NOTE: consistency-R@k scores agreement with the detector's own "
              "re-detections. A consistent mislabel scores as consistent; "
              "semantic truth requires --gt-json.")

    # per-class detail for the VSA backend (feeds the inspector)
    _, _, _, _, results = score(vsa_answers)
    hdr = (f"{'class':<16}{'n_gt':>5}{'n_mem':>7}{'R@1':>7}{'R@3':>7}"
           f"{'d@1 (m)':>9}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in results:
        print(f"{r['cls']:<16}{r['n_gt']:>5}{r['n_mem']:>7}{str(r['r1']):>7}"
              f"{str(r['r3']):>7}{r['d1'] if r['d1'] is not None else '—':>9}")

    # the fair-comparison table: same frontend, different backend
    print(f"\n=== same frontend, different backend ({metric}-R@k, "
          f"{len(mem_pts_raw)} observations in) ===")
    bhdr = (f"{'backend':<26}{'bytes':>10}{'R@1':>7}{'R@3':>7}"
            f"{'R@1 in-mem':>12}   update / merge")
    print(bhdr)
    print("-" * len(bhdr))
    backend_rows = []
    for name, fn, nbytes, upd, mrg in backends:
        r1, r3, s1, s3, _ = score(fn)
        backend_rows.append(dict(backend=name, bytes=int(nbytes), r1=r1, r3=r3,
                                 r1_in_mem=s1, r3_in_mem=s3, update=upd,
                                 merge=mrg))
        print(f"{name:<26}{nbytes/1024:>8.0f}KB{r1:>7.0%}{r3:>7.0%}"
              f"{(s1 if s1 is not None else 0):>12.0%}   {upd} / {mrg}")
    print(f"\nVSA object trace: {trace_bytes / 1024:.0f} KB "
          f"({len(mem_pts)} detections bundled, {len(classes)} classes)")

    merge = None
    if args.merge_check:
        ca = cap_per_class(sess_a, args.max_per_class)
        cb = cap_per_class(sess_b, args.max_per_class, seed=12)
        MA = build_trace(ca, enc, sem, args.hd_dim)
        MB = build_trace(cb, enc, sem, args.hd_dim)
        MJ = build_trace(ca + cb, enc, sem, args.hd_dim)
        exact = float(np.abs((MA + MB) - MJ).max())
        agree = 0
        for cname in sorted(gt_by_class):
            if cname not in sem:
                continue
            Fm = (((MA + MB) / max(np.abs(MA + MB).max(), 1e-12) / sem[cname])[None, :]
                  @ np.conj(G).T).real[0]
            Fj = ((MJ / max(np.abs(MJ).max(), 1e-12) / sem[cname])[None, :]
                  @ np.conj(G).T).real[0]
            agree += int(Fm.argmax() == Fj.argmax())
        merge = dict(exact=exact, agree=agree, n=len(gt_by_class))
        print(f"\n=== two-session merge ===\nsession A {len(sess_a)} dets, "
              f"B {len(sess_b)} dets: max |A+B − joint| = {exact:.2e}; "
              f"identical class-query answers {agree}/{len(gt_by_class)}")

    out = os.path.join(out_dir, "object_grounding.json")
    with open(out, "w") as f:
        json.dump(dict(gt=gt_tag, metric=metric, radius=args.radius,
                       backends=backend_rows, per_class_vsa=results,
                       trace_bytes=trace_bytes, n_obs=len(mem_pts_raw),
                       n_in_trace=len(mem_pts), merge=merge), f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
