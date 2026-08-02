"""Real-data classroom pipeline: Spot walk-through -> VSA associative memory.

Runs the ``neuromorphs/VSACognitiveMapping`` ``init-am`` classroom idea on the
actual dataset (``lorinachey/spot-telluride-workshop-dataset`` on HuggingFace:
a Boston Dynamics Spot classroom walk, D455 RGB + LIO-SAM odometry), using the
ported ``Phasor`` core in this package.

Stages (run in order; each consumes the previous stage's output):

    embed      YOLOv8n over every RGB frame -> detections.csv + embeddings.pt
    build      bind content to position / heading / time FPE contexts,
               bundle into three per-axis memories -> memory.pt
               (--place-mode object binds the semantic memories to the
                object's depth-deprojected world (x, y) instead of the
                robot's pose; --memory-name keeps builds side by side)
    evaluate   self-recall error per axis, in physical units -> evaluate.png
    demo       held-out frames -> RGB+boxes | position heatmap | heading polar,
               rendered to demo.gif
    build_now  decaying working memory M_now (per-class forgetting factor;
               --decay-per picks the clock: event / frame / metre / both,
               hybrid lambda_eff = lambda_time**dt * lambda_dist**dd),
               snapshots through the walk -> memory_now.pt
    query_now  M_now snapshots vs static map for one class -> comparison PNG
    compare_clocks  all four forgetting clocks on the same data -> one figure
               (decay_clocks_comparison.png)

Examples:
    python vsa_cognitive_mapping/classroom_pipeline.py embed
    python vsa_cognitive_mapping/classroom_pipeline.py build --subset stride --stride 3 --hd-dim 8192
    python vsa_cognitive_mapping/classroom_pipeline.py build --place-mode object --memory-name memory_object.pt
    python vsa_cognitive_mapping/classroom_pipeline.py evaluate
    python vsa_cognitive_mapping/classroom_pipeline.py demo --demo-frames 48
    python vsa_cognitive_mapping/classroom_pipeline.py build_now
    python vsa_cognitive_mapping/classroom_pipeline.py query_now --class-name person

Raw HF data stays in the HuggingFace cache (outside OneDrive); only compact
derived artifacts are written to ``--out-dir`` (default ``outputs/classroom``).

Heading uses a *circular* FPE base (integer frequencies, |k| <= 3) so that
ctx(psi + 2*pi) == ctx(psi) exactly -- the same trick as the upstream
``Phasor(circular=True)``; our ported core predates that flag so the base is
constructed here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vsa_cognitive_mapping.vsa import (  # noqa: E402
    Phasor, random_project_to_phasor, pca_components,
)

DATASET = "lorinachey/spot-telluride-workshop-dataset"
DEFAULT_OUT = os.path.join("outputs", "classroom")

# RealSense D455 colour intrinsics (from camera_info); depth assumed registered.
# Mirrors object_map.py -- kept in sync by hand (repo convention).
K_FX, K_FY, K_CX, K_CY = 378.22, 377.92, 320.44, 249.25
# Approximate camera mount on Spot body (metres): a bit forward and up.
CAM_OFFSET = np.array([0.30, 0.0, 0.20])


# --------------------------------------------------------------------------
# Poses
# --------------------------------------------------------------------------

def quat_to_yaw(qx, qy, qz, qw):
    return np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def load_poses_interpolated(frame_ts: np.ndarray, repo: str = None,
                            odom_config: str = "odometry_lio_sam"):
    """LIO-SAM poses (~6 Hz) interpolated to the RGB frame timestamps.

    Returns (x, y, yaw) arrays aligned with frame_ts. Frames outside the pose
    time range are clamped to the nearest pose (np.interp behaviour).

    ``odom_config`` defaults to the classroom's; pass
    ``school_run1_odometry_lio_sam`` for the scale-up sequence. Callers that
    omit it keep the previous behaviour exactly."""
    from datasets import load_dataset
    odo = load_dataset(repo or DATASET, odom_config, split="train")
    ot = np.array(odo["timestamp_ns"], dtype=np.float64)
    order = np.argsort(ot)
    ot = ot[order]
    ox = np.array(odo["x"], dtype=np.float64)[order]
    oy = np.array(odo["y"], dtype=np.float64)[order]
    yaw = quat_to_yaw(np.array(odo["qx"], dtype=np.float64)[order],
                      np.array(odo["qy"], dtype=np.float64)[order],
                      np.array(odo["qz"], dtype=np.float64)[order],
                      np.array(odo["qw"], dtype=np.float64)[order])
    yaw_u = np.unwrap(yaw)  # interpolate on the unwrapped angle, wrap after
    ft = frame_ts.astype(np.float64)
    x = np.interp(ft, ot, ox)
    y = np.interp(ft, ot, oy)
    psi = np.interp(ft, ot, yaw_u)
    psi = np.arctan2(np.sin(psi), np.cos(psi))
    return x, y, psi


# --------------------------------------------------------------------------
# Context encoders (position / circular heading / time)
# --------------------------------------------------------------------------

def circular_base(dim: int, seed: int, max_freq: int = 3) -> Phasor:
    """Base phasor whose FPE is exactly 2*pi-periodic: each dimension's phase
    is a random *integer* frequency k in [-max_freq, max_freq] \\ {0}, so
    (base ** psi)_m = exp(i * k_m * psi). max_freq must stay < pi so the
    principal branch of the complex power preserves k."""
    rng = np.random.RandomState(seed)
    k = rng.randint(1, max_freq + 1, size=dim) * rng.choice([-1, 1], size=dim)
    return Phasor(data=np.exp(1j * k.astype(np.float64)))


class ClassroomEncoders:
    def __init__(self, hd_dim: int, seed: int, pos_length_scale: float, time_length_scale: float):
        self.Bx = Phasor(dim=hd_dim, seed=seed + 0)
        self.By = Phasor(dim=hd_dim, seed=seed + 1)
        self.Bhead = circular_base(hd_dim, seed + 2)
        self.Bt = Phasor(dim=hd_dim, seed=seed + 3)
        self.pos_l = pos_length_scale
        self.time_l = time_length_scale
        self.hd_dim = hd_dim

    def ctx_pos(self, x: float, y: float) -> Phasor:
        return (self.Bx ** float(x / self.pos_l)) * (self.By ** float(y / self.pos_l))

    def ctx_head(self, psi: float) -> Phasor:
        return self.Bhead ** float(psi)

    def ctx_time(self, t: float) -> Phasor:
        return self.Bt ** float(t / self.time_l)


# --------------------------------------------------------------------------
# embed stage
# --------------------------------------------------------------------------

def cmd_embed(args):
    from datasets import load_dataset
    from ultralytics import YOLO

    os.makedirs(args.out_dir, exist_ok=True)
    ds = load_dataset(DATASET, "rgb_d455", split="train")
    # capture order = timestamp order (frame_idx is not monotonic under it)
    ts = np.array(ds["timestamp_ns"], dtype=np.int64)
    order = np.argsort(ts)
    idx_all = np.array(ds["frame_idx"], dtype=np.int64)

    keep = order[::args.stride]
    print(f"{len(ds)} frames total, embedding every {args.stride} -> {len(keep)} frames")

    model = YOLO("yolov8n.pt")  # downloads once (~6 MB) to the CWD
    det_rows, embs, kept_ts, kept_idx = [], [], [], []
    for n, i in enumerate(keep):
        img = ds[int(i)]["image"]  # PIL RGB
        res = model.predict(img, verbose=False)[0]
        emb = model.embed(img, verbose=False)[0]
        for b in res.boxes:
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
            det_rows.append([int(idx_all[i]), int(ts[i]), int(b.cls),
                             model.names[int(b.cls)], float(b.conf), x1, y1, x2, y2])
        embs.append(emb.cpu().float())
        kept_ts.append(int(ts[i])); kept_idx.append(int(idx_all[i]))
        if n % 50 == 0:
            print(f"  {n}/{len(keep)}  dets_so_far={len(det_rows)}")

    emb_t = torch.stack(embs)
    torch.save({"frame_idx": torch.tensor(kept_idx), "timestamp_ns": torch.tensor(kept_ts),
                "embedding": emb_t}, os.path.join(args.out_dir, "embeddings.pt"))
    with open(os.path.join(args.out_dir, "detections.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_idx", "timestamp_ns", "class_id", "class_name",
                    "confidence", "x1", "y1", "x2", "y2"])
        w.writerows(det_rows)
    print(f"saved {emb_t.shape} embeddings + {len(det_rows)} detections to {args.out_dir}")


def _effective_rank(X):
    """Participation ratio of the covariance spectrum, plus the 95%-variance
    component count. Both measure how many directions the features actually
    use -- a low number means the encoder is wasting its dimensions, which is
    what makes crosstalk expensive."""
    Xc = X - X.mean(axis=0, keepdims=True)
    lam = np.linalg.svd(Xc, compute_uv=False) ** 2
    if lam.sum() <= 0:
        return 0.0, 0
    pr = float(lam.sum() ** 2 / (lam ** 2).sum())
    csum = np.cumsum(lam) / lam.sum()
    return pr, int(np.searchsorted(csum, 0.95) + 1)


def _isotropy_report(name, X):
    """Mean off-diagonal cosine + effective rank. High mutual cosine means every
    stored key looks like every other one, which is exactly the condition that
    makes an associative memory blur."""
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    n = min(len(Xn), 1500)                      # cap the Gram at 1500x1500
    sel = np.linspace(0, len(Xn) - 1, n).astype(int)
    G = Xn[sel] @ Xn[sel].T
    off = G[~np.eye(n, dtype=bool)]
    pr, k95 = _effective_rank(X)
    print(f"  {name:<22} n={len(X):<6} dim={X.shape[1]:<5} "
          f"cos mean={off.mean():+.3f} p05={np.percentile(off, 5):+.3f} "
          f"p95={np.percentile(off, 95):+.3f}  eff-rank={pr:.1f}  k95={k95}")
    return {"n": int(len(X)), "dim": int(X.shape[1]), "cos_mean": float(off.mean()),
            "cos_p95": float(np.percentile(off, 95)), "eff_rank": pr, "k95": k95}


def cmd_embed_crops(args):
    """Per-DETECTION appearance features.

    ``cmd_embed`` calls ``model.embed(img)`` once per *frame*, so every
    detection in a frame shares one vector. That is fine for "where am I", but
    it means there is no per-object appearance key anywhere in the pipeline --
    so a partial-cue query ("find *this* drill", given a crop) degenerates into
    a 38-class COCO label. This stage crops each YOLO box out of its frame and
    embeds the crop, giving one appearance vector per detection.

    Writes ``crop_embeddings.pt`` and a self-consistent ``detections_crops.csv``
    whose row order IS the embedding row order, so ``det_id`` is a positional
    join key. Prints an isotropy comparison against the frame-level features,
    because whether crops are worth the compute is exactly that question.
    """
    from datasets import load_dataset
    from ultralytics import YOLO

    # --dataset routes through the Sequence abstraction so any configured
    # sequence works; the older --repo/--rgb-config flags still work for the
    # HuggingFace case.
    seq = None
    if getattr(args, "dataset", None):
        from vsa_cognitive_mapping.sequences import load_sequence
        seq = load_sequence(args.dataset)
        if not args.out_dir or args.out_dir == DEFAULT_OUT:
            args.out_dir = seq.out_dir
    os.makedirs(args.out_dir, exist_ok=True)

    if seq is not None:
        n_all = len(seq)
        ts_sorted = (seq.timestamps_s() * 1e9).astype(np.int64)
        ids_sorted = seq.frame_ids()
        order = np.arange(n_all)              # Sequence is already time-ordered
        ts = ts_sorted
        idx_all = ids_sorted
        src = f"{seq.name}"
    else:
        ds = load_dataset(args.repo, args.rgb_config, split="train")
        ts = np.array(ds["timestamp_ns"], dtype=np.int64)
        order = np.argsort(ts)
        idx_all = np.array(ds["frame_idx"], dtype=np.int64)
        n_all = len(ds)
        src = f"{args.repo} / {args.rgb_config}"
    keep = order[::args.stride]
    print(f"{src}: {n_all} frames total, cropping every {args.stride} -> {len(keep)} frames")

    model = YOLO("yolov8n.pt")
    rows, crop_embs, frame_embs = [], [], []
    n_tiny = 0

    def _image(row):
        return seq.image(int(row)) if seq is not None else ds[int(row)]["image"]

    for n, i in enumerate(keep):
        img = _image(i)                                 # PIL RGB
        W, H = img.size
        res = model.predict(img, verbose=False)[0]
        if args.keep_frame_embed:
            frame_embs.append(model.embed(img, verbose=False)[0].cpu().float())

        crops, meta = [], []
        for b in res.boxes:
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
            bw, bh = x2 - x1, y2 - y1
            pad = args.pad_frac * max(bw, bh)
            cx1 = max(0, int(round(x1 - pad))); cy1 = max(0, int(round(y1 - pad)))
            cx2 = min(W, int(round(x2 + pad))); cy2 = min(H, int(round(y2 + pad)))
            if cx2 - cx1 < 2 or cy2 - cy1 < 2:          # degenerate, skip entirely
                continue
            tiny = min(bw, bh) < args.min_side
            n_tiny += int(tiny)
            crops.append(img.crop((cx1, cy1, cx2, cy2)))
            meta.append([int(idx_all[i]), int(ts[i]), int(b.cls), model.names[int(b.cls)],
                         float(b.conf), x1, y1, x2, y2, bw * bh / float(W * H), int(tiny)])

        if crops:
            embs = model.embed(crops, verbose=False)     # one forward per crop, batched
            for e in embs:
                crop_embs.append(e.cpu().float())
            rows.extend(meta)

        if n % 25 == 0:
            print(f"  frame {n}/{len(keep)}  crops_so_far={len(rows)}", flush=True)

    if not rows:
        raise SystemExit("no detections produced any usable crop")

    C = torch.stack(crop_embs)
    out = {"det_id": torch.arange(len(rows)),
           "frame_idx": torch.tensor([r[0] for r in rows]),
           "timestamp_ns": torch.tensor([r[1] for r in rows]),
           "class_id": torch.tensor([r[2] for r in rows]),
           "confidence": torch.tensor([r[4] for r in rows], dtype=torch.float32),
           "area_frac": torch.tensor([r[9] for r in rows], dtype=torch.float32),
           "tiny": torch.tensor([r[10] for r in rows]),
           "embedding": C,
           "meta": {"stride": args.stride, "pad_frac": args.pad_frac,
                    "min_side": args.min_side, "model": "yolov8n.pt"}}
    torch.save(out, os.path.join(args.out_dir, args.crops_name))

    with open(os.path.join(args.out_dir, "detections_crops.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["det_id", "frame_idx", "timestamp_ns", "class_id", "class_name",
                    "confidence", "x1", "y1", "x2", "y2", "area_frac", "tiny"])
        for k, r in enumerate(rows):
            w.writerow([k] + r)

    print(f"\nsaved {tuple(C.shape)} crop embeddings + {len(rows)} detections "
          f"({n_tiny} flagged tiny, min_side<{args.min_side}px) to {args.out_dir}")

    # ---- does this actually buy anything? -------------------------------
    print("\nisotropy (lower cos mean / higher eff-rank is better):")
    Cn = C.numpy()
    rep = {"crop": _isotropy_report("crop (per detection)", Cn)}
    if frame_embs:
        F = torch.stack(frame_embs).numpy()
        rep["frame"] = _isotropy_report("frame (per frame)", F)
    else:
        fp = os.path.join(args.out_dir, "embeddings.pt")
        if os.path.exists(fp):
            F = torch.load(fp, weights_only=True)["embedding"].numpy()
            rep["frame"] = _isotropy_report("frame (existing file)", F)

    # class separability: do crops of different classes actually differ?
    cid = np.array([r[2] for r in rows])
    Cu = Cn / (np.linalg.norm(Cn, axis=1, keepdims=True) + 1e-12)
    within, between, seen = [], [], [c for c in np.unique(cid) if (cid == c).sum() >= 2]
    for c in seen[:20]:
        a = Cu[cid == c][:120]
        b = Cu[cid != c][np.linspace(0, (cid != c).sum() - 1, min(240, (cid != c).sum())).astype(int)]
        within.append(float((a @ a.T)[~np.eye(len(a), dtype=bool)].mean()))
        between.append(float((a @ b.T).mean()))
    if within:
        rep["within_class_cos"] = float(np.mean(within))
        rep["between_class_cos"] = float(np.mean(between))
        print(f"  within-class cos={np.mean(within):+.3f}  "
              f"between-class cos={np.mean(between):+.3f}  "
              f"margin={np.mean(within) - np.mean(between):+.3f}  "
              f"({len(seen)} classes with >=2 detections)")
    with open(os.path.join(args.out_dir, "crop_isotropy.json"), "w") as f:
        json.dump(rep, f, indent=2)


# --------------------------------------------------------------------------
# build stage
# --------------------------------------------------------------------------

def _load_embeddings(out_dir):
    d = torch.load(os.path.join(out_dir, "embeddings.pt"), weights_only=True)
    return d["frame_idx"].numpy(), d["timestamp_ns"].numpy(), d["embedding"].numpy()


def _class_seed(name: str) -> int:
    """Deterministic per-class seed. NOT Python's ``hash()`` — that is salted
    per process (PYTHONHASHSEED), so build and query_class (separate processes)
    would draw different phasors for the same class and the query would fail."""
    return int(hashlib.md5(name.encode()).hexdigest(), 16) % 10_000_000


def _load_detections_by_ts(out_dir):
    """Group YOLO detections by frame timestamp: ts -> [(class_name, conf), ...]."""
    by_ts = defaultdict(list)
    with open(os.path.join(out_dir, "detections.csv")) as f:
        for r in csv.DictReader(f):
            by_ts[int(r["timestamp_ns"])].append((r["class_name"], float(r["confidence"])))
    return by_ts


def _load_detections_boxes(out_dir):
    """Group detections WITH boxes by frame timestamp:
    ts -> [(class_name, conf, x1, y1, x2, y2), ...]. Reads the csv in file
    order, so each per-ts list is parallel to _load_detections_by_ts."""
    by_ts = defaultdict(list)
    with open(os.path.join(out_dir, "detections.csv")) as f:
        for r in csv.DictReader(f):
            by_ts[int(r["timestamp_ns"])].append(
                (r["class_name"], float(r["confidence"]), float(r["x1"]),
                 float(r["y1"]), float(r["x2"]), float(r["y2"])))
    return by_ts


def _pixel_to_world(u, v, z_m, px, py, yaw):
    """Box-centre pixel + depth -> world (x, y, z) via optical->body->world.
    Copied from object_map.py (same intrinsics + camera offset)."""
    X = (u - K_CX) / K_FX * z_m
    Y = (v - K_CY) / K_FY * z_m
    Z = z_m
    # optical (x right, y down, z fwd) -> body (x fwd, y left, z up)
    body = np.array([Z, -X, -Y]) + CAM_OFFSET
    c, s = np.cos(yaw), np.sin(yaw)
    xw = px + c * body[0] - s * body[1]
    yw = py + s * body[0] + c * body[1]
    return xw, yw, body[2]


def _localize_detections(out_dir, pose_of_ts):
    """Metric world (x, y) for each detection via median box depth, mirroring
    object_map.py's localize stage: nearest depth frame per detection (grouped
    so each depth image decodes once), median of valid depths in the box
    (0.2-8 m, min pixel count), deprojected through the D455 intrinsics and
    the robot pose + camera offset.

    pose_of_ts: {frame_ts -> (robot_x, robot_y, yaw)}; only these timestamps
    are localized. Returns (xy_by_ts, n_ok, n_fallback) where xy_by_ts[ts][k]
    is (x, y) or None (no valid depth -> caller falls back to robot pose),
    parallel to _load_detections_by_ts / _load_detections_boxes ordering."""
    from datasets import load_dataset

    boxes_by_ts = _load_detections_boxes(out_dir)
    items = []  # (ts, k, x1, y1, x2, y2)
    for t, lst in boxes_by_ts.items():
        if t not in pose_of_ts:
            continue
        for k, (_c, _conf, x1, y1, x2, y2) in enumerate(lst):
            items.append((t, k, x1, y1, x2, y2))
    xy_by_ts = {t: [None] * len(lst) for t, lst in boxes_by_ts.items()
                if t in pose_of_ts}
    if not items:
        return xy_by_ts, 0, 0
    det_ts = np.array([it[0] for it in items], np.int64)

    depth = load_dataset(DATASET, "depth_d455", split="train")
    dts = np.array(depth["timestamp_ns"], np.int64)
    dorder = np.argsort(dts)
    dts_sorted = dts[dorder]

    # nearest depth row per detection, grouped so each depth frame decodes once
    idx = np.searchsorted(dts_sorted, det_ts)
    idx = np.clip(idx, 0, len(dts_sorted) - 1)
    left = np.clip(idx - 1, 0, len(dts_sorted) - 1)
    take_left = np.abs(dts_sorted[left] - det_ts) < np.abs(dts_sorted[idx] - det_ts)
    nearest = np.where(take_left, left, idx)
    det_depth_row = dorder[nearest]

    by_frame = defaultdict(list)
    for di, drow in enumerate(det_depth_row):
        by_frame[int(drow)].append(di)

    n_ok = 0
    print(f"localizing {len(items)} detections over {len(by_frame)} depth frames ...")
    for fi, (drow, det_ids) in enumerate(by_frame.items()):
        dimg = np.array(depth[drow]["depth"]).astype(np.float32)  # uint16 mm
        for di in det_ids:
            t, k, x1, y1, x2, y2 = items[di]
            u1, u2 = int(max(0, x1)), int(min(dimg.shape[1], x2))
            v1, v2 = int(max(0, y1)), int(min(dimg.shape[0], y2))
            if u2 <= u1 or v2 <= v1:
                continue
            patch = dimg[v1:v2, u1:u2]
            valid = patch[(patch > 200) & (patch < 8000)]  # 0.2 - 8 m
            if valid.size < max(10, 0.02 * patch.size):
                continue
            z_m = float(np.median(valid)) / 1000.0
            u, v = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
            px, py, yaw = pose_of_ts[t]
            xw, yw, _zw = _pixel_to_world(u, v, z_m, px, py, yaw)
            xy_by_ts[t][k] = (float(xw), float(yw))
            n_ok += 1
        if fi % 200 == 0:
            print(f"  frame {fi}/{len(by_frame)}  placed={n_ok}")
    return xy_by_ts, n_ok, len(items) - n_ok


def _content_phasors(emb: np.ndarray, hd_dim: int, seed: int, pca_whiten: bool):
    x = emb
    if pca_whiten:
        k = min(64, x.shape[1], x.shape[0] - 1)
        x, _ = pca_components(x.astype(np.float64), n_components=k)
    z, _ = random_project_to_phasor(torch.from_numpy(np.ascontiguousarray(x)).float(),
                                    d=hd_dim, seed=seed)
    return z.numpy().astype(np.complex128)


def cmd_build(args):
    frame_idx, ts, emb = _load_embeddings(args.out_dir)
    N = len(ts)
    x, y, psi = load_poses_interpolated(ts)
    t_row = np.arange(N, dtype=np.float64)

    if args.subset == "all":
        S = np.arange(N)
    else:  # stride train/val split
        S = np.arange(0, N, args.subset_stride)
    print(f"bundling {len(S)}/{N} frames (subset={args.subset}) at hd_dim={args.hd_dim}")

    content = _content_phasors(emb, args.hd_dim, args.seed, args.pca_whiten)
    enc = ClassroomEncoders(args.hd_dim, args.seed + 100, args.pos_length_scale, args.time_length_scale)

    mems = {}
    for axis, ctx_fn, vals in [
        ("position", lambda n: enc.ctx_pos(x[n], y[n]), None),
        ("heading", lambda n: enc.ctx_head(psi[n]), None),
        ("time", lambda n: enc.ctx_time(t_row[n]), None),
    ]:
        acc = np.zeros(args.hd_dim, dtype=np.complex128)
        for n in S:
            acc += content[n] * ctx_fn(int(n)).values
        mems[axis] = acc / len(S)

    # ---- object-centric semantic memory: M_sem = sum conf * SEM_class (x) ctx_pos --
    # --place-mode robot  (default): binds each YOLO class label to the *robot's*
    #   pose when it saw that object. Answers "from where were <class>es seen?".
    # --place-mode object: binds the class to the OBJECT's own metric (x, y),
    #   deprojected from median box depth + D455 intrinsics + robot pose (same
    #   math as object_map.py). Detections without valid depth fall back to the
    #   robot pose. Answers "where ARE the <class>es?".
    # Kept as an independent memory so class-unbind is trivial -- no need to
    # factorize a position/time-bound trace with a resonator network.
    dets_by_ts = _load_detections_by_ts(args.out_dir)
    classes = sorted({c for lst in dets_by_ts.values() for c, _ in lst})
    codebook = {c: Phasor(dim=args.hd_dim, seed=_class_seed(c)) for c in classes}
    xy_by_ts = None
    if args.place_mode == "object":
        pose_of_ts = {int(ts[n]): (float(x[n]), float(y[n]), float(psi[n])) for n in S}
        xy_by_ts, n_ok, n_fb = _localize_detections(args.out_dir, pose_of_ts)
        frac = n_fb / max(1, n_ok + n_fb)
        print(f"object placement: {n_ok} depth-localized, {n_fb} fell back to "
              f"robot pose ({100 * frac:.1f}%)")
    acc = np.zeros(args.hd_dim, dtype=np.complex128)
    # conjunctive event trace: class (x) place (x) time, all in ONE product.
    # Conditional queries need only plain unbinding (supply all factors but
    # one); a resonator is only needed to recover 2+ unknowns at once.
    acc_ev = np.zeros(args.hd_dim, dtype=np.complex128)
    wsum, n_det, n_fallback = 0.0, 0, 0
    for n in S:
        tsn = int(ts[n])
        ctx_robot = enc.ctx_pos(x[n], y[n]).values
        ctx_t = enc.ctx_time(t_row[n]).values
        for k, (cname, conf) in enumerate(dets_by_ts.get(tsn, [])):
            if xy_by_ts is not None and xy_by_ts[tsn][k] is not None:
                ox, oy = xy_by_ts[tsn][k]
                ctx = enc.ctx_pos(ox, oy).values
            else:
                ctx = ctx_robot
                if xy_by_ts is not None:
                    n_fallback += 1
            acc += conf * codebook[cname].values * ctx
            acc_ev += conf * codebook[cname].values * (ctx * ctx_t)
            wsum += conf
            n_det += 1
    mems["semantic_position"] = acc / max(wsum, 1e-9)
    mems["semantic_event"] = acc_ev / max(wsum, 1e-9)
    if xy_by_ts is not None:
        print(f"  bundled fallback fraction: {n_fallback}/{n_det} "
              f"({100 * n_fallback / max(1, n_det):.1f}%)")

    torch.save({
        "memories": {k: torch.from_numpy(v) for k, v in mems.items()},
        "content_S": torch.from_numpy(content[S]),
        "S": torch.from_numpy(S),
        "semantic_codebook": {c: torch.from_numpy(p.values) for c, p in codebook.items()},
        "truth": {"x": torch.from_numpy(x), "y": torch.from_numpy(y),
                  "yaw": torch.from_numpy(psi), "t_row": torch.from_numpy(t_row)},
        "config": {"hd_dim": args.hd_dim, "seed": args.seed,
                   "pos_length_scale": args.pos_length_scale,
                   "time_length_scale": args.time_length_scale,
                   "pca_whiten": args.pca_whiten, "subset": args.subset,
                   "subset_stride": args.subset_stride,
                   "place_mode": args.place_mode},
    }, os.path.join(args.out_dir, args.memory_name))
    print(f"saved {args.memory_name} (3 axis + semantic_position + semantic_event "
          f"memories, |S|={len(S)}, place_mode={args.place_mode}) to {args.out_dir}")
    print(f"  semantic map: {n_det} detections over {len(classes)} classes bundled")


def _load_memory(out_dir, memory_name="memory.pt"):
    d = torch.load(os.path.join(out_dir, memory_name), weights_only=True)
    cfg = d["config"]
    mems = {k: v.numpy() for k, v in d["memories"].items()}
    return d, cfg, mems


def _sims(residual: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    return np.real(codebook @ np.conj(residual)) / residual.shape[0]


# --------------------------------------------------------------------------
# evaluate stage
# --------------------------------------------------------------------------

def cmd_evaluate(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d, cfg, mems = _load_memory(args.out_dir, args.memory_name)
    S = d["S"].numpy()
    codebook = d["content_S"].numpy()
    x = d["truth"]["x"].numpy(); y = d["truth"]["y"].numpy()
    psi = d["truth"]["yaw"].numpy(); t_row = d["truth"]["t_row"].numpy()
    enc = ClassroomEncoders(cfg["hd_dim"], cfg["seed"] + 100,
                            cfg["pos_length_scale"], cfg["time_length_scale"])

    axes_spec = {
        "position": (lambda n: enc.ctx_pos(x[n], y[n]),
                     lambda a, b: float(np.hypot(x[a] - x[b], y[a] - y[b])), "m"),
        "heading": (lambda n: enc.ctx_head(psi[n]),
                    lambda a, b: float(abs(np.arctan2(np.sin(psi[a] - psi[b]),
                                                      np.cos(psi[a] - psi[b])))), "rad"),
        "time": (lambda n: enc.ctx_time(t_row[n]),
                 lambda a, b: float(abs(t_row[a] - t_row[b])), "frames"),
    }

    results, err_by_axis = {}, {}
    for axis, (ctx_fn, dist, unit) in axes_spec.items():
        errs = []
        for n in S:
            residual = mems[axis] / ctx_fn(int(n)).values
            best = S[int(np.argmax(_sims(residual, codebook)))]
            errs.append(dist(int(n), int(best)))
        errs = np.array(errs)
        results[axis] = {"L1": float(np.mean(np.abs(errs))),
                         "L2": float(np.sqrt(np.mean(errs ** 2))), "unit": unit}
        err_by_axis[axis] = errs
        print(f"{axis:>9}:  L1={results[axis]['L1']:.3f} {unit}   L2={results[axis]['L2']:.3f} {unit}")

    fig, axs = plt.subplots(1, 4, figsize=(17, 4.0), constrained_layout=True)
    axs[0].plot(x, y, "-", color="0.8", lw=1)
    sc = axs[0].scatter(x[S], y[S], c=err_by_axis["position"], cmap="viridis", s=12)
    axs[0].set_title("position self-recall error over the walk")
    axs[0].set_xlabel("x (m)"); axs[0].set_ylabel("y (m)"); axs[0].set_aspect("equal", "box")
    fig.colorbar(sc, ax=axs[0], fraction=0.046, label="err (m)")
    for ax, axis in zip(axs[1:], ["position", "heading", "time"]):
        ax.hist(err_by_axis[axis], bins=40, color="#0d7d88", alpha=0.85)
        r = results[axis]
        ax.set_title(f"{axis}\nL1={r['L1']:.2f} {r['unit']} · L2={r['L2']:.2f} {r['unit']}")
        ax.set_xlabel(f"error ({r['unit']})"); ax.set_ylabel("count")
    path = os.path.join(args.out_dir, "evaluate.png")
    fig.savefig(path, dpi=130)
    print(f"saved {path}")


# --------------------------------------------------------------------------
# demo stage
# --------------------------------------------------------------------------

def cmd_demo(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from datasets import load_dataset

    d, cfg, mems = _load_memory(args.out_dir, args.memory_name)
    S = d["S"].numpy()
    codebook = d["content_S"].numpy()
    x = d["truth"]["x"].numpy(); y = d["truth"]["y"].numpy(); psi = d["truth"]["yaw"].numpy()
    frame_idx, ts, emb = _load_embeddings(args.out_dir)
    N = len(ts)
    enc = ClassroomEncoders(cfg["hd_dim"], cfg["seed"] + 100,
                            cfg["pos_length_scale"], cfg["time_length_scale"])
    content = _content_phasors(emb, cfg["hd_dim"], cfg["seed"], cfg["pca_whiten"])

    held = np.setdiff1d(np.arange(N), S)
    if len(held) == 0:
        held = S  # subset=all -> just replay stored frames
    sel = held[np.linspace(0, len(held) - 1, min(args.demo_frames, len(held))).astype(int)]
    print(f"demo over {len(sel)} held-out frames (of {len(held)})")

    # position grid contexts (precomputed once)
    pad = 0.5
    gx = np.linspace(x.min() - pad, x.max() + pad, args.grid)
    gy = np.linspace(y.min() - pad, y.max() + pad, args.grid)
    grid = np.empty((len(gy), len(gx), cfg["hd_dim"]), dtype=np.complex64)
    for j, yy in enumerate(gy):
        for i, xx in enumerate(gx):
            grid[j, i] = enc.ctx_pos(xx, yy).values
    flat = grid.reshape(-1, cfg["hd_dim"])
    angles = np.linspace(-np.pi, np.pi, 121)
    head_ctx = np.stack([enc.ctx_head(a).values for a in angles])

    # RGB frames come from the HF cache (already downloaded by `embed`)
    ds = load_dataset(DATASET, "rgb_d455", split="train")
    ts_all = np.array(ds["timestamp_ns"], dtype=np.int64)
    row_of_ts = {int(t): i for i, t in enumerate(ts_all)}

    fig = plt.figure(figsize=(13.5, 4.6), constrained_layout=True)
    ax_im = fig.add_subplot(1, 3, 1)
    ax_hm = fig.add_subplot(1, 3, 2)
    ax_po = fig.add_subplot(1, 3, 3, projection="polar")

    def render(k):
        n = int(sel[k])
        for ax in (ax_im, ax_hm, ax_po):
            ax.clear()
        img = ds[row_of_ts[int(ts[n])]]["image"]
        ax_im.imshow(img); ax_im.set_axis_off()
        ax_im.set_title(f"held-out frame (row {n})", fontsize=10)

        heat = (np.real(flat @ np.conj(mems["position"] / content[n])) /
                cfg["hd_dim"]).reshape(len(gy), len(gx))
        lim = np.max(np.abs(heat)) or 1.0
        ax_hm.imshow(heat, origin="lower", cmap="coolwarm", vmin=-lim, vmax=lim,
                     extent=[gx[0], gx[-1], gy[0], gy[-1]], aspect="equal")
        ax_hm.plot(x, y, "-", color="0.3", lw=0.8, alpha=0.6)
        ax_hm.scatter([x[n]], [y[n]], s=140, facecolor="none", edgecolor="black", linewidth=2)
        ax_hm.set_title("where have I seen this? (position)", fontsize=10)
        ax_hm.set_xlabel("x (m)"); ax_hm.set_ylabel("y (m)")

        pol = np.real(head_ctx @ np.conj(mems["heading"] / content[n])) / cfg["hd_dim"]
        r = pol - pol.min()
        pk = int(np.argmax(pol))
        err = abs(np.arctan2(np.sin(angles[pk] - psi[n]), np.cos(angles[pk] - psi[n])))
        ax_po.plot(angles, r, color="#0d7d88", lw=1.6)
        ax_po.axvline(psi[n], color="black", lw=1.6)
        ax_po.plot([angles[pk]], [r[pk]], "o", color="#b9772a", ms=7)
        ax_po.set_title(f"facing which way? (heading)\n"
                        f"peak err {np.degrees(err):.0f}°", fontsize=10)
        ax_po.set_yticklabels([])

    anim = animation.FuncAnimation(fig, render, frames=len(sel), blit=False)
    path = os.path.join(args.out_dir, "demo.gif")
    anim.save(path, writer=animation.PillowWriter(fps=args.fps))
    plt.close(fig)
    print(f"saved {path}")


# --------------------------------------------------------------------------

def cmd_query_class(args):
    """Query the semantic map by class name -> spatial heat field + PNG.

    residual = M_sem (/) SEM_class ; then sim(x,y) = Re<residual, ctx_pos(x,y)>/D
    over a grid. Peaks show where the robot was standing when it saw that class.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d, cfg, mems = _load_memory(args.out_dir, args.memory_name)
    if "semantic_position" not in mems:
        raise SystemExit("memory.pt has no semantic_position memory -- re-run `build`.")
    Msem = mems["semantic_position"]
    codebook = {c: v.numpy() for c, v in d["semantic_codebook"].items()}
    name = args.class_name
    if name not in codebook:
        raise SystemExit(f"class '{name}' not in codebook. available:\n  "
                         + ", ".join(sorted(codebook)))

    enc = ClassroomEncoders(cfg["hd_dim"], cfg["seed"] + 100,
                            cfg["pos_length_scale"], cfg["time_length_scale"])
    x = d["truth"]["x"].numpy(); y = d["truth"]["y"].numpy()
    residual = Msem / codebook[name]

    pad = 1.0
    gx = np.linspace(x.min() - pad, x.max() + pad, args.grid)
    span_ratio = (y.max() - y.min() + 2 * pad) / (x.max() - x.min() + 2 * pad)
    gy = np.linspace(y.min() - pad, y.max() + pad, max(8, int(args.grid * span_ratio)))
    field = np.empty((len(gy), len(gx)))
    for j, yy in enumerate(gy):
        for i, xx in enumerate(gx):
            ctx = enc.ctx_pos(xx, yy).values
            field[j, i] = np.real(np.sum(np.conj(residual) * ctx)) / cfg["hd_dim"]

    jpk, ipk = np.unravel_index(int(np.argmax(field)), field.shape)
    print(f"'{name}': peak similarity {field.max():.4f} at "
          f"(x={gx[ipk]:+.2f}, y={gy[jpk]:+.2f})")

    lim = np.max(np.abs(field)) or 1.0
    fig, ax = plt.subplots(figsize=(8.5, 7.5), constrained_layout=True)
    im = ax.imshow(field, origin="lower", cmap="coolwarm", vmin=-lim, vmax=lim,
                   extent=[gx[0], gx[-1], gy[0], gy[-1]], aspect="equal")
    ax.plot(x, y, "-", color="0.35", lw=1, alpha=0.6, label="robot path")
    ax.scatter([gx[ipk]], [gy[jpk]], marker="x", s=140, color="black", linewidth=2.5,
               label="peak")
    place_mode = cfg.get("place_mode", "robot")
    if place_mode == "object":
        title = f"semantic query: where ARE the '{name}'s? (object-placed)"
    else:
        title = f"semantic query: where were '{name}' observed from?"
    ax.set_title(f"{title}\nM_sem ⊘ SEM_{name}", fontsize=12)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.legend(loc="upper right", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="similarity")
    stem = os.path.splitext(os.path.basename(args.memory_name))[0]
    suffix = "" if stem == "memory" else f"_{stem}"
    path = os.path.join(args.out_dir,
                        f"semantic_query_{name.replace(' ', '_')}{suffix}.png")
    fig.savefig(path, dpi=140)
    print(f"saved {path}")


def _event_setup(args):
    """Shared loader for conjunctive queries on the semantic_event trace."""
    d, cfg, mems = _load_memory(args.out_dir, args.memory_name)
    if "semantic_event" not in mems:
        raise SystemExit("memory.pt has no semantic_event memory -- re-run `build`.")
    codebook = {c: v.numpy() for c, v in d["semantic_codebook"].items()}
    enc = ClassroomEncoders(cfg["hd_dim"], cfg["seed"] + 100,
                            cfg["pos_length_scale"], cfg["time_length_scale"])
    x = d["truth"]["x"].numpy(); y = d["truth"]["y"].numpy()
    t_row = d["truth"]["t_row"].numpy()
    return d, cfg, mems["semantic_event"], codebook, enc, x, y, t_row


def _pos_field(residual, enc, hd_dim, x, y, grid):
    """Decode a residual against a position grid -> (field, gx, gy)."""
    pad = 1.0
    gx = np.linspace(x.min() - pad, x.max() + pad, grid)
    ratio = (y.max() - y.min() + 2 * pad) / (x.max() - x.min() + 2 * pad)
    gy = np.linspace(y.min() - pad, y.max() + pad, max(8, int(grid * ratio)))
    field = np.empty((len(gy), len(gx)))
    for j, yy in enumerate(gy):
        for i, xx in enumerate(gx):
            field[j, i] = np.real(np.sum(np.conj(residual) *
                                         enc.ctx_pos(xx, yy).values)) / hd_dim
    return field, gx, gy


def cmd_query_event(args):
    """Conditional queries on M_event = sum conf * SEM (x) ctx_pos (x) ctx_time.

    Give any two of --class-name / --at-time / --at-x,--at-y; the third is
    decoded from the residual after unbinding the two you gave.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d, cfg, Mev, codebook, enc, x, y, t_row = _event_setup(args)
    hd = cfg["hd_dim"]
    have_c = args.class_name is not None
    have_t = args.at_time is not None
    have_xy = args.at_x is not None and args.at_y is not None
    if sum([have_c, have_t, have_xy]) != 2:
        raise SystemExit("give exactly two of: --class-name / --at-time / --at-x --at-y")
    if have_c and args.class_name not in codebook:
        raise SystemExit(f"unknown class '{args.class_name}'. available:\n  "
                         + ", ".join(sorted(codebook)))

    residual = Mev.copy()
    tag = []
    if have_c:
        residual = residual / codebook[args.class_name]
        tag.append(args.class_name)
    if have_t:
        residual = residual / enc.ctx_time(float(args.at_time)).values
        tag.append(f"t{int(args.at_time)}")
    if have_xy:
        residual = residual / enc.ctx_pos(args.at_x, args.at_y).values
        tag.append(f"xy{args.at_x:+.1f}{args.at_y:+.1f}")

    stem = os.path.join(args.out_dir, "event_query_" + "_".join(tag).replace(" ", "_"))
    if not have_xy:
        # decode POSITION: "where was <class> around time t?"
        field, gx, gy = _pos_field(residual, enc, hd, x, y, args.grid)
        jpk, ipk = np.unravel_index(int(np.argmax(field)), field.shape)
        n_q = int(np.clip(args.at_time, 0, len(x) - 1))
        print(f"where was '{args.class_name}' at t={args.at_time}?  "
              f"peak ({gx[ipk]:+.2f}, {gy[jpk]:+.2f})  sim={field.max():.4f}  "
              f"(robot was at ({x[n_q]:+.2f}, {y[n_q]:+.2f}))")
        lim = np.max(np.abs(field)) or 1.0
        fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
        im = ax.imshow(field, origin="lower", cmap="coolwarm", vmin=-lim, vmax=lim,
                       extent=[gx[0], gx[-1], gy[0], gy[-1]], aspect="equal")
        ax.plot(x, y, "-", color="0.35", lw=1, alpha=0.6)
        ax.scatter([x[n_q]], [y[n_q]], s=170, facecolor="none", edgecolor="black",
                   linewidth=2.2, label=f"robot @ t={int(args.at_time)}")
        ax.scatter([gx[ipk]], [gy[jpk]], marker="x", s=140, color="black",
                   linewidth=2.5, label="decoded peak")
        ax.set_title(f"M_event ⊘ SEM_{args.class_name} ⊘ ctx_time({int(args.at_time)})")
        ax.legend(fontsize=9); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        fig.colorbar(im, ax=ax, fraction=0.046, label="similarity")
        fig.savefig(stem + ".png", dpi=140); print(f"saved {stem}.png")
    elif not have_t:
        # decode TIME: "when was <class> near (x, y)?"
        tq = np.arange(len(t_row))
        curve = np.array([np.real(np.sum(np.conj(residual) *
                                         enc.ctx_time(float(t)).values)) / hd
                          for t in tq])
        pk = int(np.argmax(curve))
        print(f"when was '{args.class_name}' near ({args.at_x:+.2f}, {args.at_y:+.2f})?  "
              f"peak t={pk}  sim={curve.max():.4f}")
        fig, ax = plt.subplots(figsize=(9, 3.6), constrained_layout=True)
        ax.axhline(0, color="0.85", lw=1)
        ax.plot(tq, curve, color="#0d7d88", lw=1.6)
        ax.scatter([pk], [curve[pk]], color="#b9772a", zorder=3, label=f"peak t={pk}")
        ax.set_title(f"M_event ⊘ SEM_{args.class_name} ⊘ ctx_pos({args.at_x:+.1f},{args.at_y:+.1f})")
        ax.set_xlabel("time (frame row)"); ax.set_ylabel("similarity"); ax.legend(fontsize=9)
        fig.savefig(stem + ".png", dpi=140); print(f"saved {stem}.png")
    else:
        # decode CLASS: "what was at (x, y) around time t?"
        ranked = sorted(((c, float(np.real(np.sum(np.conj(v) * residual)) / hd))
                         for c, v in codebook.items()), key=lambda kv: -kv[1])
        print(f"what was at ({args.at_x:+.2f}, {args.at_y:+.2f}) around t={int(args.at_time)}?")
        for c, s in ranked[:8]:
            print(f"  {c:>14}  {s:+.4f}")


# --------------------------------------------------------------------------
# decaying working memory ("what is around NOW?")
# --------------------------------------------------------------------------

def _cumdist(x, y):
    """Cumulative robot travel (metres) along the interpolated (x, y) track."""
    return np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])


def _replay_now(dets_by_ts, ts, x, y, xy_by_ts, enc, codebook, classes,
                lam_time, lam_dist, decay_per, snap_rows, hd_dim):
    """Replay detections in frame order under one forgetting clock.

    ``decay_per`` selects WHAT advances each class's forgetting -- the hybrid
    form is  lambda_eff = lambda_time ** dt_frames * lambda_dist ** dd_metres:

      event -- eager: every frame row multiplies every live trace by its
               lam_time (the original build_now recursion, kept bit-exact);
      frame -- lazy: lam_time ** (frame rows elapsed since that class's
               last write), applied just before the next write / snapshot;
      metre -- lazy: lam_dist ** (metres travelled since the last write);
      both  -- lazy: the product above (time AND distance clocks).

    The lazy clocks count from the class's LAST WRITE (per-class frame row +
    cumulative travel at the last accumulator update), so a paused robot
    still forgets under the frame clock but does NOT under the metre clock.
    1/e washout horizon is -1/ln(lambda) frames (or metres).

    Returns (snaps, snaps_norm, snap_w) aligned with sorted snap_rows. The
    per-class weight-normalized bundle and the 1e-3 evidence floor are the
    event-mode originals unchanged -- only lambda_eff differs per mode."""
    N = len(ts)
    cum_d = _cumdist(x, y)
    lazy = decay_per != "event"
    snap_set = set(int(r) for r in snap_rows)
    traces = {}   # class -> complex trace, created on first sighting
    weights = {}  # class -> decayed confidence mass, same recursion
    last_n, last_d = {}, {}  # per-class frame row / travel at last write

    def pending(c, n):
        f = 1.0
        if decay_per in ("frame", "both"):
            f *= lam_time[c] ** (n - last_n[c])
        if decay_per in ("metre", "both"):
            f *= lam_dist[c] ** (cum_d[n] - last_d[c])
        return f

    snaps, snaps_norm, snap_w = [], [], []
    for n in range(N):
        if not lazy:
            for c in traces:
                traces[c] *= lam_time[c]
                weights[c] *= lam_time[c]
        tsn = int(ts[n])
        for k, (cname, conf) in enumerate(dets_by_ts.get(tsn, [])):
            if xy_by_ts is not None and xy_by_ts[tsn][k] is not None:
                px, py = xy_by_ts[tsn][k]
            else:
                px, py = float(x[n]), float(y[n])
            if lazy and cname in traces:
                f = pending(cname, n)
                traces[cname] *= f
                weights[cname] *= f
            v = traces.setdefault(cname, np.zeros(hd_dim, np.complex128))
            v += conf * codebook[cname].values * enc.ctx_pos(px, py).values
            weights[cname] = weights.get(cname, 0.0) + conf
            if lazy:
                last_n[cname] = n
                last_d[cname] = float(cum_d[n])
        if n in snap_set:
            if lazy:  # bring every trace up to date before snapshotting
                for c in traces:
                    f = pending(c, n)
                    traces[c] *= f
                    weights[c] *= f
                    last_n[c] = n
                    last_d[c] = float(cum_d[n])
            # Normalization floor: trace/weight cancels the decay (both shrink
            # by lambda^k), so without a floor a stale class would keep its
            # last-known field forever. Below ~1e-3 evidence mass the division
            # stops renormalizing and the trace decays away for real.
            zero = np.zeros(hd_dim, np.complex128)
            snaps.append(sum(traces.values(), zero))
            snaps_norm.append(sum((traces[c] / max(weights[c], 1e-3)
                                   for c in traces), zero))
            snap_w.append([weights.get(c, 0.0) for c in classes])
    return snaps, snaps_norm, snap_w


def cmd_build_now(args):
    """Decaying working memory M_now: replay detections in time order with a
    per-class exponential forgetting factor,

        M_now_c <- lambda_c * M_now_c            (every frame, all classes)
        M_now_c <- M_now_c + conf * SEM_c (x) ctx_pos(...)   (each detection)
        M_now = sum_c M_now_c

    lambda defaults: 0.995/frame for static furniture, 0.9/frame for dynamic
    classes (default ["person"]), so a person's trace follows their latest
    sighting instead of smearing over the whole walk. Snapshots of M_now at
    ~--n-snapshots evenly spaced frames are saved to --now-name (a separate
    file; the source memory file is never rewritten).

    Two snapshot variants are stored: the raw sum of class traces, and a
    weight-normalized sum (each class trace divided by its own decayed
    confidence mass -- the working-memory analogue of the /wsum the static
    semantic memory applies). The raw sum is dominated by high-count static
    classes (chair trace norm ~100x person's), whose bundling crosstalk
    swamps a rare class's field on unbind; the normalized variant equalizes
    the traces so the decode works. Per-class decayed weights are saved too,
    so query_now can tell 'decayed to zero' apart from 'crosstalk noise'.

    --decay-per picks the forgetting CLOCK (see _replay_now): 'event'
    (default, the original recursion above), 'frame' (lam per elapsed frame
    row since the class's last write), 'metre' (lambda-*-dist per metre of
    robot travel since the last write; 1/e washout = -1/ln(lambda) m), or
    'both' (hybrid product lambda_time**dt_frames * lambda_dist**dd_metres)."""
    d, cfg, mems = _load_memory(args.out_dir, args.memory_name)
    frame_idx, ts, emb = _load_embeddings(args.out_dir)
    N = len(ts)
    x = d["truth"]["x"].numpy(); y = d["truth"]["y"].numpy()
    psi = d["truth"]["yaw"].numpy()
    enc = ClassroomEncoders(cfg["hd_dim"], cfg["seed"] + 100,
                            cfg["pos_length_scale"], cfg["time_length_scale"])
    dets_by_ts = _load_detections_by_ts(args.out_dir)
    classes = sorted({c for lst in dets_by_ts.values() for c, _ in lst})
    codebook = {c: Phasor(dim=cfg["hd_dim"], seed=_class_seed(c)) for c in classes}
    dyn = set(args.dynamic_classes)
    lam = {c: (args.lambda_dynamic if c in dyn else args.lambda_static)
           for c in classes}
    lam_dist = {c: (args.lambda_dynamic_dist if c in dyn else args.lambda_static_dist)
                for c in classes}

    xy_by_ts = None
    if args.place_mode == "object":
        pose_of_ts = {int(ts[n]): (float(x[n]), float(y[n]), float(psi[n]))
                      for n in range(N)}
        xy_by_ts, n_ok, n_fb = _localize_detections(args.out_dir, pose_of_ts)
        print(f"object placement: {n_ok} depth-localized, {n_fb} fallback "
              f"({100 * n_fb / max(1, n_ok + n_fb):.1f}%)")

    snap_rows = np.unique(np.linspace(0, N - 1, args.n_snapshots).round().astype(int))
    print(f"replaying {N} frames, decay_per={args.decay_per}, "
          f"lambda_static={args.lambda_static}, "
          f"lambda_dynamic={args.lambda_dynamic} (dynamic: {sorted(dyn)})")
    if args.decay_per in ("metre", "both"):
        print(f"  distance rates: lambda_static_dist={args.lambda_static_dist}/m "
              f"(1/e at {-1 / np.log(args.lambda_static_dist):.0f} m), "
              f"lambda_dynamic_dist={args.lambda_dynamic_dist}/m "
              f"(1/e at {-1 / np.log(args.lambda_dynamic_dist):.1f} m)")
    snaps, snaps_norm, snap_w = _replay_now(
        dets_by_ts, ts, x, y, xy_by_ts, enc, codebook, classes,
        lam, lam_dist, args.decay_per, snap_rows, cfg["hd_dim"])

    path = os.path.join(args.out_dir, args.now_name)
    torch.save({
        "snapshots": torch.from_numpy(np.stack(snaps)),
        "snapshots_normed": torch.from_numpy(np.stack(snaps_norm)),
        "snapshot_weights": torch.from_numpy(np.array(snap_w, dtype=np.float64)),
        "snapshot_rows": torch.from_numpy(snap_rows.astype(np.int64)),
        "snapshot_ts": torch.from_numpy(ts[snap_rows].astype(np.int64)),
        "classes": classes,
        "config": {"hd_dim": cfg["hd_dim"], "seed": cfg["seed"],
                   "pos_length_scale": cfg["pos_length_scale"],
                   "time_length_scale": cfg["time_length_scale"],
                   "lambda_static": args.lambda_static,
                   "lambda_dynamic": args.lambda_dynamic,
                   "decay_per": args.decay_per,
                   "lambda_static_dist": args.lambda_static_dist,
                   "lambda_dynamic_dist": args.lambda_dynamic_dist,
                   "dynamic_classes": sorted(dyn),
                   "place_mode": args.place_mode,
                   "source_memory": args.memory_name},
    }, path)
    print(f"saved {len(snaps)} M_now snapshots at frame rows "
          f"{snap_rows.tolist()} -> {path}")


def cmd_query_now(args):
    """Compare the decayed working memory against the static all-time map for
    one class: top row = M_now snapshots at 4 times (the field should FOLLOW
    the latest sighting), bottom row = the same query on the static
    semantic_position memory (identical at every time -- smeared over all
    sightings). Circle = robot pose at the class's most recent detection
    before each snapshot; X = decoded field peak."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d, cfg, mems = _load_memory(args.out_dir, args.memory_name)
    now = torch.load(os.path.join(args.out_dir, args.now_name), weights_only=True)
    key = "snapshots" if args.bundle == "raw" else "snapshots_normed"
    snaps = now[key].numpy()
    snap_w = now["snapshot_weights"].numpy()
    now_classes = list(now["classes"])
    snap_rows = now["snapshot_rows"].numpy()
    ncfg = now["config"]
    if ncfg["hd_dim"] != cfg["hd_dim"] or ncfg["seed"] != cfg["seed"]:
        raise SystemExit(f"{args.now_name} was built against a different config "
                         f"than {args.memory_name} -- re-run build_now.")
    name = args.class_name
    codebook = {c: v.numpy() for c, v in d["semantic_codebook"].items()}
    if name not in codebook:
        raise SystemExit(f"class '{name}' not in codebook. available:\n  "
                         + ", ".join(sorted(codebook)))
    sem = codebook[name]
    enc = ClassroomEncoders(cfg["hd_dim"], cfg["seed"] + 100,
                            cfg["pos_length_scale"], cfg["time_length_scale"])
    x = d["truth"]["x"].numpy(); y = d["truth"]["y"].numpy()
    frame_idx, ts, emb = _load_embeddings(args.out_dir)
    dets_by_ts = _load_detections_by_ts(args.out_dir)
    det_rows = np.array([n for n in range(len(ts))
                         if any(c == name for c, _ in dets_by_ts.get(int(ts[n]), []))])
    if len(det_rows) == 0:
        raise SystemExit(f"'{name}' never detected -- nothing to track.")

    # 4 snapshots: --snap-cols indices, else evenly spaced over the saved set
    if args.snap_cols:
        cols = np.array(sorted(set(int(c) % len(snaps) for c in args.snap_cols)))
    else:
        cols = np.unique(np.linspace(0, len(snaps) - 1, 4).round().astype(int))
    ci_class = now_classes.index(name)
    w_peak = snap_w[:, ci_class].max() or 1.0
    static_field, gx, gy = _pos_field(mems["semantic_position"] / sem, enc,
                                      cfg["hd_dim"], x, y, args.grid)
    sj, si = np.unravel_index(int(np.argmax(static_field)), static_field.shape)

    fig, axs = plt.subplots(2, len(cols), figsize=(4.1 * len(cols), 7.6),
                            constrained_layout=True)
    print(f"tracking '{name}': {len(det_rows)} detection frames, "
          f"lambda_dynamic={ncfg['lambda_dynamic']} "
          f"(dynamic: {ncfg['dynamic_classes']}, bundle={args.bundle}, "
          f"decay_per={ncfg.get('decay_per', 'event')})")
    for j, ci in enumerate(cols):
        row = int(snap_rows[ci])
        w = float(snap_w[ci, ci_class])
        forgotten = w < 0.01 * w_peak  # trace decayed to ~nothing
        field, _, _ = _pos_field(snaps[ci] / sem, enc, cfg["hd_dim"], x, y, args.grid)
        jpk, ipk = np.unravel_index(int(np.argmax(field)), field.shape)
        seen = det_rows[det_rows <= row]
        last = int(seen[-1]) if len(seen) else None
        for r, (ax, f, pj, pi) in enumerate([(axs[0, j], field, jpk, ipk),
                                             (axs[1, j], static_field, sj, si)]):
            lim = np.max(np.abs(f)) or 1.0
            ax.imshow(f, origin="lower", cmap="coolwarm", vmin=-lim, vmax=lim,
                      extent=[gx[0], gx[-1], gy[0], gy[-1]], aspect="equal")
            ax.plot(x, y, "-", color="0.35", lw=0.8, alpha=0.5)
            if last is not None:
                ax.scatter([x[last]], [y[last]], s=150, facecolor="none",
                           edgecolor="black", linewidth=2.0,
                           label=f"last '{name}' @ t={last}")
            if not (r == 0 and forgotten):
                ax.scatter([gx[pi]], [gy[pj]], marker="x", s=120, color="black",
                           linewidth=2.2, label="peak")
            if r == 0:
                title = f"M_now @ t={row}  (w={w:.2f})"
                if forgotten:
                    title += "\ntrace decayed -- forgotten"
            else:
                title = f"static M_sem @ t={row}"
            ax.set_title(title, fontsize=10)
            if j == 0:
                ax.set_ylabel("y (m)")
            if r == 1:
                ax.set_xlabel("x (m)")
            if j == 0 and r == 0 and ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=7, loc="upper right")
        note = " [decayed weight ~0 -> field is crosstalk noise]" if forgotten else ""
        if last is not None:
            dpk = float(np.hypot(gx[ipk] - x[last], gy[jpk] - y[last]))
            gap = row - last
            print(f"  t={row:4d}: M_now peak ({gx[ipk]:+.2f}, {gy[jpk]:+.2f}) "
                  f"sim={field.max():+.4f} w={w:.2f} | last sighting t={last} "
                  f"({gap} frames earlier) at ({x[last]:+.2f}, {y[last]:+.2f}) "
                  f"-> peak-to-sighting {dpk:.2f} m{note}")
        else:
            print(f"  t={row:4d}: M_now peak ({gx[ipk]:+.2f}, {gy[jpk]:+.2f}) "
                  f"sim={field.max():+.4f} w={w:.2f} | no '{name}' sighting yet{note}")
    print(f"static M_sem peak ({gx[si]:+.2f}, {gy[sj]:+.2f}) "
          f"sim={static_field.max():+.4f} (same at every time)")
    fig.suptitle(f"working memory vs static map: '{name}'  "
                 f"(lambda={ncfg['lambda_dynamic']}/frame for dynamic classes)",
                 fontsize=12)
    path = os.path.join(args.out_dir,
                        f"working_memory_{name.replace(' ', '_')}.png")
    fig.savefig(path, dpi=130)
    print(f"saved {path}")


def cmd_compare_clocks(args):
    """Hybrid forgetting-clock comparison: build M_now under all four
    --decay-per modes on the SAME detections (robot place-mode) and probe one
    dynamic class's decayed weight + field-peak error at four times:

        fresh at the start of the most-MOVING stale gap, stale +W frames
        into it; fresh at the start of the SLOWEST stale gap, stale +W
        frames into it (W = --stale-frames).

    Claim under test: the metre clock keeps memory while the robot pauses
    (w stays high through a near-stationary stale window) while the frame
    clock forgets during the pause; 'both' interpolates. The robot speed
    profile over W-frame windows is reported so the reader can judge whether
    this walk contains a stationary-enough window at all. Renders ONE figure
    decay_clocks_comparison.png (speed profile + per-mode w and peak error)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d, cfg, mems = _load_memory(args.out_dir, args.memory_name)
    frame_idx, ts, emb = _load_embeddings(args.out_dir)
    N = len(ts)
    x = d["truth"]["x"].numpy(); y = d["truth"]["y"].numpy()
    enc = ClassroomEncoders(cfg["hd_dim"], cfg["seed"] + 100,
                            cfg["pos_length_scale"], cfg["time_length_scale"])
    dets_by_ts = _load_detections_by_ts(args.out_dir)
    classes = sorted({c for lst in dets_by_ts.values() for c, _ in lst})
    codebook = {c: Phasor(dim=cfg["hd_dim"], seed=_class_seed(c)) for c in classes}
    name = args.class_name
    if name not in codebook:
        raise SystemExit(f"class '{name}' not in codebook. available:\n  "
                         + ", ".join(sorted(classes)))
    dyn = set(args.dynamic_classes)
    lam_time = {c: (args.lambda_dynamic if c in dyn else args.lambda_static)
                for c in classes}
    lam_dist = {c: (args.lambda_dynamic_dist if c in dyn else args.lambda_static_dist)
                for c in classes}

    # ---- robot speed profile over W-frame windows ----------------------------
    W = args.stale_frames
    cum_d = _cumdist(x, y)
    starts = np.arange(0, N - W)
    travel = cum_d[starts + W] - cum_d[starts]
    dt_s = (ts[starts + W] - ts[starts]).astype(np.float64) / 1e9
    speed = travel / np.maximum(dt_s, 1e-9)
    print(f"speed profile over {W}-frame windows: travel "
          f"min={travel.min():.3f} m, median={np.median(travel):.3f} m, "
          f"max={travel.max():.3f} m  (speed min={speed.min():.4f}, "
          f"median={np.median(speed):.4f} m/s)")

    # ---- probe windows -------------------------------------------------------
    # Eligible stale probes p: last '<name>' write W..1.5W frames before p
    # (includes the tail after the LAST sighting, not just inter-detection
    # gaps). Rank by robot travel over the trailing W frames -> most-moving
    # and most-stationary stale windows, each paired with a fresh probe at
    # its own last-sighting row.
    det_rows = np.array([n for n in range(N)
                         if any(c == name for c, _ in dets_by_ts.get(int(ts[n]), []))])
    if len(det_rows) == 0:
        raise SystemExit(f"'{name}' never detected -- nothing to compare.")
    det_set = set(det_rows.tolist())
    last = np.full(N, -1)
    cur = -1
    for n in range(N):
        if n in det_set:
            cur = n
        last[n] = cur
    stale = np.arange(N) - last
    elig = [p for p in range(W, N)
            if last[p] >= 0 and W <= stale[p] <= int(1.5 * W)]
    if not elig:
        raise SystemExit(f"no probe row with '{name}' stale by {W}-{int(1.5 * W)} frames.")
    trail = {p: float(cum_d[p] - cum_d[p - W]) for p in elig}
    p_mov = max(elig, key=trail.get)
    p_stat = min(elig, key=trail.get)
    t_mov, t_stat = trail[p_mov], trail[p_stat]
    stationary_ok = t_stat <= args.stationary_max_travel
    print(f"stale probes ('{name}' unseen for {W}-{int(1.5 * W)} frames): "
          f"{len(elig)} candidates; most-moving p={p_mov} "
          f"(stale {stale[p_mov]}, {t_mov:.2f} m over trailing {W} frames), "
          f"most-stationary p={p_stat} (stale {stale[p_stat]}, {t_stat:.2f} m)")
    print(f"  stationary-enough stale window EXISTS (travel {t_stat:.2f} <= "
          f"{args.stationary_max_travel} m) -- frame-vs-metre contrast "
          "demonstrable during the pause." if stationary_ok else
          f"  NO nearly-stationary stale window on this walk ({t_stat:.2f} m > "
          f"{args.stationary_max_travel} m) -- contrast shown on moving "
          "windows only.")
    probes = [
        (f"fresh @t={last[p_mov]}\n(moving window)", int(last[p_mov])),
        (f"stale {stale[p_mov]}f @t={p_mov}\n({t_mov:.2f} m moved)", p_mov),
        (f"fresh @t={last[p_stat]}\n(stationary window)", int(last[p_stat])),
        (f"stale {stale[p_stat]}f @t={p_stat}\n({t_stat:.2f} m moved)", p_stat),
    ]

    # ---- replay under each clock, decode the class field at each probe -------
    modes = ["event", "frame", "metre", "both"]
    snap_rows = np.array(sorted({r for _, r in probes}))
    ci = classes.index(name)
    sem = codebook[name].values
    table = {}  # (probe_idx, mode) -> (w, err_m, sim)
    for mode in modes:
        _s, snaps_norm, snap_w = _replay_now(
            dets_by_ts, ts, x, y, None, enc, codebook, classes,
            lam_time, lam_dist, mode, snap_rows, cfg["hd_dim"])
        by_row = {int(r): (snaps_norm[k], snap_w[k][ci])
                  for k, r in enumerate(snap_rows)}
        for pi, (_label, r) in enumerate(probes):
            snap, w = by_row[r]
            field, gx, gy = _pos_field(snap / sem, enc, cfg["hd_dim"],
                                       x, y, args.grid)
            jpk, ipk = np.unravel_index(int(np.argmax(field)), field.shape)
            last = int(det_rows[det_rows <= r][-1])
            err = float(np.hypot(gx[ipk] - x[last], gy[jpk] - y[last]))
            table[pi, mode] = (float(w), err, float(field.max()))
    print(f"per-mode '{name}' decode at each probe (err = field peak to robot "
          "pose at last sighting; err is crosstalk noise once w ~ 0):")
    for pi, (label, r) in enumerate(probes):
        flat = label.replace('\n', ' ')
        print(f"  {flat}")
        for mode in modes:
            w, err, sim = table[pi, mode]
            note = "  [w ~ 0 -> forgotten]" if w < 0.01 else ""
            print(f"    {mode:>5}: w={w:8.3f}  peak_err={err:5.2f} m  "
                  f"sim={sim:+.4f}{note}")

    # ---- one figure: speed profile + per-mode w and peak error ---------------
    colors = {"event": "#0d7d88", "frame": "#b9772a",
              "metre": "#7a4b94", "both": "#3d7a3d"}
    fig = plt.figure(figsize=(13.0, 8.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.35])
    ax_sp = fig.add_subplot(gs[0, :])
    ax_w = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[1, 1])

    ax_sp.plot(starts, travel, color="0.4", lw=1.2)
    ax_sp.axhline(args.stationary_max_travel, color="0.7", lw=1, ls="--")
    ax_sp.plot(det_rows, np.full(len(det_rows), -0.1 * travel.max()), "|",
               color="#0d7d88", ms=6, alpha=0.5, label=f"'{name}' detections")
    for label, r in probes:
        ax_sp.axvline(r, color="#b9772a", lw=1.2, alpha=0.8)
        ax_sp.text(r, travel.max() * 0.97, f" t={r}", rotation=90,
                   fontsize=7, va="top")
    ax_sp.set_title(f"robot travel per {W}-frame window (dashes: 'stationary' "
                    f"threshold {args.stationary_max_travel} m; verticals: probes)",
                    fontsize=10)
    ax_sp.set_xlabel("frame row"); ax_sp.set_ylabel(f"m / {W} frames")
    ax_sp.legend(fontsize=8, loc="upper right")

    xs = np.arange(len(probes))
    wd = 0.2
    for m, mode in enumerate(modes):
        off = (m - 1.5) * wd
        wv = [max(table[pi, mode][0], 1e-4) for pi in range(len(probes))]
        ev = [table[pi, mode][1] for pi in range(len(probes))]
        ax_w.bar(xs + off, wv, wd, color=colors[mode], label=mode)
        bars = ax_e.bar(xs + off, ev, wd, color=colors[mode], label=mode)
        for pi, b in enumerate(bars):
            if table[pi, mode][0] < 0.01:  # forgotten -> peak is crosstalk
                b.set_alpha(0.35); b.set_hatch("//")
    ax_w.set_yscale("log")
    ax_w.set_title(f"decayed '{name}' weight w (log; floor 1e-4 shown)",
                   fontsize=10)
    ax_w.set_ylabel("w")
    ax_e.set_title("field-peak error to last sighting\n(hatched: w ~ 0, "
                   "peak is crosstalk)", fontsize=10)
    ax_e.set_ylabel("error (m)")
    for ax in (ax_w, ax_e):
        ax.set_xticks(xs)
        ax.set_xticklabels([p[0] for p in probes], fontsize=8)
        ax.legend(fontsize=8)
    fig.suptitle(
        f"forgetting clocks: lambda_eff = lambda_time**dt_frames * "
        f"lambda_dist**dd_metres  ('{name}': time {args.lambda_dynamic}/frame, "
        f"dist {args.lambda_dynamic_dist}/m)", fontsize=11)
    path = os.path.join(args.out_dir, "decay_clocks_comparison.png")
    fig.savefig(path, dpi=130)
    print(f"saved {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("embed", help="YOLOv8n detect + embed every RGB frame")
    pe.add_argument("--stride", type=int, default=1, help="embed every Nth frame")

    pec = sub.add_parser("embed-crops",
                         help="per-DETECTION appearance features: crop each YOLO box "
                              "and embed the crop (partial-cue keys)")
    pec.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    pec.add_argument("--pad-frac", type=float, default=0.08,
                     help="expand each box by this fraction of its long side before "
                          "cropping, to keep a little context (0 = tight box)")
    pec.add_argument("--min-side", type=float, default=16.0,
                     help="boxes with a shorter side than this are still embedded but "
                          "flagged 'tiny' -- their crops are mostly upsampling artefact")
    pec.add_argument("--crops-name", default="crop_embeddings.pt")
    pec.add_argument("--dataset", default=None,
                     help="path to a sequence config JSON (see "
                          "vsa_cognitive_mapping/configs/TEMPLATE_new_dataset.json). "
                          "Preferred: works for local folders as well as HF, and "
                          "sets --out-dir from the config. Validate it first with "
                          "`python -m vsa_cognitive_mapping.sequences validate`.")
    pec.add_argument("--repo", default=DATASET)
    pec.add_argument("--rgb-config", default="rgb_d455",
                     help="HF config name, used only when --dataset is absent. "
                          "NOTE: --out-dir alone does NOT select a sequence -- pass "
                          "--rgb-config too or this silently re-crops the classroom")
    pec.add_argument("--keep-frame-embed", action="store_true",
                     help="also embed the whole frame, for a like-for-like isotropy "
                          "comparison on exactly these frames")

    pb = sub.add_parser("build", help="build the three per-axis memories")
    pb.add_argument("--hd-dim", type=int, default=8192)
    pb.add_argument("--seed", type=int, default=0)
    pb.add_argument("--subset", choices=["all", "stride"], default="stride")
    pb.add_argument("--subset-stride", type=int, default=3,
                    help="with --subset stride: bundle every Nth frame, hold out the rest")
    pb.add_argument("--pos-length-scale", type=float, default=0.75)
    pb.add_argument("--time-length-scale", type=float, default=20.0)
    pb.add_argument("--pca-whiten", dest="pca_whiten", action="store_true", default=True,
                    help="standardized-PCA preprocess before projection (default ON: raw "
                         "YOLO content sits in a 0.76-1.00 similarity band, which swamps "
                         "content->context unbinding with crosstalk)")
    pb.add_argument("--no-pca-whiten", dest="pca_whiten", action="store_false")
    pb.add_argument("--place-mode", choices=["robot", "object"], default="robot",
                    help="semantic memories bind each class to the robot's pose "
                         "(default) or to the object's own depth-deprojected "
                         "world (x, y); invalid depth falls back to robot pose")

    pv = sub.add_parser("evaluate", help="self-recall error per axis")

    pd = sub.add_parser("demo", help="held-out demo gif (image | heatmap | polar)")
    pd.add_argument("--demo-frames", type=int, default=48)
    pd.add_argument("--grid", type=int, default=44)
    pd.add_argument("--fps", type=int, default=4)

    pq = sub.add_parser("query_class", help="semantic query: class name -> spatial heatmap")
    pq.add_argument("--class-name", required=True, help='e.g. "chair", "person", "tv"')
    pq.add_argument("--grid", type=int, default=120)

    pev = sub.add_parser("query_event",
                         help="conjunctive query on class (x) place (x) time: give two, decode the third")
    pev.add_argument("--class-name", default=None)
    pev.add_argument("--at-time", type=float, default=None, help="frame row index")
    pev.add_argument("--at-x", type=float, default=None)
    pev.add_argument("--at-y", type=float, default=None)
    pev.add_argument("--grid", type=int, default=90)

    pbn = sub.add_parser("build_now",
                         help="decaying working memory: replay detections, save M_now snapshots")
    pbn.add_argument("--decay-per", choices=["event", "frame", "metre", "both"],
                     default="event",
                     help="forgetting clock: 'event' (original per-event "
                          "recursion, default), 'frame' (lambda per elapsed "
                          "frame row since the class's last write), 'metre' "
                          "(lambda-*-dist per metre of robot travel since the "
                          "last write), or 'both' (hybrid product "
                          "lambda_time**dt * lambda_dist**dd)")
    pbn.add_argument("--n-snapshots", type=int, default=8)
    pbn.add_argument("--place-mode", choices=["robot", "object"], default="robot")

    pcc = sub.add_parser("compare_clocks",
                         help="build M_now under all four --decay-per clocks, "
                              "probe one class -> decay_clocks_comparison.png")
    pcc.add_argument("--class-name", default="person", help="dynamic class to probe")
    pcc.add_argument("--grid", type=int, default=80)
    pcc.add_argument("--stale-frames", type=int, default=50,
                     help="staleness horizon W: probe W frames after a sighting")
    pcc.add_argument("--stationary-max-travel", type=float, default=0.5,
                     help="a stale window counts as 'nearly stationary' if the "
                          "robot travels <= this many metres across it")

    for p in (pbn, pcc):
        p.add_argument("--lambda-static", type=float, default=0.995,
                       help="per-frame forgetting factor for static classes")
        p.add_argument("--lambda-dynamic", type=float, default=0.9,
                       help="per-frame forgetting factor for dynamic classes")
        p.add_argument("--lambda-static-dist", type=float, default=0.999,
                       help="per-METRE forgetting factor for static classes "
                            "(decay-per metre/both; 1/e washout distance "
                            "= -1/ln(lambda), so 0.999 -> ~1000 m)")
        p.add_argument("--lambda-dynamic-dist", type=float, default=0.7,
                       help="per-METRE forgetting factor for dynamic classes "
                            "(0.7 -> 1/e washout at ~2.8 m of robot travel)")
        p.add_argument("--dynamic-classes", nargs="*", default=["person"])

    pqn = sub.add_parser("query_now",
                         help="M_now snapshots vs static map for one class -> comparison figure")
    pqn.add_argument("--class-name", required=True, help='e.g. "person"')
    pqn.add_argument("--grid", type=int, default=80)
    pqn.add_argument("--bundle", choices=["normed", "raw"], default="normed",
                     help="which stored M_now variant to decode: 'normed' "
                          "(per-class decayed-weight normalization; default) or "
                          "'raw' (plain sum -- static-class crosstalk swamps "
                          "rare classes, kept for comparison)")
    pqn.add_argument("--snap-cols", type=int, nargs="*", default=None,
                     help="indices of the saved snapshots to show (default: "
                          "4 evenly spaced)")

    for p in (pbn, pqn):
        p.add_argument("--now-name", default="memory_now.pt",
                       help="working-memory snapshot file inside --out-dir")

    for p in (pe, pec, pb, pv, pd, pq, pev, pbn, pqn, pcc):
        p.add_argument("--out-dir", default=DEFAULT_OUT)
        p.add_argument("--memory-name", default="memory.pt",
                       help="memory file name inside --out-dir (build writes "
                            "it, the query/eval stages read it)")

    args = ap.parse_args()
    {"embed": cmd_embed, "embed-crops": cmd_embed_crops,
     "build": cmd_build, "evaluate": cmd_evaluate,
     "demo": cmd_demo, "query_class": cmd_query_class,
     "query_event": cmd_query_event, "build_now": cmd_build_now,
     "query_now": cmd_query_now, "compare_clocks": cmd_compare_clocks}[args.cmd](args)


if __name__ == "__main__":
    main()
