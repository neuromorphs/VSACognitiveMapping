"""Build a self-contained HTML page with three frame-by-frame players.

ffmpeg is not installed here, and an mp4 would not be inspectable anyway, so
"video" is done the way the demo page does it: small JPEGs inlined as data URIs
and stepped by JavaScript. One file, no network, scrubbable.

Players:

  A  flythrough      camera + detection boxes, world-frame position and heading
  B  memory recall   same frames, plus the position field the VSA memory decodes
                     from the classes visible in that frame, with the error
  C  encoder compare a query crop and the nearest neighbour each encoder returns

A and B share one image payload -- they differ only in the right-hand panel.

The memory in B is built here rather than loaded, because `astm_traces` and
`classroom_pipeline.load_poses_interpolated` both hardcode the classroom's HF
configs. It uses this package's own primitives (`ClassroomEncoders.ctx_pos`,
`_class_seed`, `Phasor`), and is the same construction the demo page replays:

    M = sum_i  conf_i * ( CLASS_i  (x)  ctx_pos(robot_x_i, robot_y_i) )

and per frame, for each visible class c, unbind and correlate the residual
against a grid of positions, then sum the per-class fields by confidence.

Usage:
    python tools/export_video_page.py --scene school_run1
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vsa_cognitive_mapping.classroom_pipeline import (  # noqa: E402
    ClassroomEncoders, _class_seed, quat_to_yaw)

REPO = "lorinachey/spot-telluride-workshop-dataset"
SCENES = {
    "classroom":   {"rgb": "rgb_d455",             "odom": "odometry_lio_sam"},
    "school_run1": {"rgb": "school_run1_rgb_d455", "odom": "school_run1_odometry_lio_sam"},
}


# ---------------------------------------------------------------- helpers

def jpeg_uri(img, width, quality):
    w, h = img.size
    im = img.convert("RGB").resize((width, max(1, int(round(h * width / w)))))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def poses_for(repo, odom_config, frame_ts):
    """Odometry interpolated onto frame timestamps. Same maths as
    classroom_pipeline.load_poses_interpolated, but with the config passed in."""
    from datasets import load_dataset
    odo = load_dataset(repo, odom_config, split="train")
    ot = np.array(odo["timestamp_ns"], dtype=np.float64)
    o = np.argsort(ot); ot = ot[o]
    ox = np.array(odo["x"], dtype=np.float64)[o]
    oy = np.array(odo["y"], dtype=np.float64)[o]
    yaw = quat_to_yaw(np.array(odo["qx"], dtype=np.float64)[o],
                      np.array(odo["qy"], dtype=np.float64)[o],
                      np.array(odo["qz"], dtype=np.float64)[o],
                      np.array(odo["qw"], dtype=np.float64)[o])
    t = np.asarray(frame_ts, dtype=np.float64)
    x = np.interp(t, ot, ox); y = np.interp(t, ot, oy)
    # interpolate yaw through the unit circle so the +/-pi wrap does not spin
    cy = np.interp(t, ot, np.cos(yaw)); sy = np.interp(t, ot, np.sin(yaw))
    return x, y, np.arctan2(sy, cy)


def read_detections(path):
    """frame_idx -> [(class_name, conf, x1, y1, x2, y2), ...]"""
    import csv as _csv
    by = defaultdict(list)
    with open(path, newline="") as f:
        for r in _csv.DictReader(f):
            by[int(r["frame_idx"])].append((r["class_name"], float(r["confidence"]),
                                            float(r["x1"]), float(r["y1"]),
                                            float(r["x2"]), float(r["y2"])))
    return by


def quant(field):
    """float field -> uint8 base64, rescaled to its own min/max."""
    lo, hi = float(field.min()), float(field.max())
    if hi - lo < 1e-12:
        u = np.zeros(field.shape, dtype=np.uint8)
    else:
        u = np.clip((field - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    return base64.b64encode(u.tobytes()).decode(), lo, hi


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="school_run1", choices=list(SCENES))
    ap.add_argument("--out-dir", default=None, help="default outputs/<scene>")
    ap.add_argument("--frames", type=int, default=170, help="frames in players A/B")
    ap.add_argument("--img-width", type=int, default=328)
    ap.add_argument("--quality", type=int, default=58)
    ap.add_argument("--hd-dim", type=int, default=2048,
                    help="memory dimensionality for player B (2048 keeps the "
                         "grid codebook small; the paper's runs use 8192)")
    ap.add_argument("--grid", type=int, default=52)
    ap.add_argument("--mem-stride", type=int, default=3,
                    help="bundle every Nth frame descriptor into the memory; the "
                         "upstream pipeline uses 3 for the same reason (adjacent "
                         "frames are near-duplicates that waste bundling capacity)")
    ap.add_argument("--queries", type=int, default=42, help="rows in player C")
    ap.add_argument("--thumb", type=int, default=104)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pos-length-scale", type=float, default=0.75)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    scene = args.scene
    out_dir = args.out_dir or os.path.join("outputs", scene)
    cfg = SCENES[scene]
    from datasets import load_dataset

    print(f"[{scene}] loading {cfg['rgb']} ...", flush=True)
    ds = load_dataset(REPO, cfg["rgb"], split="train")
    ts_all = np.array(ds["timestamp_ns"], dtype=np.int64)
    idx_all = np.array(ds["frame_idx"], dtype=np.int64)
    order = np.argsort(ts_all)
    ts_s, idx_s = ts_all[order], idx_all[order]
    n_all = len(order)
    px, py, pyaw = poses_for(REPO, cfg["odom"], ts_s)
    print(f"  {n_all} frames, odometry span x[{px.min():.1f},{px.max():.1f}] "
          f"y[{py.min():.1f},{py.max():.1f}]", flush=True)

    # --- pose sanity check -------------------------------------------------
    # A diverged SLAM solution silently turns every recall number into a
    # measurement of the odometry rather than of the memory. school_run1's
    # LIO-SAM run does exactly this (209 jumps > 5 m, 23 km of "path" in 7.5
    # minutes), so refuse to present recall as a result without saying so.
    step = np.hypot(np.diff(px), np.diff(py))
    dt = np.maximum(np.diff(ts_s.astype(np.float64)) / 1e9, 1e-9)
    speed = step / dt
    jumps = int((step > 5.0).sum())
    vmax = float(np.nanmax(speed)) if len(speed) else 0.0
    warn = None
    if jumps > 0 or vmax > 5.0:
        warn = {"jumps_over_5m": jumps,
                "path_len_m": round(float(step.sum()), 1),
                "duration_s": round(float((ts_s[-1] - ts_s[0]) / 1e9), 1),
                "max_speed_ms": round(vmax, 1),
                "extent_m": [round(float(px.max() - px.min()), 1),
                             round(float(py.max() - py.min()), 1)]}
        print(f"  !! ODOMETRY DIVERGED: {jumps} jumps > 5 m, "
              f"{step.sum():.0f} m of path in {(ts_s[-1]-ts_s[0])/1e9:.0f} s, "
              f"max implied speed {vmax:.0f} m/s. Recall numbers below measure "
              f"the odometry, not the memory.", flush=True)

    dets = read_detections(os.path.join(out_dir, "detections_crops.csv"))

    # ---------- position memory, one per encoder (player B) ---------------
    # Content is a per-frame appearance descriptor (the mean of that frame's
    # crop embeddings), bound to the robot's pose. Querying with a frame's own
    # descriptor makes the residual peak at where that frame was taken -- the
    # "have I seen this before?" readout.
    #
    # Binding CLASS atoms instead does not localise: unbinding class c returns a
    # bundle over every place c was ever seen, so summing per-class fields gives
    # the union of those places rather than their intersection. Measured at
    # 3.19 m median before this was changed.
    from vsa_cognitive_mapping.vsa import pca_components, random_project_to_phasor

    enc = ClassroomEncoders(args.hd_dim, args.seed, args.pos_length_scale, 20.0)
    pos_of_frame = {int(f): (px[i], py[i], pyaw[i]) for i, f in enumerate(idx_s)}
    n_ev = sum(len(v) for v in dets.values())

    ENC_FILES = [("yolov8n", "crop_embeddings.pt", "CNN · detection"),
                 ("resnet50", "crop_embeddings_resnet50.pt", "CNN · ImageNet"),
                 ("dinov2", "crop_embeddings_dinov2.pt", "ViT · self-sup"),
                 ("untrained", "crop_embeddings_resnet50-untrained.pt", "CNN · random init")]
    avail = [(nm, fn, tg) for nm, fn, tg in ENC_FILES
             if os.path.exists(os.path.join(out_dir, fn))]
    print(f"[encoders] available: {[a[0] for a in avail]}", flush=True)

    # decode grid over the walked extent, with a margin
    ex = [px.min() - 1.2, px.max() + 1.2, py.min() - 1.2, py.max() + 1.2]
    gx = np.linspace(ex[0], ex[1], args.grid)
    gy = np.linspace(ex[2], ex[3], args.grid)
    print(f"[memory] {args.grid}x{args.grid} decode grid ...", flush=True)
    G = np.empty((args.grid * args.grid, args.hd_dim), dtype=np.complex64)
    k = 0
    for yy in gy:
        for xx in gx:
            G[k] = enc.ctx_pos(float(xx), float(yy)).values.astype(np.complex64)
            k += 1

    base_meta = torch.load(os.path.join(out_dir, "crop_embeddings.pt"),
                           weights_only=False)
    crop_frames = base_meta["frame_idx"].numpy()

    MEM = {}          # encoder -> dict(M, z_of_frame, dim)
    for nm, fn, tg in avail:
        X = torch.load(os.path.join(out_dir, fn), weights_only=False)["embedding"].numpy()
        # frame descriptor = mean of that frame's crops, L2-normalised
        uf, inv = np.unique(crop_frames, return_inverse=True)
        D = np.zeros((len(uf), X.shape[1]), dtype=np.float64)
        np.add.at(D, inv, X)
        cnt = np.bincount(inv, minlength=len(uf)).astype(np.float64)[:, None]
        D /= np.maximum(cnt, 1)
        D /= (np.linalg.norm(D, axis=1, keepdims=True) + 1e-12)

        # whiten over the frame descriptors, as the pipeline does
        Wht, _ = pca_components(D, n_components=min(128, D.shape[1]))
        z, _Wp = random_project_to_phasor(
            torch.from_numpy(np.ascontiguousarray(Wht)).float(),
            d=args.hd_dim, seed=args.seed)
        Z = z.numpy().astype(np.complex128)

        # bundle a strided subset so the memory is not saturated
        keep = np.arange(0, len(uf), args.mem_stride)
        M = np.zeros(args.hd_dim, dtype=np.complex128)
        for j in keep:
            p = pos_of_frame.get(int(uf[j]))
            if p is None:
                continue
            M += Z[j] * enc.ctx_pos(float(p[0]), float(p[1])).values
        M /= max(np.abs(M).max(), 1e-12)
        MEM[nm] = {"M": M, "Z": Z, "frames": uf, "tag": tg,
                   "dim": int(X.shape[1]), "k": int(len(keep))}
        print(f"  {nm:<10} {X.shape[1]:>5}-d crops -> {len(uf)} frame descriptors, "
              f"bundled k={len(keep)}", flush=True)

    # ---------- players A + B: frames -------------------------------------
    sel = np.linspace(0, n_all - 1, min(args.frames, n_all)).astype(int)
    print(f"[frames] exporting {len(sel)} frames ...", flush=True)
    frames, steps = [], []
    for n, i in enumerate(sel):
        row = ds[int(order[i])]
        img = row["image"]
        W, H = img.size
        frames.append(jpeg_uri(img, args.img_width, args.quality))
        fi = int(idx_s[i])
        rows = dets.get(fi, [])
        boxes = [{"c": r[0], "p": round(r[1], 2),
                  "b": [round(r[2] / W, 4), round(r[3] / H, 4),
                        round(r[4] / W, 4), round(r[5] / H, 4)]} for r in rows]

        # --- decode this frame's position under each encoder's memory
        truth = [float(px[i]), float(py[i])]
        per = {}
        for nm, mm in MEM.items():
            w = np.searchsorted(mm["frames"], fi)
            if w >= len(mm["frames"]) or mm["frames"][w] != fi:
                continue                                # no detections in this frame
            resid = mm["M"] / mm["Z"][w]                # unbind the frame descriptor
            fld = (G.conj() @ resid).real / args.hd_dim
            b64, _lo, _hi = quant(fld.reshape(args.grid, args.grid))
            j = int(np.argmax(fld))
            pk = [float(gx[j % args.grid]), float(gy[j // args.grid])]
            per[nm] = {"fld": b64, "peak": [round(pk[0], 3), round(pk[1], 3)],
                       "err": round(float(np.hypot(pk[0] - truth[0], pk[1] - truth[1])), 3)}

        steps.append({"f": fi, "t": float((ts_s[i] - ts_s[0]) / 1e9),
                      "x": round(truth[0], 3), "y": round(truth[1], 3),
                      "yaw": round(float(pyaw[i]), 4),
                      "nd": len(rows), "boxes": boxes, "e": per})
        if n % 25 == 0:
            print(f"  frame {n}/{len(sel)}", flush=True)

    err_stats = {}
    for nm in MEM:
        e = np.array([s["e"][nm]["err"] for s in steps if nm in s["e"]])
        if len(e):
            err_stats[nm] = {"median": float(np.median(e)), "mean": float(e.mean()),
                             "p90": float(np.percentile(e, 90)), "max": float(e.max()),
                             "n": int(len(e))}
            print(f"[recall] {nm:<10} median {np.median(e):.3f} m  mean {e.mean():.3f} m  "
                  f"p90 {np.percentile(e, 90):.3f} m  (n={len(e)})", flush=True)

    # ---------- player C: encoder retrieval -------------------------------
    have = avail
    cmp_rows, enc_meta = [], []
    if len(have) >= 2:
        base = torch.load(os.path.join(out_dir, "crop_embeddings.pt"), weights_only=False)
        ccls = base["class_id"].numpy(); cfr = base["frame_idx"].numpy()
        import csv as _csv
        names, boxes_c = {}, []
        with open(os.path.join(out_dir, "detections_crops.csv"), newline="") as f:
            for r in _csv.DictReader(f):
                names[int(r["class_id"])] = r["class_name"]
                boxes_c.append((int(r["frame_idx"]), float(r["x1"]), float(r["y1"]),
                                float(r["x2"]), float(r["y2"])))

        E = {}
        for nm, fn, _tg in have:
            X = torch.load(os.path.join(out_dir, fn), weights_only=False)["embedding"].numpy()
            E[nm] = (X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)).astype(np.float32)
            enc_meta.append({"name": nm, "dim": int(X.shape[1]), "tag": _tg,
                             "recall": err_stats.get(nm),
                             "k": MEM[nm]["k"] if nm in MEM else None})

        # queries: spread across classes, preferring big boxes (legible thumbnails)
        area = np.array([(b[3] - b[1]) * (b[4] - b[2]) for b in boxes_c])
        rng = np.random.RandomState(args.seed)
        picks, per = [], max(1, args.queries // max(1, len(np.unique(ccls))))
        for c in np.unique(ccls):
            cand = np.where(ccls == c)[0]
            cand = cand[np.argsort(-area[cand])][:max(6, per * 3)]
            picks.extend(rng.choice(cand, size=min(per, len(cand)), replace=False).tolist())
        rng.shuffle(picks)
        picks = picks[:args.queries]

        GAP = 300
        pos_map = {int(v): i for i, v in enumerate(idx_all)}
        cache = {}

        def thumb(row_i):
            if row_i in cache:
                return cache[row_i]
            fi, x1, y1, x2, y2 = boxes_c[row_i]
            di = pos_map.get(int(fi))
            if di is None:
                return None
            img = ds[int(di)]["image"]; W, H = img.size
            pad = 0.08 * max(x2 - x1, y2 - y1)
            c = img.crop((max(0, int(x1 - pad)), max(0, int(y1 - pad)),
                          min(W, int(x2 + pad)), min(H, int(y2 + pad))))
            cache[row_i] = jpeg_uri(c, args.thumb, 62)
            return cache[row_i]

        print(f"[encoders] {len(picks)} query crops ...", flush=True)
        for qn, q in enumerate(picks):
            qt = thumb(q)
            if qt is None:
                continue
            res = []
            for nm, _fn, _tg in have:
                s = E[nm] @ E[nm][q]
                s[np.abs(cfr - cfr[q]) <= GAP] = -2.0
                j = int(np.argmax(s))
                res.append({"e": nm, "cls": names.get(int(ccls[j]), "?"),
                            "ok": bool(ccls[j] == ccls[q]),
                            "sim": round(float(s[j]), 3), "img": thumb(j)})
            cmp_rows.append({"cls": names.get(int(ccls[q]), "?"), "img": qt, "r": res})
            if qn % 10 == 0:
                print(f"  query {qn}/{len(picks)}", flush=True)

    payload = {
        "scene": scene, "n_all_frames": int(n_all), "n_events": int(n_ev),
        "hd": args.hd_dim, "grid": args.grid, "extent": [round(v, 2) for v in ex],
        "pos_l": args.pos_length_scale,
        "path": [[round(float(a), 2), round(float(b), 2)] for a, b in
                 zip(px[::max(1, n_all // 900)], py[::max(1, n_all // 900)])],
        "err": err_stats, "mem_stride": args.mem_stride, "warn": warn,
        "steps": steps, "frames": frames,
        "encoders": enc_meta, "compare": cmp_rows, "gap": 300,
    }
    out = args.out or os.path.join(out_dir, f"video_{scene}.json")
    with open(out, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    mb = os.path.getsize(out) / 1e6
    print(f"\nwrote {out}  ({mb:.1f} MB)")
    print(f"  frames={len(frames)}  steps={len(steps)}  compare_rows={len(cmp_rows)}")


if __name__ == "__main__":
    main()
