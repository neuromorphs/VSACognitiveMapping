"""Export a STATIC, shareable snapshot of the workshop classroom demo.

Precomputes everything the browser page needs (le-marmotte pattern) so the
replay runs from any static file server -- GitHub Pages, `python -m
http.server`, or a zip -- with no Python backend:

  share/classroom_demo/
    index.html               self-contained viewer (written separately)
    assets/replay.json       per-step records + global meta (extent, traj, ...)
    assets/frames/<k>.jpg    held-out RGB frame, width 480, JPEG q80
    assets/heats.bin         ALL heat fields, one concatenated binary:
                             step k occupies bytes [k*S, (k+1)*S) where
                             S = res_y*res_x, uint8 quantized per step
                             (min/max in replay.json steps[k].hmin/hmax)
    assets/polar.bin         ALL polar curves, stride 181 bytes/step, uint8
                             quantized per step (pmin/pmax in replay.json)
    assets/class_heat.json   per-class "where was <class> seen from?" heat
                             (uint8 quantized, base64) + count + peak

Every 3rd held-out row is exported (~550 steps) to keep the folder small.

Run from the repo root (takes a few minutes -- DemoState alone is ~40 s):
    python vsa_cognitive_mapping/export_static_demo.py
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from workshop_demo import DemoState, WORKSHOP  # noqa: E402 (heavy import ok)

OUT = Path(REPO_ROOT) / "share" / "classroom_demo"
ASSETS = OUT / "assets"
FRAMES = ASSETS / "frames"

STRIDE = 3          # every 3rd held-out row
FRAME_W = 480       # exported frame width
JPEG_Q = 80


def quantize(arr):
    """float array -> (uint8 array, lo, hi); lossless-enough for display."""
    a = np.asarray(arr, dtype=np.float64)
    lo, hi = float(a.min()), float(a.max())
    span = hi - lo
    if span <= 0:
        return np.zeros(a.shape, dtype=np.uint8), lo, hi
    return np.round((a - lo) / span * 255.0).astype(np.uint8), lo, hi


def rnd(x, nd=4):
    return round(float(x), nd)


def dir_size(path):
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


def main():
    t_start = time.perf_counter()
    memory = os.path.join(WORKSHOP, "outputs", "classroom_detections",
                          "associative_memory_stride.pt")
    embeddings = os.path.join(WORKSHOP, "outputs", "classroom_detections",
                              "embeddings.pt")
    state = DemoState(memory, embeddings)

    FRAMES.mkdir(parents=True, exist_ok=True)

    rows = [int(r) for r in state.val_rows[::STRIDE]]
    n_steps = len(rows)
    print(f"exporting {n_steps} steps (every {STRIDE}rd of "
          f"{len(state.val_rows)} held-out rows) -> {OUT}")

    # ---- per-step records + binaries ------------------------------------
    steps = []
    heat_shape = None
    t0 = time.perf_counter()
    with open(ASSETS / "heats.bin", "wb") as hf, \
         open(ASSETS / "polar.bin", "wb") as pf:
        for k, row in enumerate(rows):
            r = state.reverse(row)
            heat = np.asarray(r["heat"])
            if heat_shape is None:
                heat_shape = heat.shape
                print(f"heat field shape: {heat_shape[0]} x {heat_shape[1]}")
            qh, hlo, hhi = quantize(heat)
            hf.write(qh.tobytes())

            pol = np.asarray(r["polar"])
            qp, plo, phi = quantize(pol)
            pf.write(qp.tobytes())

            img = state.rgb[int(row)]["image"].convert("RGB")
            if img.width > FRAME_W:
                img = img.resize((FRAME_W, int(img.height * FRAME_W / img.width)))
            img.save(str(FRAMES / f"{k}.jpg"), format="JPEG", quality=JPEG_Q)

            dets = [{"cls": d["cls"], "conf": rnd(d["conf"], 2),
                     "box": [rnd(v, 1) for v in d["box"]], "cid": d["cid"]}
                    for d in state.dets_json(row)]
            steps.append({
                "row": int(row),
                "truth": {"x": rnd(r["truth"]["x"]), "y": rnd(r["truth"]["y"]),
                          "yaw": rnd(r["truth"]["yaw"])},
                "peak": [rnd(r["peak"][0], 3), rnd(r["peak"][1], 3)],
                "est_yaw": rnd(r["est_yaw"]),
                "time": {"recalled_frame_idx": int(r["time"]["recalled_frame_idx"]),
                         "dt_s": rnd(r["time"]["dt_s"], 2)},
                "hmin": rnd(hlo, 5), "hmax": rnd(hhi, 5),
                "pmin": rnd(plo, 5), "pmax": rnd(phi, 5),
                "dets": dets,
            })
            if (k + 1) % 100 == 0:
                print(f"  step {k + 1}/{n_steps} "
                      f"({time.perf_counter() - t0:.0f}s)")

    ny, nx = int(heat_shape[0]), int(heat_shape[1])

    # ---- global meta + replay.json ---------------------------------------
    traj = np.stack([state.frames["odom_x"], state.frames["odom_y"]], 1)
    traj = traj[:: max(1, len(traj) // 500)]
    meta = {
        "extent": [rnd(v) for v in (*state.x_range, *state.y_range)],
        "res": [ny, nx],                      # heat rows x cols
        "n_steps": n_steps,
        "heat_stride": ny * nx,               # bytes per step in heats.bin
        "polar_stride": len(state.angles),    # bytes per step in polar.bin
        "angles": [rnd(a, 5) for a in state.angles],
        "traj": [[rnd(a, 3), rnd(b, 3)] for a, b in traj],
        "hd": int(state.hd),
        "subset": str(state.saved["subset"]),
        "n_train": int(len(state.train_rows)),
        "n_frames": int(state.frames["n_frames"]),
        "frame_stride": STRIDE,
    }
    with open(ASSETS / "replay.json", "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "steps": steps}, f, separators=(",", ":"))

    # ---- per-class heat fields --------------------------------------------
    print("exporting class heat fields ...")
    classes = {}
    for cname in sorted(state.class_probe, key=lambda c: -state.class_count[c]):
        r = state.where_class(cname)
        q, lo, hi = quantize(np.asarray(r["heat"]))
        classes[cname] = {
            "heat_b64": base64.b64encode(q.tobytes()).decode("ascii"),
            "min": rnd(lo, 5), "max": rnd(hi, 5),
            "peak": [rnd(r["peak"][0], 3), rnd(r["peak"][1], 3)],
            "count": int(r["count"]),
        }
    with open(ASSETS / "class_heat.json", "w", encoding="utf-8") as f:
        json.dump({"res": [ny, nx], "classes": classes},
                  f, separators=(",", ":"))
    print(f"  {len(classes)} classes")

    # ---- verify + size report ----------------------------------------------
    mb = 1024.0 * 1024.0
    missing = []
    for k in range(n_steps):
        if not (FRAMES / f"{k}.jpg").exists():
            missing.append(f"frames/{k}.jpg")
    for name, want in (("heats.bin", n_steps * ny * nx),
                       ("polar.bin", n_steps * len(state.angles))):
        p = ASSETS / name
        if not p.exists():
            missing.append(name)
        elif p.stat().st_size != want:
            missing.append(f"{name} size {p.stat().st_size} != {want}")
    for name in ("replay.json", "class_heat.json"):
        if not (ASSETS / name).exists():
            missing.append(name)
    if missing:
        print("MISSING/BAD:", missing)
        sys.exit(1)

    sizes = {
        "frames": dir_size(FRAMES),
        "heats.bin": (ASSETS / "heats.bin").stat().st_size,
        "polar.bin": (ASSETS / "polar.bin").stat().st_size,
        "replay.json": (ASSETS / "replay.json").stat().st_size,
        "class_heat.json": (ASSETS / "class_heat.json").stat().st_size,
    }
    print("\nexport sizes:")
    for name, s in sizes.items():
        print(f"  {name:16s} {s / mb:8.2f} MB")
    print(f"  {'TOTAL assets':16s} {sum(sizes.values()) / mb:8.2f} MB")
    print(f"\nall {n_steps} steps verified; done in "
          f"{time.perf_counter() - t_start:.0f}s")


if __name__ == "__main__":
    main()
