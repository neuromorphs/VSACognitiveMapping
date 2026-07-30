"""Interactive browser version of the WORKSHOP classroom demo.

Drives the actual Telluride workshop code (``external/VSACognitiveMapping-init-am``)
rather than our port: the memory file is the one written by *their*
``classroom_associative_memory.py build``, and every encode / unbind / cleanup
in this server calls *their* functions (imported straight from the extracted
zip). What their ``demo`` command renders offline to demo.mp4 — held-out frame
| position-similarity heatmap | heading polar — becomes a live page:

  * a timeline you scrub (or auto-play) over the held-out frames, each tick
    running the reverse query ("where/when have I seen this?") live;
  * click anywhere on the map to run the forward query ("what did I see at
    this position?") -> top-k recalled stored frames with thumbnails, their
    `query` command's math.

Prereqs (run inside external/VSACognitiveMapping-init-am with PYTHONPATH=src):
    python scripts/classroom/detect_and_embed_classroom.py --device cpu
    python scripts/classroom/classroom_associative_memory.py build \
        --subset stride --stride 3 --pca-whiten --hd-dim 8192

Run:
    python vsa_cognitive_mapping/workshop_demo.py --port 8020
"""

from __future__ import annotations

import argparse
import base64
import csv
import importlib.util
import io
import json
import os
import sys
import time
from pathlib import Path

import threading

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKSHOP = os.path.join(REPO_ROOT, "external", "VSACognitiveMapping-init-am")

# Their package FIRST on the path so `vsa_cognitive_mapping` resolves to the
# workshop package, not this repo's same-named one.
sys.path.insert(0, os.path.join(WORKSHOP, "src"))

from vsa_cognitive_mapping.vsa import (  # noqa: E402  (THEIR module)
    pca_components, phasor_cross_correlation, random_project_to_phasor,
)


def _load_script(name):
    """Import a workshop script file as a module (they're script-style, not
    packaged), so the server reuses their functions verbatim."""
    path = Path(WORKSHOP) / "scripts" / "classroom" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"workshop_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CAM = _load_script("classroom_associative_memory")


# ---------------------------------------------------------------------------
# App state: everything cmd_demo precomputes, kept resident
# ---------------------------------------------------------------------------

class DemoState:
    def __init__(self, memory_path, embeddings_path, grid_resolution=56, n_angles=181,
                 repo=None, rgb_config="rgb_d455", odom_config="odometry_lio_sam",
                 detections_path=None):
        self.repo = repo or CAM.REPO
        self.rgb_config = rgb_config
        self.odom_config = odom_config
        self.detections_path = Path(detections_path) if detections_path else (
            Path(WORKSHOP) / "outputs" / "classroom_detections" / "detections.csv")
        print(f"loading {memory_path} ...")
        self.saved = torch.load(memory_path, weights_only=True)
        saved = self.saved
        self.hd = saved["hd_dim"]
        self.bases = saved["bases"]
        self.codebook = saved["codebook"]
        self.train_rows = self.codebook["row_idx"].numpy()

        print("loading frame data (their load_frame_data) ...")
        self.frames = CAM.load_frame_data(self.repo, self.rgb_config, self.odom_config,
                                          embeddings_path, None)
        n = self.frames["n_frames"]
        self.val_rows = np.setdiff1d(np.arange(n), self.train_rows)

        # content for EVERY frame with the saved W (+ saved PCA setting) --
        # exactly their cmd_demo re-projection
        content_input = self.frames["embeddings_t"]
        if saved["pca_whiten"]:
            whitened, _ = pca_components(content_input.numpy(), saved["pca_components"])
            content_input = torch.from_numpy(whitened.astype(np.float32))
        content, _ = random_project_to_phasor(content_input, d=self.hd, W=saved["W"])
        self.content = content.numpy()

        # decode probes (their grid/sweep helpers)
        pad = 1.0
        self.x_range = (float(self.frames["odom_x"].min() - pad), float(self.frames["odom_x"].max() + pad))
        self.y_range = (float(self.frames["odom_y"].min() - pad), float(self.frames["odom_y"].max() + pad))
        self.res = grid_resolution
        self.grid_x, self.grid_y, self.grid_codes = CAM.position_grid_codes(
            self.x_range, self.y_range, grid_resolution, self.hd,
            self.bases["x_seed"], self.bases["y_seed"], self.bases["pos_length_scale"])
        self.angles, self.angle_codes = CAM.heading_sweep_codes(
            self.hd, self.bases["yaw_seed"], self.bases["yaw_max_freq"], n_angles)
        self.train_time_codes = CAM.encode_time(
            self.codebook["row_idx"].numpy(), self.hd,
            self.bases["time_seed"], self.bases["time_length_scale"])

        self.M_pos = saved["memory_position"].numpy()
        self.M_head = saved["memory_heading"].numpy()
        self.M_time = saved["memory_time"].numpy()
        self.codebook_content = self.codebook["content"].numpy()

        # --- fast decode path -------------------------------------------------
        # Their phasor_cross_correlation unit-normalizes BOTH matrices on every
        # call -- for the grid that re-touches ~400 MB of complex128 per tick.
        # Hoist the invariant: unit-normalize + conjugate each decoder matrix
        # ONCE (complex64); per query only the 8192-elt residual is normalized.
        # _xcorr_fast(dec, r) == phasor_cross_correlation(r[None,:], mat)[0].
        def _unit_conj(mat):
            m = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8)
            return np.ascontiguousarray(np.conj(m).astype(np.complex64))
        self.G_dec = _unit_conj(self.grid_codes)
        self.A_dec = _unit_conj(self.angle_codes)
        self.T_dec = _unit_conj(self.train_time_codes)
        self.CB_dec = _unit_conj(self.codebook_content)

        det_path = self.detections_path
        self.det_by_frame = CAM.load_detections_by_frame(det_path) if det_path.exists() else {}

        # --- class -> "where have I seen this?" probes --------------------------
        # For each class, phase-normalize the bundle of STORED frames' content
        # whose detections include that class: q_c = exp(i*angle(sum content)).
        # Unit-modulus, so it is a safe unbinding key for M_pos; the residual's
        # grid sweep lights up the vantage points where that class was seen.
        row_of_fidx = {int(f): i for i, f in enumerate(self.frames["dataset_frame_idx"])}
        train_set = set(int(r) for r in self.train_rows)
        cls_rows = {}
        for fidx, df in self.det_by_frame.items():
            row = row_of_fidx.get(int(fidx))
            if row is None or row not in train_set:
                continue
            for cname in df["class_name"].unique():
                cls_rows.setdefault(cname, []).append(row)
        self.class_probe, self.class_count = {}, {}
        for cname, rows in cls_rows.items():
            if len(rows) < 5:
                continue
            bundle = self.content[np.array(rows)].sum(axis=0)
            mag = np.abs(bundle)
            probe = np.where(mag > 1e-12, bundle / np.maximum(mag, 1e-12), 1.0 + 0j)
            self.class_probe[cname] = probe
            self.class_count[cname] = len(rows)

        print("loading RGB dataset handle ...")
        from datasets import load_dataset
        self.rgb = load_dataset(self.repo, self.rgb_config, split="train").sort("timestamp_ns")
        self._thumb_cache = {}
        print(f"ready: {n} frames ({len(self.train_rows)} stored / {len(self.val_rows)} held out), "
              f"hd={self.hd}, grid {grid_resolution}x{grid_resolution}")

    # ---- helpers ---------------------------------------------------------
    def frame_jpeg(self, row, width=480):
        key = (int(row), width)
        if key not in self._thumb_cache:
            img = self.rgb[int(row)]["image"].convert("RGB")
            if width and img.width > width:
                img = img.resize((width, int(img.height * width / img.width)))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=82)
            self._thumb_cache[key] = base64.b64encode(buf.getvalue()).decode()
            # all 2478 frames at width 480 is ~100 MB — keep them all so
            # playback never re-decodes; only evict beyond a generous cap.
            if len(self._thumb_cache) > 3000:
                self._thumb_cache.pop(next(iter(self._thumb_cache)))
        return self._thumb_cache[key]

    def warm_thumbs(self, rows):
        """Background thread: pre-encode every frame's JPEG once so playback
        ticks never pay the PNG-decode + JPEG-encode cost (~60-90 ms/frame)."""
        t0 = time.perf_counter()
        for i, row in enumerate(rows):
            try:
                self.frame_jpeg(int(row))
            except Exception:
                pass
            if (i + 1) % 500 == 0:
                print(f"  thumb warmup {i + 1}/{len(rows)} "
                      f"({time.perf_counter() - t0:.0f}s)")
        print(f"thumb warmup done: {len(rows)} frames in "
              f"{time.perf_counter() - t0:.0f}s")

    def dets_json(self, row):
        fidx = int(self.frames["dataset_frame_idx"][int(row)])
        df = self.det_by_frame.get(fidx)
        if df is None:
            return []
        return [{"cls": r["class_name"], "conf": float(r["confidence"]),
                 "box": [float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])],
                 "cid": int(r["class_id"])} for _, r in df.iterrows()]

    # ---- fast xcorr (identical math to their phasor_cross_correlation) ----
    @staticmethod
    def _xcorr_fast(dec, residual):
        r = (residual / (np.linalg.norm(residual) + 1e-8)).astype(np.complex64)
        return np.real(dec @ r)

    # ---- their demo math, per scrub tick -----------------------------------
    def reverse(self, row):
        t0 = time.perf_counter()
        q = self.content[int(row)]
        residual_pos = self.M_pos / q
        heat = self._xcorr_fast(self.G_dec, residual_pos)
        heat = heat.reshape(self.res, self.res)
        pr, pc = np.unravel_index(int(np.argmax(heat)), heat.shape)

        residual_head = self.M_head / q
        pol = self._xcorr_fast(self.A_dec, residual_head)

        residual_time = self.M_time / q
        tsims = self._xcorr_fast(self.T_dec, residual_time)
        bi = int(np.argmax(tsims))
        recalled_row = int(self.codebook["row_idx"][bi])
        dt_s = abs(int(self.frames["timestamp_ns"][int(row)]) -
                   int(self.frames["timestamp_ns"][recalled_row])) / 1e9

        ms = (time.perf_counter() - t0) * 1e3
        return {
            "heat": np.round(heat, 5).tolist(),
            "heat_min": float(heat.min()), "heat_max": float(heat.max()),
            "peak": [float(self.grid_x[pc]), float(self.grid_y[pr])],
            "polar": np.round(pol, 5).tolist(),
            "est_yaw": float(self.angles[int(np.argmax(pol))]),
            "time": {"recalled_row": recalled_row, "sim": float(tsims[bi]),
                     "recalled_frame_idx": int(self.codebook["frame_idx"][bi]),
                     "dt_s": float(dt_s)},
            "truth": {"x": float(self.frames["x"][int(row)]), "y": float(self.frames["y"][int(row)]),
                      "yaw": float(self.frames["yaw"][int(row)])},
            "ms": ms,
        }

    # ---- class -> location: "where have I seen <class>?" -------------------
    def where_class(self, cname):
        t0 = time.perf_counter()
        probe = self.class_probe[cname]
        residual = self.M_pos / probe
        heat = self._xcorr_fast(self.G_dec, residual).reshape(self.res, self.res)
        pr, pc = np.unravel_index(int(np.argmax(heat)), heat.shape)
        return {
            "heat": np.round(heat, 5).tolist(),
            "heat_min": float(heat.min()), "heat_max": float(heat.max()),
            "peak": [float(self.grid_x[pc]), float(self.grid_y[pr])],
            "count": self.class_count[cname],
            "ms": (time.perf_counter() - t0) * 1e3,
        }

    # ---- their query math (forward direction) -------------------------------
    def forward(self, x, y, k=5):
        ctx = CAM.encode_position(np.array([[x, y]]), self.hd, self.bases["x_seed"],
                                  self.bases["y_seed"], self.bases["pos_length_scale"])[0]
        sims = self._xcorr_fast(self.CB_dec, self.M_pos / ctx)
        order = np.argsort(sims)[::-1][:k]
        out = []
        for idx in order:
            row = int(self.codebook["row_idx"][idx])
            out.append({"row": row, "sim": float(sims[idx]),
                        "x": float(self.codebook["x"][idx]), "y": float(self.codebook["y"][idx]),
                        "yaw": float(self.codebook["yaw"][idx]),
                        "img": self.frame_jpeg(row, width=200),
                        "dets": [d["cls"] for d in self.dets_json(row)][:4]})
        return out


# ---------------------------------------------------------------------------
# ASTM: multi-trace spatio-temporal memory (lazy-loaded on first request)
# ---------------------------------------------------------------------------
# Import caution: WORKSHOP/src sits at sys.path[0], so `vsa_cognitive_mapping`
# in sys.modules is THEIR package -- and our astm_traces.py imports
# `vsa_cognitive_mapping.classroom_pipeline`, which their package lacks.
# Fix: swap their package entries out of sys.modules, put the repo root first
# on sys.path, exec astm_traces.py by file path (module name
# "astm_traces_local"), then restore. The loaded module keeps direct
# references to everything it needs, so the restore is safe.

_ASTM = {"st": None, "err": None, "lock": threading.Lock()}
ASTM_PT = os.path.join(REPO_ROOT, "outputs", "classroom", "astm_traces.pt")


def _astm_load():
    saved = {k: sys.modules.pop(k) for k in list(sys.modules)
             if k.split(".")[0] == "vsa_cognitive_mapping"}
    sys.path.insert(0, REPO_ROOT)
    try:
        path = os.path.join(REPO_ROOT, "vsa_cognitive_mapping", "astm_traces.py")
        spec = importlib.util.spec_from_file_location("astm_traces_local", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["astm_traces_local"] = mod
        spec.loader.exec_module(mod)
    finally:
        for k in list(sys.modules):
            if k.split(".")[0] == "vsa_cognitive_mapping":
                del sys.modules[k]
        sys.modules.update(saved)
        while REPO_ROOT in sys.path:
            sys.path.remove(REPO_ROOT)
    if not os.path.exists(ASTM_PT):
        raise FileNotFoundError(
            f"{ASTM_PT} missing -- run `python vsa_cognitive_mapping/astm_traces.py "
            f"export` then `build --hd-dim 8192`")
    print(f"loading ASTM traces {ASTM_PT} (grid=56) ...")
    tr = mod.TraceSet.load(ASTM_PT, grid=56)  # grid=56: keep decoder grids moderate
    print(f"ASTM ready: {tr.n_events} events, {len(tr.classes)} classes, hd={tr.hd}")
    return {"tr": tr, "router": mod.QueryRouter(tr)}


def _astm():
    with _ASTM["lock"]:
        if _ASTM["st"] is None and _ASTM["err"] is None:
            try:
                _ASTM["st"] = _astm_load()
            except Exception as e:  # noqa: BLE001 - surfaced as JSON to client
                _ASTM["err"] = f"{type(e).__name__}: {e}"
    return _ASTM["st"], _ASTM["err"]


def _astm_when(s):
    if not s:
        return None
    if ":" in s:
        a, b = s.split(":")
        return (float(a), float(b))
    return float(s)


def _astm_extras(tr, decode, what, where, when):
    """Full similarity field / curve / ranking for the panel (the router only
    returns the peak). Mirrors QueryRouter's routing rules exactly."""
    C = tr.C[what] if what is not None else None
    S = tr.enc.ctx_pos(*where).values if where is not None else None
    T_point, T_probe = None, None
    if when is not None:
        if isinstance(when, (tuple, list)):
            T_probe = tr.time_range(float(when[0]), float(when[1]))
        else:
            T_point = tr.enc.ctx_time(float(when)).values

    if decode == "where":
        if C is not None and (T_point is not None or T_probe is not None):
            residual = tr.M["event"] / C
            if T_point is not None:
                residual = residual / T_point
        elif C is not None:
            residual = tr.M["what_where"] / C
        else:
            residual = tr.M["where_when"] / T_point if T_point is not None \
                else tr.M["where_when"]
        probe = tr.G_flat if T_probe is None \
            else tr.G_flat * T_probe[None, :].astype(np.complex64)
        s = np.real(probe @ np.conj(residual.astype(np.complex64))) / tr.hd
        field = s.reshape(len(tr.gy), len(tr.gx))
        return {"field": np.round(field, 4).tolist(),
                "field_min": float(field.min()), "field_max": float(field.max()),
                "gx": [float(tr.gx[0]), float(tr.gx[-1]), len(tr.gx)],
                "gy": [float(tr.gy[0]), float(tr.gy[-1]), len(tr.gy)]}

    if decode == "when":
        if C is not None and S is not None:
            residual = tr.M["event"] / C / S
        elif C is not None:
            residual = tr.M["what_when"] / C
        else:
            residual = tr.M["where_when"] / S
        s = np.real(tr.T_axis @ np.conj(residual.astype(np.complex64))) / tr.hd
        step = max(1, int(np.ceil(len(s) / 400.0)))
        return {"curve": np.round(s[::step], 4).tolist(), "t_step": step,
                "t_argmax": int(np.argmax(s)), "t_axis_max": int(len(s) - 1)}

    # decode == "what"
    if S is not None and (T_point is not None or T_probe is not None):
        residual = tr.M["event"] / S
        if T_point is not None:
            residual = residual / T_point
    elif S is not None:
        residual = tr.M["what_where"] / S
    else:
        residual = tr.M["what_when"] / T_point if T_point is not None \
            else tr.M["what_when"]
    probe = tr.C_mat if T_probe is None \
        else tr.C_mat * T_probe[None, :].astype(np.complex64)
    s = np.real(probe @ np.conj(residual.astype(np.complex64))) / tr.hd
    order = np.argsort(s)[::-1][:8]
    return {"ranked": [{"cls": tr.classes[int(i)], "sim": float(s[int(i)])}
                       for i in order]}


def astm_meta():
    st, err = _astm()
    if err:
        return {"error": err}
    tr = st["tr"]
    return {"classes": list(tr.classes), "t_max": float(tr.t_max),
            "n_events": int(tr.n_events), "hd": int(tr.hd),
            "bounds": [float(b) for b in tr.bounds],
            "gx": [float(tr.gx[0]), float(tr.gx[-1]), len(tr.gx)],
            "gy": [float(tr.gy[0]), float(tr.gy[-1]), len(tr.gy)]}


def astm_query(q):
    st, err = _astm()
    if err:
        return {"error": err}
    tr, router = st["tr"], st["router"]
    decode = q.get("decode", ["where"])[0]
    what = q.get("what", [""])[0] or None
    xs, ys = q.get("x", [""])[0], q.get("y", [""])[0]
    where = (float(xs), float(ys)) if xs != "" and ys != "" else None
    try:
        when = _astm_when(q.get("when", [""])[0])
        ans, info = router.query(decode, what=what, where=where, when=when)
    except (ValueError, KeyError) as e:
        return {"error": str(e)}
    out = {"decode": decode, "trace": info["trace"], "sim": float(info["sim"]),
           "answer": [float(a) for a in ans] if isinstance(ans, tuple) else ans,
           "ms": {k[3:]: round(info[k], 2) for k in
                  ("ms_encode", "ms_unbind", "ms_decode", "ms_total")}}
    out.update(_astm_extras(tr, decode, what, where, when))
    return out


def astm_moved(q):
    st, err = _astm()
    if err:
        return {"error": err}
    wa = _astm_when(q.get("wa", ["0:413"])[0])
    wb = _astm_when(q.get("wb", ["826:1238"])[0])
    if not isinstance(wa, tuple) or not isinstance(wb, tuple):
        return {"error": "wa and wb must be ranges a:b"}
    t0 = time.perf_counter()
    rows = st["router"].which_moved(wa, wb, min_sim=0.0)[:5]
    return {"top": rows, "wa": list(wa), "wb": list(wb),
            "ms": (time.perf_counter() - t0) * 1e3}


# ---------------------------------------------------------------------------
# Semantic map sources (lazy-loaded on first request, like ASTM)
# ---------------------------------------------------------------------------
# Two extra memories built by our classroom_pipeline.py:
#   memory_object.pt -- semantic_position bound to the OBJECT's own depth-
#     deprojected world (x, y) ("where ARE the chairs?"), not the robot's
#     vantage pose like the workshop memory above;
#   memory_now.pt -- decaying working memory M_now snapshots (per-class
#     forgetting factor), written by `build_now`.
# Both were encoded with classroom_pipeline's ClassroomEncoders(hd, seed+100,
# pos_l, time_l) bases -- NOT this demo's workshop bases -- so decoding needs
# that module. Same import caution as ASTM: load it by file path under a
# private name with their same-named package swapped out of sys.modules.

SEM_DIR = os.path.join(REPO_ROOT, "outputs", "classroom")
SEM_OBJ_PT = os.path.join(SEM_DIR, "memory_object.pt")
SEM_NOW_PT = os.path.join(SEM_DIR, "memory_now.pt")
# share the ASTM lock: both lazy loaders do the sys.modules/sys.path swap
# dance, which must never run concurrently (loads never nest, so no deadlock)
_SEM = {"cp": None, "obj": None, "obj_err": None, "now": None, "now_err": None,
        "lock": _ASTM["lock"]}


def _sem_cp():
    """classroom_pipeline.py loaded by file path (ClassroomEncoders, Phasor,
    _class_seed), cached; caller holds _SEM['lock']."""
    if _SEM["cp"] is not None:
        return _SEM["cp"]
    saved = {k: sys.modules.pop(k) for k in list(sys.modules)
             if k.split(".")[0] == "vsa_cognitive_mapping"}
    sys.path.insert(0, REPO_ROOT)
    try:
        path = os.path.join(REPO_ROOT, "vsa_cognitive_mapping", "classroom_pipeline.py")
        spec = importlib.util.spec_from_file_location("classroom_pipeline_local", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["classroom_pipeline_local"] = mod
        spec.loader.exec_module(mod)
    finally:
        for k in list(sys.modules):
            if k.split(".")[0] == "vsa_cognitive_mapping":
                del sys.modules[k]
        sys.modules.update(saved)
        while REPO_ROOT in sys.path:
            sys.path.remove(REPO_ROOT)
    _SEM["cp"] = mod
    return mod


def _sem_grid_dec(enc, x, y, width=64, pad=1.0):
    """Position-grid decoder over the truth bounds + pad, mirroring the
    pipeline's _pos_field axes. ctx_pos(x, y) = exp(i*(x*ax + y*ay)/l) with
    ax/ay the base phase angles, so the whole grid is two outer products.
    Conjugated complex64 up front (the G_dec trick above), computed once:
    field = Re(G @ residual) / hd, identical to the pipeline's decode."""
    gx = np.linspace(x.min() - pad, x.max() + pad, width)
    ratio = (y.max() - y.min() + 2 * pad) / (x.max() - x.min() + 2 * pad)
    gy = np.linspace(y.min() - pad, y.max() + pad, max(8, int(width * ratio)))
    ax = (np.angle(enc.Bx.values) / enc.pos_l).astype(np.float32)
    ay = (np.angle(enc.By.values) / enc.pos_l).astype(np.float32)
    X, Y = np.meshgrid(gx.astype(np.float32), gy.astype(np.float32))
    phases = np.outer(X.ravel(), ax) + np.outer(Y.ravel(), ay)
    G = (np.cos(phases) - 1j * np.sin(phases)).astype(np.complex64)  # conj
    return gx, gy, np.ascontiguousarray(G)


def _sem_obj_load():
    if not os.path.exists(SEM_OBJ_PT):
        raise FileNotFoundError(
            f"{SEM_OBJ_PT} missing -- run `python vsa_cognitive_mapping/"
            f"classroom_pipeline.py build --place-mode object "
            f"--memory-name memory_object.pt`")
    cp = _sem_cp()
    print(f"loading semantic object memory {SEM_OBJ_PT} ...")
    d = torch.load(SEM_OBJ_PT, weights_only=True)
    cfg = d["config"]
    enc = cp.ClassroomEncoders(cfg["hd_dim"], cfg["seed"] + 100,
                               cfg["pos_length_scale"], cfg["time_length_scale"])
    x = d["truth"]["x"].numpy()
    y = d["truth"]["y"].numpy()
    gx, gy, G = _sem_grid_dec(enc, x, y)
    counts = {}  # per-class detection counts, for the panel readout
    det_csv = os.path.join(SEM_DIR, "detections.csv")
    if os.path.exists(det_csv):
        with open(det_csv) as f:
            for r in csv.DictReader(f):
                counts[r["class_name"]] = counts.get(r["class_name"], 0) + 1
    st = {"M": d["memories"]["semantic_position"].numpy(),
          "codebook": {c: v.numpy() for c, v in d["semantic_codebook"].items()},
          "counts": counts, "cfg": cfg, "hd": cfg["hd_dim"],
          "gx": gx, "gy": gy, "G": G}
    print(f"semantic object memory ready: {len(st['codebook'])} classes, "
          f"hd={cfg['hd_dim']}, grid {len(gx)}x{len(gy)}, "
          f"place_mode={cfg.get('place_mode', 'robot')}")
    return st


def _sem_now_load():
    if not os.path.exists(SEM_NOW_PT):
        raise FileNotFoundError(
            f"{SEM_NOW_PT} missing -- run `python vsa_cognitive_mapping/"
            f"classroom_pipeline.py build_now`")
    cp = _sem_cp()
    print(f"loading working memory {SEM_NOW_PT} ...")
    d = torch.load(SEM_NOW_PT, weights_only=True)
    cfg = d["config"]
    classes = list(d["classes"])
    # the codebook is not stored -- rebuild it the way build_now does
    # (deterministic per-class md5 seed, identical across processes)
    codebook = {c: cp.Phasor(dim=cfg["hd_dim"], seed=cp._class_seed(c)).values
                for c in classes}
    enc = cp.ClassroomEncoders(cfg["hd_dim"], cfg["seed"] + 100,
                               cfg["pos_length_scale"], cfg["time_length_scale"])
    # build_now saves no truth arrays; reuse the object memory's grid when the
    # encoder params match (same dataset -> same bounds), else rebuild from the
    # source memory's truth.
    st_obj = _SEM["obj"]
    if (st_obj is not None
            and st_obj["cfg"]["hd_dim"] == cfg["hd_dim"]
            and st_obj["cfg"]["seed"] == cfg["seed"]
            and st_obj["cfg"]["pos_length_scale"] == cfg["pos_length_scale"]):
        gx, gy, G = st_obj["gx"], st_obj["gy"], st_obj["G"]
    else:
        truth_path = os.path.join(SEM_DIR, cfg.get("source_memory", "memory.pt"))
        if not os.path.exists(truth_path):
            truth_path = SEM_OBJ_PT
        if not os.path.exists(truth_path):
            raise FileNotFoundError(
                f"no truth source for the now-grid ({truth_path} missing) -- "
                f"run `python vsa_cognitive_mapping/classroom_pipeline.py build`")
        t = torch.load(truth_path, weights_only=True)["truth"]
        gx, gy, G = _sem_grid_dec(enc, t["x"].numpy(), t["y"].numpy())
    st = {"snaps": d["snapshots_normed"].numpy(),
          "w": d["snapshot_weights"].numpy(),
          "rows": [int(r) for r in d["snapshot_rows"].numpy()],
          "classes": classes, "ci": {c: i for i, c in enumerate(classes)},
          "codebook": codebook, "cfg": cfg, "hd": cfg["hd_dim"],
          "gx": gx, "gy": gy, "G": G}
    print(f"working memory ready: {len(st['rows'])} snapshots at rows "
          f"{st['rows']}, {len(classes)} classes, hd={cfg['hd_dim']}")
    return st


def _sem_get(key):
    load = {"obj": _sem_obj_load, "now": _sem_now_load}[key]
    ek = key + "_err"
    with _SEM["lock"]:
        if _SEM[key] is None and _SEM[ek] is None:
            try:
                _SEM[key] = load()
            except Exception as e:  # noqa: BLE001 - surfaced as JSON to client
                _SEM[ek] = f"{type(e).__name__}: {e}"
    return _SEM[key], _SEM[ek]


def _sem_field(st, residual):
    """Decode a residual over the memory's grid -> (field, peak_xy)."""
    field = (np.real(st["G"] @ residual.astype(np.complex64)) / st["hd"]) \
        .reshape(len(st["gy"]), len(st["gx"]))
    pr, pc = np.unravel_index(int(np.argmax(field)), field.shape)
    return field, [float(st["gx"][pc]), float(st["gy"][pr])]


def _sem_field_json(st, field, peak):
    return {"field": np.round(field, 5).tolist(),
            "field_min": float(field.min()), "field_max": float(field.max()),
            "gx": [float(st["gx"][0]), float(st["gx"][-1]), len(st["gx"])],
            "gy": [float(st["gy"][0]), float(st["gy"][-1]), len(st["gy"])],
            "peak": peak}


def sem_meta():
    out = {}
    st, err = _sem_get("obj")
    out["object"] = {"error": err} if err else {
        "classes": sorted(st["codebook"], key=lambda c: -st["counts"].get(c, 0)),
        "counts": st["counts"], "hd": st["hd"],
        "place_mode": st["cfg"].get("place_mode", "robot")}
    st, err = _sem_get("now")
    out["now"] = {"error": err} if err else {
        "classes": st["classes"], "snapshots": st["rows"],
        "lambda_static": st["cfg"]["lambda_static"],
        "lambda_dynamic": st["cfg"]["lambda_dynamic"],
        "dynamic_classes": st["cfg"]["dynamic_classes"]}
    return out


def sem_where(q, state):
    cls = q.get("cls", [""])[0]
    mode = q.get("mode", ["vantage"])[0]
    if mode == "vantage":
        # delegate: identical math to /api/where_class (kept for compat),
        # repackaged into the field/gx/gy shape the sem panel renders
        if cls not in state.class_probe:
            return {"error": f"unknown class '{cls}'",
                    "classes": sorted(state.class_probe)}
        r = state.where_class(cls)
        return {"mode": "vantage",
                "field": r["heat"], "field_min": r["heat_min"],
                "field_max": r["heat_max"],
                "gx": [float(state.grid_x[0]), float(state.grid_x[-1]), state.res],
                "gy": [float(state.grid_y[0]), float(state.grid_y[-1]), state.res],
                "peak": r["peak"], "count": r["count"], "ms": r["ms"]}
    if mode != "object":
        return {"error": "mode must be vantage|object"}
    st, err = _sem_get("obj")
    if err:
        return {"error": err}
    if cls not in st["codebook"]:
        return {"error": f"unknown class '{cls}'", "classes": sorted(st["codebook"])}
    t0 = time.perf_counter()
    field, peak = _sem_field(st, st["M"] / st["codebook"][cls])
    out = {"mode": "object", "count": int(st["counts"].get(cls, 0)),
           "ms": (time.perf_counter() - t0) * 1e3}
    out.update(_sem_field_json(st, field, peak))
    return out


def sem_now(q):
    st, err = _sem_get("now")
    if err:
        return {"error": err}
    cls = q.get("cls", [""])[0]
    if cls not in st["codebook"]:
        return {"error": f"unknown class '{cls}'", "classes": st["classes"]}
    k = int(q.get("snap", [str(len(st["rows"]) - 1)])[0])
    if not 0 <= k < len(st["rows"]):
        return {"error": f"snap must be in 0..{len(st['rows']) - 1}",
                "snapshots": st["rows"]}
    t0 = time.perf_counter()
    field, peak = _sem_field(st, st["snaps"][k] / st["codebook"][cls])
    w = float(st["w"][k, st["ci"][cls]])
    out = {"snap": k, "row": st["rows"][k], "snapshots": st["rows"],
           "w": w, "forgotten": bool(w < 1e-3),  # decayed evidence mass ~0:
           # below the build_now normalization floor the field is crosstalk
           "ms": (time.perf_counter() - t0) * 1e3}
    out.update(_sem_field_json(st, field, peak))
    return out


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

def serve(state: DemoState, port: int):
    import http.server
    import socketserver
    from urllib.parse import urlparse, parse_qs

    traj = np.stack([state.frames["odom_x"], state.frames["odom_y"]], 1)
    traj = traj[:: max(1, len(traj) // 500)]
    meta = {
        "n_frames": int(state.frames["n_frames"]),
        "val_rows": [int(v) for v in state.val_rows],
        "extent": [*state.x_range, *state.y_range],
        "res": state.res,
        "angles": [float(a) for a in state.angles],
        "traj": [[float(a), float(b)] for a, b in traj],
        "hd": int(state.hd),
        "subset": state.saved["subset"],
        "n_train": int(len(state.train_rows)),
        "classes": [{"name": c, "count": state.class_count[c]}
                    for c in sorted(state.class_probe, key=lambda c: -state.class_count[c])],
    }

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, obj, ctype="application/json"):
            body = obj.encode() if isinstance(obj, str) else json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            try:
                if u.path in ("/", "/index.html"):
                    self._send(PAGE, "text/html; charset=utf-8")
                elif u.path == "/api/meta":
                    self._send(meta)
                elif u.path == "/api/frame":
                    row = int(q["row"][0])
                    self._send({"img": state.frame_jpeg(row), "dets": state.dets_json(row)})
                elif u.path == "/api/reverse":
                    self._send(state.reverse(int(q["row"][0])))
                elif u.path == "/api/tick":
                    # combined frame + reverse in one round-trip (playback path)
                    row = int(q["row"][0])
                    self._send({"frame": {"img": state.frame_jpeg(row),
                                          "dets": state.dets_json(row)},
                                "reverse": state.reverse(row)})
                elif u.path == "/api/query":
                    self._send({"top": state.forward(float(q["x"][0]), float(q["y"][0]))})
                elif u.path == "/api/where_class":
                    self._send(state.where_class(q["cls"][0]))
                elif u.path == "/api/sem/meta":
                    self._send(sem_meta())
                elif u.path == "/api/sem/where":
                    self._send(sem_where(q, state))
                elif u.path == "/api/sem/now":
                    self._send(sem_now(q))
                elif u.path == "/api/astm/meta":
                    self._send(astm_meta())
                elif u.path == "/api/astm/query":
                    self._send(astm_query(q))
                elif u.path == "/api/astm/moved":
                    self._send(astm_moved(q))
                else:
                    self.send_response(404); self.end_headers()
            except Exception as e:  # noqa: BLE001 - report to client during dev
                self.send_response(500)
                self.end_headers()
                self.wfile.write(str(e).encode())

    # pre-encode frame JPEGs in the background (held-out playback rows first)
    warm_order = list(state.val_rows) + list(state.train_rows)
    threading.Thread(target=state.warm_thumbs, args=(warm_order,), daemon=True).start()

    with socketserver.ThreadingTCPServer(("127.0.0.1", port), Handler) as httpd:
        httpd.daemon_threads = True
        print(f"\n  workshop interactive demo ->  http://localhost:{port}\n  Ctrl+C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped.")


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Workshop VSA Memory — Live Demo</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
 :root{--bg:#101318;--panel:#171c24;--line:#262e3a;--ink:#e8edf3;--mut:#93a1b3;
       --blue:#2a78d6;--red:#e34948;}
 *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--ink);
   font-family:-apple-system,"Segoe UI",system-ui,sans-serif;padding:14px}
 h1{font-size:16px;margin:0 0 2px} .sub{color:var(--mut);font-size:12px;margin:0 0 12px}
 .row{display:flex;gap:12px;flex-wrap:wrap}
 .card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:10px;flex:1;min-width:300px}
 .card h2{font-size:12px;color:var(--mut);margin:0 0 8px;font-weight:600;text-transform:uppercase;letter-spacing:.1em}
 canvas{display:block;width:100%;border-radius:6px;background:#0b0e12}
 .bar{display:flex;align-items:center;gap:10px;margin:12px 0}
 input[type=range]{flex:1}
 button{background:var(--panel);color:var(--ink);border:1px solid var(--line);border-radius:7px;
        padding:6px 14px;cursor:pointer;font-size:13px} button:hover{border-color:var(--blue)}
 #status{color:var(--mut);font-size:12px;min-width:270px}
 #qout{display:flex;gap:8px;flex-wrap:wrap;margin-top:8px}
 .thumb{width:150px;font-size:10.5px;color:var(--mut)}
 .thumb img{width:100%;border-radius:5px;display:block}
 .hint{color:var(--mut);font-size:11.5px;margin-top:6px}
 .aform{display:flex;gap:6px;flex-wrap:wrap;align-items:center;font-size:12px;margin-bottom:8px;color:var(--mut)}
 .aform input,.aform select{background:#0b0e12;color:var(--ink);border:1px solid var(--line);border-radius:6px;padding:4px 6px;font-size:12px}
 .aform input[type=number]{width:62px} #astm_when{width:96px}
 .arow{display:flex;align-items:center;gap:8px;font-size:11.5px;color:var(--mut);margin:2px 0}
 .arow .lbl{width:96px;text-align:right;color:var(--ink)}
 .abar{height:13px;background:var(--blue);border-radius:3px}
</style></head><body>
 <h1>Workshop associative memory — live</h1>
 <p class="sub">Held-out frame → unbind three memories (position / heading / time), the workshop's own code answering "have I been here before?" in real time. Click the map for the forward query.</p>
 <div class="bar">
  <button id="play">▶ play</button>
  <input type="range" id="slider" min="0" max="1" value="0">
  <span id="status">loading…</span>
 </div>
 <div class="row">
  <div class="card" style="max-width:520px"><h2>current view (held out)</h2><canvas id="cv_rgb" width="480" height="360"></canvas></div>
  <div class="card"><h2>recalled position — similarity heatmap
    <select id="clsSel" style="float:right;background:#0b0e12;color:#e8edf3;border:1px solid #262e3a;border-radius:6px;font-size:12px;padding:2px 6px">
      <option value="">— live frames —</option>
    </select>
    <select id="semMode" title="semantic map source for the class query" style="float:right;margin-right:6px;background:#0b0e12;color:#e8edf3;border:1px solid #262e3a;border-radius:6px;font-size:12px;padding:2px 6px">
      <option value="vantage">vantage</option>
      <option value="object">object</option>
      <option value="now">now</option>
    </select></h2>
    <canvas id="cv_heat" width="520" height="430"></canvas>
    <div class="hint" id="heat_hint">✕ ground truth · ● estimated peak · click anywhere: "what did I see here?"</div>
    <div class="bar" id="nowBar" style="display:none;margin:6px 0 0">
      <input type="range" id="nowSnap" min="0" max="7" value="7">
      <span class="hint" id="nowLbl" style="margin-top:0;min-width:150px"></span>
    </div>
    <div id="qout"></div></div>
  <div class="card" style="max-width:430px"><h2>recalled heading — similarity vs angle</h2><canvas id="cv_pol" width="400" height="400"></canvas>
    <div class="hint" id="timeline_hint"></div></div>
  <div class="card" style="max-width:470px"><h2>spatio-temporal memory (ASTM)</h2>
    <div class="aform">
      <select id="astm_cls"><option value="">— any class —</option></select>
      <input id="astm_when" placeholder="t or a:b" title="time: a single row or a range a:b">
      <input id="astm_x" type="number" step="0.1" placeholder="x">
      <input id="astm_y" type="number" step="0.1" placeholder="y">
    </div>
    <div class="aform">decode:
      <label><input type="radio" name="astm_dec" value="where" checked> where</label>
      <label><input type="radio" name="astm_dec" value="when"> when</label>
      <label><input type="radio" name="astm_dec" value="what"> what</label>
      <button id="astm_run">run</button>
      <button id="astm_mv">which moved?</button>
    </div>
    <canvas id="cv_astm" width="420" height="300" style="display:none"></canvas>
    <div id="astm_out" class="hint">loading ASTM traces…</div>
    <div class="hint" id="astm_lat"></div>
    <div class="hint">4 traces (what⊗where, what⊗when, where⊗when, event); the decode target is left blank, the rest condition the query. time unit = <b>event rows (0–1238)</b> of the stride-2 export — not the scrub timeline's frame rows.</div>
  </div>
 </div>
<script>
let META=null, rowList=[], cur=0, playing=false, timer=null;
const $=id=>document.getElementById(id);
const cvR=$("cv_rgb"), cxR=cvR.getContext("2d");
const cvH=$("cv_heat"), cxH=cvH.getContext("2d");
const cvP=$("cv_pol"), cxP=cvP.getContext("2d");

function viridisRGB(t){ // compact 5-stop approximation -> [r,g,b]
 const s=[[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]];
 t=Math.max(0,Math.min(1,t))*(s.length-1); const i=Math.floor(t), f=t-i;
 const a=s[i], b=s[Math.min(i+1,s.length-1)];
 return a.map((v,k)=>Math.round(v+(b[k]-v)*f));}
function viridis(t){const c=viridisRGB(t);return `rgb(${c.join(",")})`;}
// Render a heat matrix into a canvas via ImageData + one scaled drawImage --
// a single blit instead of res*res fillRect calls (the old per-cell path was
// ~3k canvas ops per tick and the main source of playback stutter).
const _heatOff=document.createElement("canvas");
function blitHeat(ctx2d,W,H,z,lo,hi){
 const ny=z.length,nx=z[0].length;
 _heatOff.width=nx;_heatOff.height=ny;
 const og=_heatOff.getContext("2d");
 const im=og.createImageData(nx,ny);
 const span=hi-lo+1e-9;
 for(let j=0;j<ny;j++){const src=z[j],ro=(ny-1-j)*nx*4; // y-flip
  for(let i=0;i<nx;i++){const c=viridisRGB((src[i]-lo)/span),o=ro+i*4;
   im.data[o]=c[0];im.data[o+1]=c[1];im.data[o+2]=c[2];im.data[o+3]=255;}}
 og.putImageData(im,0,0);
 ctx2d.imageSmoothingEnabled=true;
 ctx2d.drawImage(_heatOff,0,0,W,H);}

let classMode=null, semMode="vantage", SEM=null;
const IDLE_HINT='✕ ground truth · ● estimated peak · click anywhere: "what did I see here?"';
async function boot(){
 META=await (await fetch("/api/meta")).json();
 rowList=META.val_rows;
 $("slider").max=rowList.length-1;
 $("slider").oninput=e=>{cur=+e.target.value; tick();};
 $("play").onclick=()=>{playing=!playing; $("play").textContent=playing?"⏸ pause":"▶ play";
   if(playing) loop(); };
 cvH.addEventListener("click", onMapClick);
 fillClasses();
 $("clsSel").onchange=classQuery;
 $("semMode").onchange=async()=>{
   semMode=$("semMode").value;
   $("nowBar").style.display = semMode==="now" ? "flex" : "none";
   if(semMode!=="vantage" && !SEM){
     try{ SEM=await (await fetch("/api/sem/meta")).json(); }
     catch(e){ $("heat_hint").textContent="sem meta failed: "+e; return; }
   }
   fillClasses();
   classQuery();
 };
 $("nowSnap").oninput=()=>{ if(semMode==="now") classQuery(); };
 $("astm_run").onclick=astmRun;
 $("astm_mv").onclick=astmMoved;
 astmBoot();
 tick();
}
// class dropdown per map source: vantage = workshop memory's stored-frame
// classes; object / now = the respective memory's own semantic codebook.
function fillClasses(){
 const sel=$("clsSel"), keep=sel.value;
 sel.innerHTML='<option value="">— live frames —</option>';
 let names=[];
 if(semMode==="vantage") names=META.classes.map(c=>[c.name,c.count]);
 else{
  const m=SEM&&SEM[semMode];   // SEM.object / SEM.now
  if(!m||m.error){$("heat_hint").textContent=`${semMode} memory unavailable: ${m?m.error:"no meta"}`;return;}
  names=m.classes.map(c=>[c,semMode==="object"?m.counts[c]:null]);
  if(semMode==="now"){const ns=$("nowSnap");ns.max=m.snapshots.length-1;
   if(+ns.value>+ns.max)ns.value=ns.max;}
 }
 names.forEach(([n,cnt])=>{const o=document.createElement("option");
   o.value=n;o.textContent=cnt!=null?`${n} (${cnt})`:n;sel.appendChild(o);});
 if([...sel.options].some(o=>o.value===keep)) sel.value=keep;
}
async function classQuery(){
 const sel=$("clsSel");
 if(!sel.value){classMode=null;$("heat_hint").textContent=IDLE_HINT;$("nowLbl").textContent="";tick();return;}
 playing=false;$("play").textContent="▶ play";
 classMode=`${sel.value}|${semMode}`;
 if(semMode==="vantage"){
  const r=await (await fetch(`/api/where_class?cls=${encodeURIComponent(sel.value)}`)).json();
  r.truth=null;
  drawHeat(r);
  $("heat_hint").textContent=`where were "${sel.value}"s seen from? (${r.count} stored frames · ${r.ms.toFixed(0)} ms) · ● peak vantage`;
 }else if(semMode==="object"){
  const r=await (await fetch(`/api/sem/where?cls=${encodeURIComponent(sel.value)}&mode=object`)).json();
  if(r.error){$("heat_hint").textContent="error: "+r.error;return;}
  drawSemHeat(r);
  $("heat_hint").textContent=`"${sel.value}" object positions (depth-bound memory) · ● peak (${r.peak[0].toFixed(2)}, ${r.peak[1].toFixed(2)}) · ${r.count} detections · ${r.ms.toFixed(0)} ms`;
 }else{
  const k=+$("nowSnap").value;
  const r=await (await fetch(`/api/sem/now?cls=${encodeURIComponent(sel.value)}&snap=${k}`)).json();
  if(r.error){$("heat_hint").textContent="error: "+r.error;return;}
  drawSemHeat(r);
  $("nowLbl").textContent=`snapshot ${r.snap+1}/${r.snapshots.length} · frame row ${r.row}`;
  $("heat_hint").innerHTML=`M_now @ row ${r.row} · "${sel.value}" decayed weight w=${r.w.toExponential(2)} · ${r.ms.toFixed(0)} ms · `+
   (r.forgotten?`<b style="color:#e34948">forgotten (trace decayed)</b>`:`● peak (${r.peak[0].toFixed(2)}, ${r.peak[1].toFixed(2)})`);
 }
}
async function loop(){ if(!playing) return; cur=(cur+1)%rowList.length; $("slider").value=cur; await tick(); timer=setTimeout(loop,40); }

// ---- tick cache + prefetch: fetch frame+reverse in ONE request, keep a
// small LRU of upcoming ticks so playback/stepping never waits on the wire.
const tickCache=new Map(); const TICK_CACHE_MAX=24;
async function getTick(row){
 if(tickCache.has(row)) return tickCache.get(row);
 const p=(await fetch(`/api/tick?row=${row}`)).json();
 tickCache.set(row,p);
 if(tickCache.size>TICK_CACHE_MAX) tickCache.delete(tickCache.keys().next().value);
 return p;
}
function prefetch(){
 for(let k=1;k<=4;k++){
  const row=rowList[(cur+k)%rowList.length];
  if(!tickCache.has(row)) getTick(row); // fire and forget
 }
}
let tickSeq=0;
async function tick(){
 const row=rowList[cur], seq=++tickSeq;
 const d=await getTick(row);
 if(seq!==tickSeq) return; // a newer scrub superseded this tick
 const f=d.frame, r=d.reverse;
 drawRGB(f); if(!classMode) drawHeat(r); drawPolar(r);
 $("status").textContent=`held-out ${cur+1}/${rowList.length} (row ${row}) · query ${r.ms.toFixed(0)} ms · time recall: frame ${r.time.recalled_frame_idx} (${r.time.dt_s.toFixed(1)}s away)`;
 $("timeline_hint").textContent=`est yaw ${(r.est_yaw*57.3).toFixed(0)}° vs truth ${(r.truth.yaw*57.3).toFixed(0)}°`;
 prefetch();
}
function drawRGB(f){
 const img=new Image();
 img.onload=()=>{cvR.height=cvR.width*img.height/img.width;
  cxR.drawImage(img,0,0,cvR.width,cvR.height);
  const sx=cvR.width/640, sy=cvR.height/480;
  f.dets.forEach(d=>{const[a,b,c,e]=d.box;
   cxR.strokeStyle=`hsl(${(d.cid*47)%360} 70% 60%)`;cxR.lineWidth=2;
   cxR.strokeRect(a*sx,b*sy,(c-a)*sx,(e-b)*sy);
   cxR.fillStyle=cxR.strokeStyle;cxR.font="11px system-ui";
   cxR.fillText(`${d.cls} ${d.conf.toFixed(2)}`,a*sx+2,Math.max(10,b*sy-3));});};
 img.src="data:image/jpeg;base64,"+f.img;
}
let lastHeat=null;
function drawHeat(r){
 lastHeat=r;
 const [x0,x1,y0,y1]=META.extent;
 const W=cvH.width,H=cvH.height;
 cxH.clearRect(0,0,W,H);
 blitHeat(cxH,W,H,r.heat,r.heat_min,r.heat_max||1);
 const X=x=> (x-x0)/(x1-x0)*W, Y=y=> H-(y-y0)/(y1-y0)*H;
 cxH.strokeStyle="rgba(255,255,255,.55)";cxH.lineWidth=1.2;cxH.beginPath();
 META.traj.forEach((p,i)=>{i?cxH.lineTo(X(p[0]),Y(p[1])):cxH.moveTo(X(p[0]),Y(p[1]))});cxH.stroke();
 // truth X (absent in class mode)
 if(r.truth){
  cxH.strokeStyle="#fff";cxH.lineWidth=2.5;
  const tx=X(r.truth.x),ty=Y(r.truth.y);
  cxH.beginPath();cxH.moveTo(tx-7,ty-7);cxH.lineTo(tx+7,ty+7);cxH.moveTo(tx+7,ty-7);cxH.lineTo(tx-7,ty+7);cxH.stroke();
 }
 // est peak
 cxH.fillStyle="#e34948";cxH.beginPath();cxH.arc(X(r.peak[0]),Y(r.peak[1]),6,0,7);cxH.fill();
 cxH.strokeStyle="#0b0e12";cxH.stroke();
}
// object / now fields carry their own axes (the classroom memories' grid,
// truth bounds + 1 m pad) — same blit path, mapped through r.gx / r.gy.
function drawSemHeat(r){
 const W=cvH.width,H=cvH.height;
 cxH.clearRect(0,0,W,H);
 blitHeat(cxH,W,H,r.field,r.field_min,r.field_max||1);
 const [x0,x1]=r.gx,[y0,y1]=r.gy;
 const X=x=>(x-x0)/(x1-x0)*W, Y=y=>H-(y-y0)/(y1-y0)*H;
 cxH.strokeStyle="rgba(255,255,255,.55)";cxH.lineWidth=1.2;cxH.beginPath();
 META.traj.forEach((p,i)=>{i?cxH.lineTo(X(p[0]),Y(p[1])):cxH.moveTo(X(p[0]),Y(p[1]))});cxH.stroke();
 cxH.fillStyle="#e34948";cxH.beginPath();cxH.arc(X(r.peak[0]),Y(r.peak[1]),6,0,7);cxH.fill();
 cxH.strokeStyle="#0b0e12";cxH.stroke();
}
function drawPolar(r){
 const W=cvP.width,H=cvP.height,cx=W/2,cy=H/2,R=Math.min(W,H)/2-24;
 cxP.clearRect(0,0,W,H);
 cxP.strokeStyle="#262e3a";
 [0.33,0.66,1].forEach(f=>{cxP.beginPath();cxP.arc(cx,cy,R*f,0,7);cxP.stroke();});
 const vals=r.polar, mn=Math.min(...vals), mx=Math.max(...vals);
 cxP.strokeStyle="#2a78d6";cxP.lineWidth=2;cxP.beginPath();
 META.angles.forEach((a,i)=>{
  const rr=R*(vals[i]-mn)/(mx-mn+1e-9);
  const px=cx+rr*Math.cos(a), py=cy-rr*Math.sin(a);
  i?cxP.lineTo(px,py):cxP.moveTo(px,py);});
 cxP.closePath();cxP.stroke();
 // truth heading
 cxP.strokeStyle="#fff";cxP.setLineDash([5,4]);cxP.lineWidth=2;cxP.beginPath();
 cxP.moveTo(cx,cy);cxP.lineTo(cx+R*Math.cos(r.truth.yaw),cy-R*Math.sin(r.truth.yaw));cxP.stroke();cxP.setLineDash([]);
 // est
 const rr=R;
 cxP.fillStyle="#e34948";cxP.beginPath();
 cxP.arc(cx+(R*0.97)*Math.cos(r.est_yaw),cy-(R*0.97)*Math.sin(r.est_yaw),6,0,7);cxP.fill();
}
async function onMapClick(e){
 const rect=cvH.getBoundingClientRect();
 const [x0,x1,y0,y1]=META.extent;
 const wx=x0+(e.clientX-rect.left)/rect.width*(x1-x0);
 const wy=y0+(1-(e.clientY-rect.top)/rect.height)*(y1-y0);
 $("qout").innerHTML=`<span class="hint">querying (${wx.toFixed(2)}, ${wy.toFixed(2)}) …</span>`;
 const res=await (await fetch(`/api/query?x=${wx}&y=${wy}`)).json();
 $("qout").innerHTML=res.top.map(t=>
  `<div class="thumb"><img src="data:image/jpeg;base64,${t.img}">
    sim ${t.sim.toFixed(3)} · (${t.x.toFixed(1)},${t.y.toFixed(1)})<br>${t.dets.join(", ")||"—"}</div>`).join("");
}

// ---- ASTM panel -----------------------------------------------------------
let ASTM=null;
async function astmBoot(){
 try{
  const m=await (await fetch("/api/astm/meta")).json();
  if(m.error){$("astm_out").textContent="ASTM unavailable: "+m.error;return;}
  ASTM=m;
  const sel=$("astm_cls");
  m.classes.forEach(c=>{const o=document.createElement("option");o.value=c;o.textContent=c;sel.appendChild(o);});
  if(m.classes.includes("chair")) sel.value="chair";
  $("astm_when").value="150";
  $("astm_out").textContent=`${m.n_events} events · ${m.classes.length} classes · hd ${m.hd} · pick knowns, choose the unknown, run`;
 }catch(e){$("astm_out").textContent="ASTM meta failed: "+e;}
}
async function astmRun(){
 const dec=document.querySelector('input[name=astm_dec]:checked').value;
 const p=new URLSearchParams({decode:dec});
 const c=$("astm_cls").value; if(c&&dec!=="what") p.set("what",c);
 const w=$("astm_when").value.trim(); if(w&&dec!=="when") p.set("when",w);
 const x=$("astm_x").value, y=$("astm_y").value;
 if(x!==""&&y!==""&&dec!=="where"){p.set("x",x);p.set("y",y);}
 $("astm_out").textContent="querying…"; $("astm_lat").textContent="";
 const r=await (await fetch("/api/astm/query?"+p)).json();
 if(r.error){$("astm_out").textContent="error: "+r.error;$("cv_astm").style.display="none";return;}
 if(dec==="where") astmDrawHeat(r);
 else if(dec==="when") astmDrawCurve(r);
 else astmDrawWhat(r);
 $("astm_lat").textContent=`trace ${r.trace} · encode ${r.ms.encode} ms · unbind ${r.ms.unbind} ms · decode ${r.ms.decode} ms · total ${r.ms.total} ms`;
}
function astmDrawHeat(r){
 const cv=$("cv_astm"),cx=cv.getContext("2d");
 const nx=r.gx[2],ny=r.gy[2];
 cv.style.display="block"; cv.width=420; cv.height=Math.round(420*ny/nx);
 const W=cv.width,H=cv.height;
 blitHeat(cx,W,H,r.field,r.field_min,r.field_max);
 const X=x=>(x-r.gx[0])/(r.gx[1]-r.gx[0])*W, Y=y=>H-(y-r.gy[0])/(r.gy[1]-r.gy[0])*H;
 cx.fillStyle="#e34948";cx.beginPath();cx.arc(X(r.answer[0]),Y(r.answer[1]),6,0,7);cx.fill();
 cx.strokeStyle="#0b0e12";cx.stroke();
 $("astm_out").textContent=`● peak at (${r.answer[0].toFixed(2)}, ${r.answer[1].toFixed(2)}) · sim ${r.sim.toFixed(4)}`;
}
function astmDrawCurve(r){
 const cv=$("cv_astm"),cx=cv.getContext("2d");
 cv.style.display="block"; cv.width=420; cv.height=170;
 const W=cv.width,H=cv.height,v=r.curve,n=v.length;
 const mn=Math.min(...v),mx=Math.max(...v);
 cx.clearRect(0,0,W,H);
 cx.strokeStyle="#2a78d6";cx.lineWidth=1.6;cx.beginPath();
 v.forEach((s,i)=>{const px=i/(n-1)*W, py=H-10-(s-mn)/(mx-mn+1e-9)*(H-22);
  i?cx.lineTo(px,py):cx.moveTo(px,py);});
 cx.stroke();
 cx.fillStyle="#e34948";cx.beginPath();
 cx.arc(r.t_argmax/r.t_axis_max*W, 12, 5, 0, 7);cx.fill();
 cx.fillStyle="#93a1b3";cx.font="10px system-ui";
 cx.fillText("0",2,H-1);cx.fillText(String(r.t_axis_max),W-26,H-1);
 $("astm_out").textContent=`peak time t = ${r.answer.toFixed(0)} (event row) · sim ${r.sim.toFixed(4)}`;
}
function astmDrawWhat(r){
 $("cv_astm").style.display="none";
 const mx=Math.max(r.ranked[0].sim,1e-9);
 $("astm_out").innerHTML=r.ranked.map(t=>
  `<div class="arow"><span class="lbl">${t.cls}</span>
    <div class="abar" style="width:${Math.max(2,180*t.sim/mx)}px"></div>
    <span>${t.sim.toFixed(4)}</span></div>`).join("");
}
async function astmMoved(){
 if(!ASTM){$("astm_out").textContent="ASTM not loaded yet";return;}
 const t=Math.round(ASTM.t_max), wa=`0:${Math.round(t/3)}`, wb=`${Math.round(2*t/3)}:${t}`;
 $("astm_out").textContent=`vocabulary scan: which class moved between rows ${wa} and ${wb}? …`;
 $("astm_lat").textContent=""; $("cv_astm").style.display="none";
 const r=await (await fetch(`/api/astm/moved?wa=${wa}&wb=${wb}`)).json();
 if(r.error){$("astm_out").textContent="error: "+r.error;return;}
 $("astm_out").innerHTML=`<div class="hint">moved most, rows ${wa} vs ${wb} (${r.ms.toFixed(0)} ms):</div>`+
  r.top.map(t=>`<div class="arow"><span class="lbl">${t.class}</span>
   <span>(${t.a[0].toFixed(1)},${t.a[1].toFixed(1)}) → (${t.b[0].toFixed(1)},${t.b[1].toFixed(1)})</span>
   <span><b>${t.moved_m.toFixed(2)} m</b></span>
   <span>sims ${t.sim_a.toFixed(3)}/${t.sim_b.toFixed(3)}</span></div>`).join("");
}
boot();
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--memory", default=os.path.join(
        WORKSHOP, "outputs", "classroom_detections", "associative_memory_stride.pt"))
    ap.add_argument("--embeddings", default=os.path.join(
        WORKSHOP, "outputs", "classroom_detections", "embeddings.pt"))
    ap.add_argument("--port", type=int, default=8020)
    ap.add_argument("--grid-resolution", type=int, default=56)
    ap.add_argument("--repo", default=None,
                    help="HF dataset repo (default: the Telluride classroom dataset)")
    ap.add_argument("--rgb-config", default="rgb_d455",
                    help='e.g. "school_run1_rgb_d455" for the school run')
    ap.add_argument("--odom-config", default="odometry_lio_sam",
                    help='e.g. "school_run1_odometry_lio_sam"')
    ap.add_argument("--detections", default=None,
                    help="detections.csv path (default: workshop classroom outputs)")
    args = ap.parse_args()
    state = DemoState(args.memory, args.embeddings, grid_resolution=args.grid_resolution,
                      repo=args.repo, rgb_config=args.rgb_config,
                      odom_config=args.odom_config, detections_path=args.detections)
    serve(state, args.port)


if __name__ == "__main__":
    main()
