"""ASTM P0: multi-trace algebraic spatio-temporal memory + query router.

Implements the architecture frozen in
``wiki/analysis/2026-07-29-vsa-query-layer-paper-plan.md``:

  * A canonical **event stream** E = {(f, c, j, x, y, t, conf)} exported once
    from the existing classroom pipeline outputs (detections.csv + poses).
  * A **fixed set of associative traces** — one per query shape, the VSA
    equivalent of secondary indexes (design rule: never store more factors in
    a trace than its target query can supply; measured basis = the 22x
    under-unbinding attenuation):

        M_what_where  = sum conf * C(c)   (x) S(x,y)
        M_what_when   = sum conf * C(c)   (x) T(t)
        M_where_when  = sum conf * S(x,y) (x) T(t)
        M_event       = sum conf * C(c)   (x) S(x,y) (x) T(t)

  * **Temporal range kernels** T_[a,b] = (1/(b-a)) * int_a^b T^t dt, closed
    form per frequency component. Used on the PROBE side (correlation), never
    as a division key: the integral is not unit-modulus, so dividing by it
    would amplify components near its nulls.
  * **Optional multi-scale time** (--time-scales "5,20,80"): several FPE time
    bases at different length scales, combined per event by ``bundle`` (mean
    of the per-scale phasors -> multi-resolution kernel; NOT unit-modulus, so
    time is then only ever used probe-side) or ``bind`` (elementwise product
    -> sharpened kernel; the product of unit-modulus phasors is unit-modulus,
    so it remains a valid division key and the closed-form range kernel works
    on the summed per-scale frequencies).
  * A **query router**: any one unknown of {what, where, when}, given the
    others (or nothing but one factor), routes to the marginal or event
    trace; discrete unknowns are decoded by vocabulary scan; per-stage
    latency (encode / unbind / decode) is recorded for every query. Decoding
    is a single matvec against decoder matrices precomputed once at build
    (probe-side factors fold into the 1-D residual first, never into the
    decoder matrix).
  * **Calibrated abstention** (v2, evaluation-hardening 2026-07-30): at
    build time ~200 null probes per (decode, trace, probe-kind) cell —
    mismatched/fake keys routed through the same trace and probe kind as a
    real query, plus random residuals at that trace's RMS amplitude — give
    per-cell null peak-similarity distributions; every query reports
    z = (sim - null_mean)/null_std against the MATCHING cell (info["null"]
    names it) and confident = (z >= threshold, default 3). The v1
    one-null-per-decode-type scheme (misapplied across marginal/range
    queries) is kept only as a fallback for old .pt files.
  * **Baseline A** — the exact temporal event table — runs beside every VSA
    query so answers are scored automatically (accuracy vs the table is the
    paper's correctness metric; the table will win at this scale — the paper
    claim is the accuracy-latency-energy-bytes frontier at scale, not a win
    here).

Usage (from repo root, after classroom_pipeline embed has been run):

    python vsa_cognitive_mapping/astm_traces.py export
    python vsa_cognitive_mapping/astm_traces.py export --place-mode object   # -> events_object.csv
    python vsa_cognitive_mapping/astm_traces.py export --min-class-events 8 --class-blocklist "airplane,cat"
    python vsa_cognitive_mapping/astm_traces.py build --hd-dim 8192
    python vsa_cognitive_mapping/astm_traces.py build --class-weighting balanced
    python vsa_cognitive_mapping/astm_traces.py build --events-name events_object.csv --traces-name astm_traces_object.pt
    python vsa_cognitive_mapping/astm_traces.py build --time-scales 5,20,80 --time-combine bundle
    python vsa_cognitive_mapping/astm_traces.py bench
    python vsa_cognitive_mapping/astm_traces.py bench --gt-bandwidths "0.4,0.75,1.5"  # GT-bandwidth sensitivity + supported criterion
    python vsa_cognitive_mapping/astm_traces.py bench --compare      # 3-way single/bundle/bind
    python vsa_cognitive_mapping/astm_traces.py bench --bias-bench   # conf/balanced/log class-frequency bias
    python vsa_cognitive_mapping/astm_traces.py query --what chair --when 150
    python vsa_cognitive_mapping/astm_traces.py query --what chair --when 100:300
    python vsa_cognitive_mapping/astm_traces.py query --where -1.1 -0.2 --when 1000:1200

Terminology note (fact-check Round 4): this is a *fixed-dimensional
associative map state*; the reported memory accounting includes the traces
AND the codebooks/bases/decoder grids.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vsa_cognitive_mapping.vsa import Phasor  # noqa: E402
from vsa_cognitive_mapping.classroom_pipeline import (  # noqa: E402
    ClassroomEncoders, _class_seed, _load_embeddings, _load_detections_by_ts,
    _localize_detections, load_poses_interpolated,
)

DEFAULT_OUT = os.path.join("outputs", "classroom")
EVENTS_CSV = "events.csv"
EVENTS_OBJECT_CSV = "events_object.csv"
TRACES_PT = "astm_traces.pt"

# correctness thresholds used for bench scoring / abstention stats
POS_OK_M = 1.0      # position answers within 1 m of the exact modal answer
TIME_OK_FR = 40.0   # time answers within 40 frames (= 2x base time scale)

# ground-truth KDE bandwidths used by ExactEventTable by default. NOTE
# (evaluation-hardening review, tracker entry (r)): these EQUAL the FPE
# length scales (pos 0.75 m / time 20 fr), which makes table and memory
# near-identical estimators that can mode-flip together. `bench
# --gt-bandwidths` re-scores the same answers under several bandwidths and
# under a bandwidth-free "supported" criterion — see cmd_bench.
GT_POS_BW = 0.75
GT_TIME_BW = 20.0


def _class_seed_mixed(name: str, mix=None) -> int:
    """Per-class atom seed, optionally mixed with an integer.

    mix=None -> exactly ``_class_seed(name)`` (the historical default; every
    logged artifact used this). mix=k -> a deterministic md5-based reseed of
    the whole class codebook (NOT Python's salted ``hash()``), so sweep
    replicates can draw genuinely different class atoms per seed."""
    if mix is None:
        return _class_seed(name)
    return int(hashlib.md5(f"{name}|{int(mix)}".encode()).hexdigest(),
               16) % 10_000_000


# ==========================================================================
# Event stream (canonical input shared by every representation)
# ==========================================================================

def cmd_export(args):
    """detections.csv + poses -> canonical event stream csv.

    --place-mode robot (default): place = robot pose at observation time
    (consistent with the measured July numbers) -> events.csv.
    --place-mode object: place = the detection's world position via median
    box depth (read-only reuse of classroom_pipeline._localize_detections,
    same intrinsics/camera-offset path as object_map.py), falling back to
    the robot pose when the box has no valid depth -> events_object.csv
    (or --events-name). The fallback fraction is reported.

    Hygiene filters (--min-class-events, --class-blocklist) drop classes at
    this canonical-event-stream boundary — COCO detector noise ("airplane"
    in a classroom) is removed once, so every downstream representation
    (traces AND exact-table baseline) sees the same cleaned stream.

    Track id j = -1 (no instance tracker yet)."""
    frame_idx, ts, _ = _load_embeddings(args.out_dir)
    x, y, psi = load_poses_interpolated(ts)
    dets_by_ts = _load_detections_by_ts(args.out_dir)

    xy_by_ts = None
    if args.place_mode == "object":
        pose_of_ts = {int(t_ns): (float(x[n]), float(y[n]), float(psi[n]))
                      for n, t_ns in enumerate(ts)}
        xy_by_ts, _n_ok, _n_miss = _localize_detections(args.out_dir, pose_of_ts)

    rows, n_fallback = [], 0
    for n, t_ns in enumerate(ts):
        dets = dets_by_ts.get(int(t_ns), [])
        # per-ts lists from _load_detections_by_ts / _localize_detections are
        # parallel (both read detections.csv in file order)
        obj_xy = xy_by_ts.get(int(t_ns)) if xy_by_ts is not None else None
        for k, (cname, conf) in enumerate(dets):
            ex, ey = float(x[n]), float(y[n])
            if obj_xy is not None and obj_xy[k] is not None:
                ex, ey = obj_xy[k]
            elif xy_by_ts is not None:
                n_fallback += 1  # object mode but no valid depth -> robot pose
            rows.append([int(frame_idx[n]), cname, -1, ex, ey, float(n), float(conf)])
    n_total = len(rows)

    # ---- hygiene filters (canonical-event-stream boundary) ------------------
    block = {c.strip() for c in (args.class_blocklist or "").split(",") if c.strip()}
    counts = Counter(r[1] for r in rows)
    unknown = block - set(counts)
    if unknown:
        print(f"  (blocklist classes never seen: {sorted(unknown)})")
    dropped = {}
    for c in sorted(counts):
        if c in block:
            dropped[c] = (counts[c], "blocklist")
        elif args.min_class_events and counts[c] < args.min_class_events:
            dropped[c] = (counts[c], f"count {counts[c]} < {args.min_class_events}")
    if dropped:
        rows = [r for r in rows if r[1] not in dropped]
        for c, (k, why) in sorted(dropped.items(), key=lambda kv: -kv[1][0]):
            print(f"  dropped class {c!r}: {k} events ({why})")
        print(f"hygiene filters dropped {len(dropped)} classes / "
              f"{sum(k for k, _ in dropped.values())} of {n_total} events")

    name = args.events_name or (EVENTS_OBJECT_CSV if args.place_mode == "object"
                                else EVENTS_CSV)
    path = os.path.join(args.out_dir, name)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "class", "track", "x", "y", "t", "conf"])
        w.writerows(rows)
    classes = sorted({r[1] for r in rows})
    print(f"exported {len(rows)} events, {len(classes)} classes "
          f"(place-mode={args.place_mode}) -> {path}")
    if xy_by_ts is not None:
        print(f"  object localization fallback to robot pose: "
              f"{n_fallback}/{n_total} events ({100.0 * n_fallback / max(n_total, 1):.1f}%)")


def load_events(out_dir, events_name=EVENTS_CSV):
    path = os.path.join(out_dir, events_name)
    ev = {"class": [], "x": [], "y": [], "t": [], "conf": []}
    with open(path) as f:
        for r in csv.DictReader(f):
            ev["class"].append(r["class"])
            ev["x"].append(float(r["x"]))
            ev["y"].append(float(r["y"]))
            ev["t"].append(float(r["t"]))
            ev["conf"].append(float(r["conf"]))
    for k in ("x", "y", "t", "conf"):
        ev[k] = np.array(ev[k])
    return ev


# ==========================================================================
# Baseline A: exact temporal event table
# ==========================================================================

class ExactEventTable:
    """The trivially-correct baseline (and automatic ground truth)."""

    def __init__(self, ev):
        self.ev = ev
        self.classes = sorted(set(ev["class"]))
        self.cls_arr = np.array(ev["class"])

    def _mask(self, what=None, when=None):
        m = np.ones(len(self.cls_arr), bool)
        if what is not None:
            m &= (self.cls_arr == what)
        if when is not None:
            a, b = when
            m &= (self.ev["t"] >= a) & (self.ev["t"] <= b)
        return m

    def where(self, what=None, when=None, bw=GT_POS_BW):
        """Conf-weighted MODAL position of matching events (None if none).

        Mode, not mean: the VSA decode answers 'the densest place', and for
        multimodal classes the mean lands in empty space. Mode = event whose
        Gaussian-weighted neighbourhood (bandwidth bw, matching the FPE
        pos length-scale) carries the most confidence mass."""
        m = self._mask(what, when)
        if not m.any():
            return None
        xs, ys, w = self.ev["x"][m], self.ev["y"][m], self.ev["conf"][m]
        d2 = (xs[:, None] - xs[None, :]) ** 2 + (ys[:, None] - ys[None, :]) ** 2
        dens = (np.exp(-d2 / (2 * bw * bw)) * w[None, :]).sum(axis=1)
        k = int(np.argmax(dens))
        return (float(xs[k]), float(ys[k]))

    def when(self, what=None, near_xy=None, radius=1.0, bw=GT_TIME_BW):
        """Conf-weighted MODAL time (bandwidth bw = FPE time length-scale)."""
        m = self._mask(what)
        if near_xy is not None:
            d = np.hypot(self.ev["x"] - near_xy[0], self.ev["y"] - near_xy[1])
            m &= (d <= radius)
        if not m.any():
            return None
        t, w = self.ev["t"][m], self.ev["conf"][m]
        d2 = (t[:, None] - t[None, :]) ** 2
        dens = (np.exp(-d2 / (2 * bw * bw)) * w[None, :]).sum(axis=1)
        return float(t[int(np.argmax(dens))])

    def what(self, near_xy=None, when=None, radius=1.0):
        m = self._mask(None, when)
        if near_xy is not None:
            d = np.hypot(self.ev["x"] - near_xy[0], self.ev["y"] - near_xy[1])
            m &= (d <= radius)
        if not m.any():
            return None
        scores = defaultdict(float)
        for c, cf in zip(self.cls_arr[m], self.ev["conf"][m]):
            scores[c] += cf
        return max(scores, key=scores.get)

    # ---- bandwidth-free "supported" criterion (FIX 1b, tracker entry (r)) --
    def supported(self, kind, ans, gt_kwargs):
        """Is the VSA answer supported by ANY raw event, no KDE involved?

        pos:   answer within POS_OK_M of any event of the queried class in
               the queried window;
        time:  answer within TIME_OK_FR of any matching event (class and,
               if given, near_xy/radius constraint);
        class: at least one event of the ANSWERED class satisfies the
               query's place/time constraints.

        gt_kwargs = the same kwargs the exact-table scoring call used, so the
        window/radius conventions match the modal scoring exactly."""
        if ans is None:
            return False
        kw = gt_kwargs
        if kind == "pos":
            m = self._mask(kw.get("what"), kw.get("when"))
            if not m.any():
                return False
            d = np.hypot(self.ev["x"][m] - ans[0], self.ev["y"][m] - ans[1])
            return bool((d <= POS_OK_M).any())
        if kind == "time":
            m = self._mask(kw.get("what"))
            if kw.get("near_xy") is not None:
                d = np.hypot(self.ev["x"] - kw["near_xy"][0],
                             self.ev["y"] - kw["near_xy"][1])
                m &= (d <= kw.get("radius", 1.0))
            if not m.any():
                return False
            return bool((np.abs(self.ev["t"][m] - float(ans)) <= TIME_OK_FR).any())
        # class: any event of the answered class within the constraints
        m = self._mask(ans, kw.get("when"))
        if kw.get("near_xy") is not None:
            d = np.hypot(self.ev["x"] - kw["near_xy"][0],
                         self.ev["y"] - kw["near_xy"][1])
            m &= (d <= kw.get("radius", 1.0))
        return bool(m.any())


# ==========================================================================
# Trace set
# ==========================================================================

def _range_kernel(w, a, b):
    """(1/(b-a)) * int_a^b e^{i w t} dt per frequency component, closed form."""
    out = np.empty(w.shape, np.complex128)
    nz = np.abs(w) > 1e-12
    out[nz] = (np.exp(1j * w[nz] * b) - np.exp(1j * w[nz] * a)) / (1j * w[nz] * (b - a))
    out[~nz] = 1.0
    return out


class TraceSet:
    """Fixed-dimensional associative map state: 4 traces + codebooks + bases.

    Time encoding is single- or multi-scale FPE, controlled by
    ``time_scales`` (list of length scales) and ``time_combine``:

      * single scale (default, len(time_scales)==1): T(t) = Bt**(t/ell),
        exactly the original behaviour (uses the ClassroomEncoders base).
      * ``bind``: T(t) = prod_s Bs**(t/ell_s) = exp(i * sum_s(w_s/ell_s) * t).
        Unit-modulus, so it stays a valid division key, and the closed-form
        range kernel applies directly to the summed frequencies. The
        similarity kernel is the PRODUCT of the per-scale kernels (sharpened;
        width set by the smallest scale).
      * ``bundle``: T(t) = mean_s Bs**(t/ell_s). Multi-resolution SUM of the
        per-scale kernels, but NOT unit-modulus — it must never be used as a
        division key (dividing would amplify components near its nulls), so
        the router folds ALL time factors probe-side (correlation). Range
        kernels are computed per scale in closed form and averaged.

    Extra per-scale bases are drawn locally (seed + 1000 + scale_index),
    deterministically from the TraceSet seed; classroom_pipeline is untouched.
    """

    def __init__(self, hd_dim, seed, pos_length_scale, time_length_scale,
                 classes, bounds, t_max, grid=72,
                 time_scales=None, time_combine="bundle",
                 class_seed_mix=None):
        self.hd = hd_dim
        self.seed = seed
        self.enc = ClassroomEncoders(hd_dim, seed + 100,
                                     pos_length_scale, time_length_scale)
        self.classes = list(classes)
        # class_seed_mix=None -> historical _class_seed(name) atoms (all
        # logged artifacts); an int mixes it into the md5 so replicates
        # (e.g. sweep seeds) draw independent class codebooks (FIX 4).
        self.class_seed_mix = class_seed_mix
        self.C = {c: Phasor(dim=hd_dim,
                            seed=_class_seed_mixed(c, class_seed_mix)).values
                  for c in self.classes}
        self.M = {k: np.zeros(hd_dim, np.complex128)
                  for k in ("what_where", "what_when", "where_when", "event")}
        self.wsum = 0.0
        self.n_events = 0
        self.bounds = bounds
        self.t_max = float(t_max)
        self.null_calib = None  # set by calibrate()
        self.class_weighting = "conf"   # set by _build_traceset (provenance)
        self.events_name = EVENTS_CSV   # which event stream built this state

        # ---- time bases (single- or multi-scale) --------------------------
        scales = [float(s) for s in (time_scales if time_scales
                                     else [time_length_scale])]
        if time_combine not in ("bundle", "bind"):
            raise ValueError("time_combine must be 'bundle' or 'bind'")
        self.time_scales = scales
        self.time_combine = str(time_combine)
        if len(scales) == 1:
            tbases = [self.enc.Bt]  # exact original single-scale behaviour
        else:
            tbases = [Phasor(dim=hd_dim, seed=seed + 1000 + k)
                      for k in range(len(scales))]
        # per-scale frequencies w_s = angle(base_s)/ell_s  (n_scales, hd)
        self.omega_scales = np.stack([np.angle(B.values) / ell
                                      for B, ell in zip(tbases, scales)])
        # effective frequencies for single/bind (sum over scales)
        self.omega = self.omega_scales.sum(axis=0)

        # decoder grids (probe matrices; part of the memory accounting).
        # Precomputed ONCE; queries never rebuild or rescale them — probe-side
        # factors are folded into the 1-D residual before a single matvec.
        pad = 1.0
        self.gx = np.linspace(bounds[0] - pad, bounds[1] + pad, grid)
        ratio = (bounds[3] - bounds[2] + 2 * pad) / (bounds[1] - bounds[0] + 2 * pad)
        self.gy = np.linspace(bounds[2] - pad, bounds[3] + pad,
                              max(8, int(grid * ratio)))
        G = np.empty((len(self.gy), len(self.gx), hd_dim), np.complex64)
        for j, yy in enumerate(self.gy):
            for i, xx in enumerate(self.gx):
                G[j, i] = self.enc.ctx_pos(xx, yy).values
        self.G_flat = G.reshape(-1, hd_dim)
        n_t = int(t_max) + 1
        self.T_axis = np.empty((n_t, hd_dim), np.complex64)
        for t in range(n_t):
            self.T_axis[t] = self.ctx_time_vec(float(t))
        self.C_mat = np.stack([self.C[c] for c in self.classes]).astype(np.complex64)

    # ---- time encoding -----------------------------------------------------
    def ctx_time_vec(self, t):
        """Multi-scale time vector for scalar t (complex128, len hd).

        single/bind: exp(i * omega * t) — unit-modulus (valid division key).
        bundle: mean_s exp(i * omega_s * t) — NOT unit-modulus (probe only).
        """
        if self.time_combine == "bind" or len(self.time_scales) == 1:
            return np.exp(1j * self.omega * float(t))
        return np.exp(1j * self.omega_scales * float(t)).mean(axis=0)

    # ---- write path ------------------------------------------------------
    def add_event(self, cname, x, y, t, conf):
        S = self.enc.ctx_pos(x, y).values
        T = self.ctx_time_vec(t)
        C = self.C[cname]
        self.M["what_where"] += conf * C * S
        self.M["what_when"] += conf * C * T
        self.M["where_when"] += conf * S * T
        self.M["event"] += conf * C * S * T
        self.wsum += conf
        self.n_events += 1

    def finalize(self):
        for k in self.M:
            self.M[k] = self.M[k] / max(self.wsum, 1e-9)

    # ---- range kernel (probe side only) ----------------------------------
    def time_range(self, a, b):
        """(1/(b-a)) * int_a^b T(t) dt, closed form. a==b -> point.

        bind/single: one closed form on the effective (summed) frequencies.
        bundle: closed form per scale, then averaged (linearity of the
        integral over the bundle mean). Never unit-modulus -> probe side only.
        """
        if b <= a:
            return self.ctx_time_vec(a)
        if self.time_combine == "bind" or len(self.time_scales) == 1:
            return _range_kernel(self.omega, float(a), float(b))
        return np.stack([_range_kernel(w, float(a), float(b))
                         for w in self.omega_scales]).mean(axis=0)

    # ---- null calibration (abstention) -------------------------------------
    # v2 calibration cells (FIX 2, tracker entry (r)): one null distribution
    # per (decode, trace routed to, probe kind). The v1 scheme stored ONE null
    # per decode type, always measured on M_event with point probes — that
    # null was then misapplied to marginal-trace queries (marginals have
    # higher coherent mass -> z inflated) and to range-kernel queries (the
    # range kernel is not unit-modulus and shrinks similarities ~4x -> z
    # deflated). Every (trace, probe-kind) combination the QueryRouter can
    # actually produce gets its own cell:
    CALIB_CELLS = (
        ("where", "event", "point"), ("where", "event", "range"),
        ("where", "what_where", "none"),
        ("where", "where_when", "point"), ("where", "where_when", "range"),
        ("when", "event", "none"), ("when", "what_when", "none"),
        ("when", "where_when", "none"),
        ("what", "event", "point"), ("what", "event", "range"),
        ("what", "what_where", "none"),
        ("what", "what_when", "point"), ("what", "what_when", "range"),
    )

    def calibrate(self, ev, n_null=200, seed=12345, range_frac=1.0 / 3.0):
        """Per-(trace, probe-kind) null peak-similarity distributions (v2).

        For each cell in CALIB_CELLS, half the probes are MISMATCHED keys
        routed through the SAME trace + probe kind as a real query:

          * event-trace cells re-pair real vocabulary items with a guard so
            the exact table has no answer (class x time with no event of the
            class within 2*median(time_scales) of the probe time / window;
            class x place with none within 2*pos_l; place x time with none
            near in both) — the v1 machinery, now per cell;
          * marginal-trace cells where every real key HAS an answer (every
            class has events; the robot is always somewhere) use FAKE keys:
            a random unit phasor in place of the class atom / place / time
            vector (the fake-class-null approach from
            moved_object_synthetic), or a range kernel built on random
            frequencies for range cells;
          * range cells use a REPRESENTATIVE window width W = range_frac *
            t_max (default 1/3, matching the bench windows). NOTE the
            residual width-dependence: the range kernel keeps fewer
            components as the window widens, so nulls calibrated at W are
            approximate for very different widths — re-calibrate range_frac
            if the deployed query mix changes.

        The other half are random-phase residuals scaled to THAT trace's RMS
        component amplitude (v1 used the event trace's RMS for all cells),
        multiplied by a representative probe kernel for point/range cells so
        range-kernel shrinkage is included in the null.

        Storage: dict keyed "decode|trace|kind" (+ "_meta"). QueryRouter
        falls back to legacy per-decode keys for old .pt files."""
        rng = np.random.RandomState(seed)
        router = QueryRouter(self)
        cls_arr = np.array(ev["class"])
        t_arr, x_arr, y_arr = ev["t"], ev["x"], ev["y"]
        n_ev = len(t_arr)
        t_guard = 2.0 * float(np.median(self.time_scales))
        r_guard = 2.0 * float(self.enc.pos_l)
        t_max = float(self.t_max)
        W = float(range_frac) * t_max
        bx = (float(x_arr.min()), float(x_arr.max()))
        by = (float(y_arr.min()), float(y_arr.max()))
        rms = {k: float(np.sqrt(np.mean(np.abs(v) ** 2)))
               for k, v in self.M.items()}
        med_ell = float(np.median(self.time_scales))

        def rand_phasor():
            return np.exp(1j * rng.uniform(0, 2 * np.pi, self.hd))

        def rand_start():
            return float(rng.uniform(0.0, max(t_max - W, 1e-6)))

        def fake_range_kernel():
            # wrong-key range kernel: fresh random frequencies at the median
            # time length-scale, same window width as real range probes
            w = rng.uniform(-np.pi, np.pi, self.hd) / med_ell
            a = rand_start()
            return _range_kernel(w, a, a + W)

        def decode_sim(decode, residual, probe_time=None):
            if decode == "where":
                return router._decode_grid(residual, probe_time=probe_time)[1]
            if decode == "when":
                return router._decode_time(residual)[1]
            return router._decode_class(residual, probe_time=probe_time)[1]

        def mismatch_probe(decode, trace, kind):
            """One mismatched/fake-key (residual, probe_time), or None to
            retry (guard rejected the draw)."""
            if trace == "event":
                i, j = rng.randint(n_ev), rng.randint(n_ev)
                c = cls_arr[i]
                mc = cls_arr == c
                if decode == "where" and kind == "point":
                    t = t_arr[j]
                    if np.any(np.abs(t_arr[mc] - t) <= t_guard):
                        return None
                    return (self.M["event"] / self.C[c],
                            self.ctx_time_vec(t))
                if decode == "where" and kind == "range":
                    a = rand_start()
                    if np.any((t_arr[mc] >= a - t_guard)
                              & (t_arr[mc] <= a + W + t_guard)):
                        return None
                    return self.M["event"] / self.C[c], self.time_range(a, a + W)
                if decode == "when":
                    xx, yy = x_arr[j], y_arr[j]
                    if np.any(np.hypot(x_arr[mc] - xx, y_arr[mc] - yy) <= r_guard):
                        return None
                    return (self.M["event"] / self.C[c]
                            / self.enc.ctx_pos(xx, yy).values, None)
                if decode == "what" and kind == "point":
                    xx, yy, t = x_arr[i], y_arr[i], t_arr[j]
                    near = ((np.hypot(x_arr - xx, y_arr - yy) <= r_guard)
                            & (np.abs(t_arr - t) <= t_guard))
                    if near.any():
                        return None
                    return (self.M["event"] / self.enc.ctx_pos(xx, yy).values,
                            self.ctx_time_vec(t))
                if decode == "what" and kind == "range":
                    xx, yy = x_arr[i], y_arr[i]
                    a = rand_start()
                    near = ((np.hypot(x_arr - xx, y_arr - yy) <= r_guard)
                            & (t_arr >= a - t_guard) & (t_arr <= a + W + t_guard))
                    if near.any():
                        return None
                    return (self.M["event"] / self.enc.ctx_pos(xx, yy).values,
                            self.time_range(a, a + W))
            if trace == "what_where":
                if decode == "where":  # fake class atom (never stored)
                    return self.M["what_where"] / rand_phasor(), None
                # decode == "what": place with no event within the guard
                p = (float(rng.uniform(*bx)), float(rng.uniform(*by)))
                if np.any(np.hypot(x_arr - p[0], y_arr - p[1]) <= r_guard):
                    return None
                return self.M["what_where"] / self.enc.ctx_pos(*p).values, None
            if trace == "what_when":
                if decode == "when":  # fake class atom
                    return self.M["what_when"] / rand_phasor(), None
                # decode == "what": fake time key (every real t has events)
                if kind == "range":
                    return self.M["what_when"], fake_range_kernel()
                return self.M["what_when"], rand_phasor()
            # trace == "where_when"
            if decode == "when":  # place with no event within the guard
                p = (float(rng.uniform(*bx)), float(rng.uniform(*by)))
                if np.any(np.hypot(x_arr - p[0], y_arr - p[1]) <= r_guard):
                    return None
                return self.M["where_when"] / self.enc.ctx_pos(*p).values, None
            # decode == "where": fake time key (the robot is always somewhere)
            if kind == "range":
                return self.M["where_when"], fake_range_kernel()
            return self.M["where_when"], rand_phasor()

        self.null_calib = {}
        for decode, trace, kind in self.CALIB_CELLS:
            sims, n_mis, tries = [], n_null // 2, 0
            while len(sims) < n_mis and tries < n_mis * 100:
                tries += 1
                probe = mismatch_probe(decode, trace, kind)
                if probe is None:
                    continue
                sims.append(decode_sim(decode, *probe))
            n_mis_got = len(sims)
            for _ in range(n_null - len(sims)):  # random residuals @ trace RMS
                res = rms[trace] * rand_phasor()
                if kind == "point":
                    pt = self.ctx_time_vec(float(rng.uniform(0.0, t_max)))
                elif kind == "range":
                    a = rand_start()
                    pt = self.time_range(a, a + W)
                else:
                    pt = None
                sims.append(decode_sim(decode, res, pt))
            s = np.array(sims)
            self.null_calib[f"{decode}|{trace}|{kind}"] = {
                "mean": float(s.mean()), "std": float(max(s.std(), 1e-12)),
                "p95": float(np.percentile(s, 95)),
                "p99": float(np.percentile(s, 99)), "n": int(len(s)),
                "n_mismatch": int(n_mis_got)}
        self.null_calib["_meta"] = {"version": 2, "n_null": int(n_null),
                                    "range_frac": float(range_frac),
                                    "range_width_fr": float(W)}
        return self.null_calib

    # ---- memory accounting ------------------------------------------------
    def memory_bytes(self):
        traces = sum(v.nbytes for v in self.M.values())
        codebook = sum(v.nbytes for v in self.C.values())
        bases = 4 * self.hd * 16  # Bx, By, Bhead, Bt complex128
        if len(self.time_scales) > 1:  # extra local multi-scale time bases
            bases += len(self.time_scales) * self.hd * 16
        decoders = self.G_flat.nbytes + self.T_axis.nbytes + self.C_mat.nbytes
        return {"traces": traces, "class_codebook": codebook, "bases": bases,
                "decoder_grids": decoders,
                "total": traces + codebook + bases + decoders}

    # ---- persistence -------------------------------------------------------
    def save(self, path):
        torch.save({
            "M": {k: torch.from_numpy(v) for k, v in self.M.items()},
            "meta": {"hd": int(self.hd), "seed": int(self.seed),
                     "pos_l": float(self.enc.pos_l), "time_l": float(self.enc.time_l),
                     "classes": list(self.classes),
                     "bounds": [float(b) for b in self.bounds],
                     "t_max": float(self.t_max), "wsum": float(self.wsum),
                     "n_events": int(self.n_events),
                     "time_scales": [float(s) for s in self.time_scales],
                     "time_combine": self.time_combine,
                     "class_weighting": self.class_weighting,
                     "events_name": self.events_name,
                     "class_seed_mix": (None if self.class_seed_mix is None
                                        else int(self.class_seed_mix)),
                     "null_calib": self.null_calib},
        }, path)

    @classmethod
    def load(cls, path, grid=72):
        d = torch.load(path, weights_only=True)
        m = d["meta"]
        ts = cls(m["hd"], m["seed"], m["pos_l"], m["time_l"], m["classes"],
                 tuple(m["bounds"]), m["t_max"], grid=grid,
                 time_scales=m.get("time_scales"),
                 time_combine=m.get("time_combine", "bundle"),
                 class_seed_mix=m.get("class_seed_mix"))
        ts.M = {k: v.numpy() for k, v in d["M"].items()}
        ts.wsum = m["wsum"]
        ts.n_events = m["n_events"]
        ts.null_calib = m.get("null_calib")
        ts.class_weighting = m.get("class_weighting", "conf")
        ts.events_name = m.get("events_name", EVENTS_CSV)
        return ts


# ==========================================================================
# Query router
# ==========================================================================

class QueryRouter:
    """Route (what?, where?, when?) queries to the right trace.

    Exactly one field may be unknown (None). `when` may be a scalar (point),
    an (a, b) tuple (range -> probe-side kernel), or None-with-others-known
    meaning 'decode when'. Every call returns (answer, info) where info has
    per-stage latencies in ms, the peak similarity, and (if the TraceSet was
    calibrated) `z` = (sim - null_mean)/null_std for the decode type and
    `confident` = (z >= z_threshold).

    ALL time factors are applied probe-side: for unit-modulus time keys
    (single scale / bind-combine) multiplying the conjugated residual by T is
    identical to dividing the residual by T; for bundle-combine (non-unit
    keys) probe-side correlation is the only valid option. Class and place
    keys stay unit-modulus and are still divided out."""

    def __init__(self, tr: TraceSet, z_threshold=3.0):
        self.tr = tr
        self.z_threshold = float(z_threshold)

    # -- decode helpers ------------------------------------------------------
    # Each decode = fold probe-side factors into the 1-D conjugated residual
    # (a few 8192-elt multiplies), then ONE matvec against a decoder matrix
    # that was precomputed at build. Identity used:
    #   Re(sum_d G[k,d] * T[d] * conj(res[d])) = Re( G @ (T * conj(res)) )
    # so the (N, hd) probe matrix G*T is never materialized.
    def _decode_grid(self, residual, probe_time=None):
        v = np.conj(residual)
        if probe_time is not None:
            v = v * probe_time
        s = np.real(self.tr.G_flat @ v.astype(np.complex64)) / self.tr.hd
        k = int(np.argmax(s))
        j, i = divmod(k, len(self.tr.gx))
        return (float(self.tr.gx[i]), float(self.tr.gy[j])), float(s[k])

    def _decode_time(self, residual):
        s = np.real(self.tr.T_axis @ np.conj(residual).astype(np.complex64)) / self.tr.hd
        k = int(np.argmax(s))
        return float(k), float(s[k])

    def _decode_class(self, residual, probe_time=None, probe_pos=None):
        v = np.conj(residual)
        if probe_time is not None:
            v = v * probe_time
        if probe_pos is not None:
            v = v * probe_pos
        s = np.real(self.tr.C_mat @ v.astype(np.complex64)) / self.tr.hd
        k = int(np.argmax(s))
        return self.tr.classes[k], float(s[k])

    # -- the router -----------------------------------------------------------
    def query(self, decode, what=None, where=None, when=None):
        """decode in {'where','when','what'}; supply any subset of the others.

        Routing rule = the multi-trace design rule: pick the trace whose
        factors are exactly {decode target} + {supplied knowns}. Time
        (point or range) is always a probe-side kernel — see class docstring."""
        tr = self.tr
        if decode not in ("where", "when", "what"):
            raise ValueError("decode must be where/when/what")
        if locals()[decode] is not None:
            raise ValueError(f"'{decode}' is the decode target; don't supply it")
        if decode == "when" and isinstance(when, (tuple, list)):
            raise ValueError("cannot decode 'when' and give a when-range")

        t0 = time.perf_counter()
        C = tr.C[what] if what is not None else None
        S = tr.enc.ctx_pos(*where).values if where is not None else None
        T_probe, probe_kind = None, "none"
        if when is not None:
            if isinstance(when, (tuple, list)):
                T_probe = tr.time_range(float(when[0]), float(when[1]))
                probe_kind = "range"
            else:
                T_probe = tr.ctx_time_vec(float(when))
                probe_kind = "point"
        t1 = time.perf_counter()

        if decode == "where":
            if C is not None and T_probe is not None:
                trace, residual = "event", tr.M["event"] / C
            elif C is not None:
                trace, residual = "what_where", tr.M["what_where"] / C
            elif T_probe is not None:
                trace, residual = "where_when", tr.M["where_when"]
            else:
                raise ValueError("decode=where needs what and/or when")
            t2 = time.perf_counter()
            answer, sim = self._decode_grid(residual, probe_time=T_probe)

        elif decode == "when":
            if C is not None and S is not None:
                trace, residual = "event", tr.M["event"] / C / S
            elif C is not None:
                trace, residual = "what_when", tr.M["what_when"] / C
            elif S is not None:
                trace, residual = "where_when", tr.M["where_when"] / S
            else:
                raise ValueError("decode=when needs what and/or where")
            t2 = time.perf_counter()
            answer, sim = self._decode_time(residual)

        else:  # decode == "what"
            if S is not None and T_probe is not None:
                trace, residual = "event", tr.M["event"] / S
            elif S is not None:
                trace, residual = "what_where", tr.M["what_where"] / S
            elif T_probe is not None:
                trace, residual = "what_when", tr.M["what_when"]
            else:
                raise ValueError("decode=what needs where and/or when")
            t2 = time.perf_counter()
            answer, sim = self._decode_class(residual, probe_time=T_probe)

        return self._pack(answer, sim, trace, decode, probe_kind, t0, t1, t2)

    def _pack(self, answer, sim, trace, decode, probe_kind, t0, t1, t2):
        t3 = time.perf_counter()
        z, confident, null_used = None, None, None
        calib = self.tr.null_calib
        if calib:
            key = f"{decode}|{trace}|{probe_kind}"
            if key in calib:            # v2: per-(trace, probe-kind) null
                c, null_used = calib[key], key
            elif decode in calib:       # legacy .pt: one null per decode type
                c, null_used = calib[decode], f"legacy:{decode}"
            else:
                c = None
            if c is not None:
                z = float((sim - c["mean"]) / max(c["std"], 1e-12))
                confident = bool(z >= self.z_threshold)
        info = {"trace": trace, "sim": sim, "z": z, "confident": confident,
                "null": null_used,
                "ms_encode": (t1 - t0) * 1e3,
                "ms_unbind": (t2 - t1) * 1e3,
                "ms_decode": (t3 - t2) * 1e3,
                "ms_total": (t3 - t0) * 1e3}
        return answer, info

    # -- vocabulary scan: which class moved between two windows --------------
    def which_moved(self, win_a, win_b, min_sim=0.0):
        """For each class: decode position in window A and B from M_event;
        rank by displacement. Deterministic vocabulary scan, no resonator."""
        Ta = self.tr.time_range(*win_a)
        Tb = self.tr.time_range(*win_b)
        out = []
        for c in self.tr.classes:
            residual = self.tr.M["event"] / self.tr.C[c]
            pa, sa = self._decode_grid(residual, probe_time=Ta)
            pb, sb = self._decode_grid(residual, probe_time=Tb)
            if min(sa, sb) < min_sim:
                continue
            out.append({"class": c, "a": pa, "b": pb,
                        "moved_m": float(np.hypot(pb[0] - pa[0], pb[1] - pa[1])),
                        "sim_a": sa, "sim_b": sb})
        return sorted(out, key=lambda r: -r["moved_m"])


# ==========================================================================
# CLI commands
# ==========================================================================

def _parse_scales(s):
    if s is None:
        return None
    return [float(v) for v in str(s).split(",") if v.strip()]


def _class_weights(ev, mode):
    """Per-event bundle weights implementing --class-weighting.

    The frequency-bias problem: with w_i = conf_i the coherent within-class
    mass of class c grows ~linearly in its total conf mass W_c while
    cross-class crosstalk grows ~sqrt(W_c) — so recall strength tracks
    sighting count (chair peak ~0.25 vs rare classes near the noise floor).

      conf:      w_i = conf_i                      (original behaviour)
      balanced:  w_i = conf_i / sqrt(W_c)          W_c = sum of conf over
                 class c. Equalizes per-class coherent magnitudes at the
                 crosstalk growth rate (sqrt because coherent mass grows
                 ~linearly, crosstalk ~sqrt).
      log:       w_i = conf_i * log(1 + K_c) / K_c K_c = event count of
                 class c. Class mass becomes ~mean_conf * log(1 + K_c):
                 frequent classes keep a damped (logarithmic) advantage
                 instead of full equalization.

    Global scale is irrelevant (finalize() divides by the total weight)."""
    conf = ev["conf"]
    if mode == "conf":
        return conf.copy()
    if mode not in ("balanced", "log"):
        raise ValueError("class_weighting must be conf/balanced/log")
    cls = np.array(ev["class"])
    w = conf.copy()
    for c in np.unique(cls):
        m = cls == c
        if mode == "balanced":
            w[m] = conf[m] / np.sqrt(max(float(conf[m].sum()), 1e-12))
        else:  # log
            K = int(m.sum())
            w[m] = conf[m] * (np.log1p(K) / K)
    return w


def _build_traceset(ev, hd_dim, seed, pos_l, time_l, grid,
                    time_scales=None, time_combine="bundle", n_null=200,
                    class_weighting="conf", class_seed_mix=None):
    classes = sorted(set(ev["class"]))
    bounds = (ev["x"].min(), ev["x"].max(), ev["y"].min(), ev["y"].max())
    tr = TraceSet(hd_dim, seed, pos_l, time_l, classes, bounds, ev["t"].max(),
                  grid=grid, time_scales=time_scales, time_combine=time_combine,
                  class_seed_mix=class_seed_mix)
    tr.class_weighting = str(class_weighting)
    w = _class_weights(ev, class_weighting)
    for i in range(len(ev["t"])):
        tr.add_event(ev["class"][i], ev["x"][i], ev["y"][i], ev["t"][i], w[i])
    tr.finalize()
    if n_null:
        tr.calibrate(ev, n_null=n_null)
    return tr


def cmd_build(args):
    ev = load_events(args.out_dir, args.events_name)
    tr = _build_traceset(ev, args.hd_dim, args.seed, args.pos_length_scale,
                         args.time_length_scale, args.grid,
                         time_scales=_parse_scales(args.time_scales),
                         time_combine=args.time_combine,
                         n_null=args.null_samples,
                         class_weighting=args.class_weighting,
                         class_seed_mix=args.class_seed_mix)
    tr.events_name = args.events_name
    path = os.path.join(args.out_dir, args.traces_name)
    tr.save(path)
    mem = tr.memory_bytes()
    print(f"built 4 traces from {tr.n_events} events, {len(tr.classes)} classes, "
          f"hd={args.hd_dim}, time_scales={tr.time_scales} ({tr.time_combine}), "
          f"class_weighting={tr.class_weighting}, events={args.events_name}")
    print("memory accounting (bytes):", json.dumps(mem, indent=1))
    if tr.null_calib:
        meta = tr.null_calib.get("_meta", {})
        print("null calibration (peak-sim distribution of no-answer probes; "
              f"v{meta.get('version', 1)}, per decode|trace|probe-kind"
              + (f", range width {meta.get('range_width_fr', 0):.0f} fr"
                 if meta else "") + "):")
        for d, c in sorted(tr.null_calib.items()):
            if d == "_meta":
                continue
            print(f"  {d:22s}: mean={c['mean']:+.5f} std={c['std']:.5f} "
                  f"p95={c['p95']:+.5f} p99={c['p99']:+.5f} "
                  f"(n={c['n']}, mismatched {c.get('n_mismatch', '?')})")
    print(f"saved {path}")


def _parse_when(s):
    if s is None:
        return None
    if ":" in s:
        a, b = s.split(":")
        return (float(a), float(b))
    return float(s)


def cmd_query(args):
    tr = TraceSet.load(os.path.join(args.out_dir, args.traces_name), grid=args.grid)
    router = QueryRouter(tr, z_threshold=args.z_threshold)
    where = tuple(args.where) if args.where else None
    ans, info = router.query(args.decode, what=args.what, where=where,
                             when=_parse_when(args.when))
    print(f"answer: {ans}")
    zs = "n/a (uncalibrated)" if info["z"] is None else \
        f"{info['z']:+.1f} ({'confident' if info['confident'] else 'ABSTAIN'})"
    print(f"  trace={info['trace']}  sim={info['sim']:+.4f}  z={zs}  "
          f"encode={info['ms_encode']:.2f}ms unbind={info['ms_unbind']:.2f}ms "
          f"decode={info['ms_decode']:.2f}ms total={info['ms_total']:.2f}ms")


# ---- bench -----------------------------------------------------------------

def _fmt(v):
    if isinstance(v, tuple):
        return f"({v[0]:+.1f},{v[1]:+.1f})"
    if isinstance(v, float):
        return f"{v:.0f}"
    return str(v)


def _gt_answer(table, gt, pos_bw=GT_POS_BW, time_bw=GT_TIME_BW):
    """Evaluate a recorded ground-truth spec (fn, kwargs) at a bandwidth.
    Class queries have no KDE (conf-sum within radius/window) — bw-free."""
    fn, kw = gt
    if fn == "where":
        return table.where(**kw, bw=pos_bw)
    if fn == "when":
        return table.when(**kw, bw=time_bw)
    return table.what(**kw)


def _score_answer(kind, ans, exact):
    """(err_str, correct_bool_or_None) under the standard thresholds."""
    if exact is None:
        return "n/a (no events)", None
    if kind == "pos":
        e = np.hypot(ans[0] - exact[0], ans[1] - exact[1])
        return f"{e:.2f} m", bool(e <= POS_OK_M)
    if kind == "time":
        e = abs(ans - exact)
        return f"{e:.0f} fr", bool(e <= TIME_OK_FR)
    correct = ans == exact
    return ("OK" if correct else f"got {ans} vs {exact}"), correct


def _bench_battery(router, table, ev):
    """The 21-query battery. Returns structured rows (one per query) with
    error, correctness (pos<=1 m, time<=40 fr, class exact), sim, z, ms.

    Each row also records its ground-truth spec ("gt": (fn, kwargs)) so the
    SAME answers can be re-scored under other KDE bandwidths and under the
    bandwidth-free supported criterion (FIX 1) — query construction (which
    reads modal answers from the table, e.g. tq below) is pinned to the
    default bandwidths so the battery itself never changes."""
    tr = router.tr
    freq = defaultdict(float)
    for c, cf in zip(ev["class"], ev["conf"]):
        freq[c] += cf
    top = sorted(freq, key=freq.get, reverse=True)[:3]
    t_hi = ev["t"].max()
    wins = [(0.0, t_hi / 3), (t_hi / 3, 2 * t_hi / 3), (2 * t_hi / 3, t_hi)]

    rows = []

    def run(desc, kind, decode, vsa_args, gt):
        ans, info = router.query(decode, **vsa_args)
        exact_answer = _gt_answer(table, gt)
        err, correct = _score_answer(kind, ans, exact_answer)
        rows.append({"desc": desc, "kind": kind, "trace": info["trace"],
                     "ans": ans, "exact": exact_answer, "err": err,
                     "correct": correct, "sim": info["sim"], "z": info["z"],
                     "confident": info["confident"], "ms": info["ms_total"],
                     "gt": gt, "null": info.get("null")})

    for c in top:
        # Q1 where(what) all-time  [marginal what_where]
        run(f"where({c}) all-time", "pos", "where", dict(what=c),
            ("where", dict(what=c)))
        # Q2 when(what)            [marginal what_when]
        run(f"when({c})", "time", "when", dict(what=c), ("when", dict(what=c)))
        # Q7 where(what, when-point) [conjunctive event trace]
        tq = table.when(what=c)
        if tq is not None:
            run(f"where({c}, t={tq:.0f})", "pos", "where",
                dict(what=c, when=tq),
                ("where", dict(what=c, when=(tq - 40, tq + 40))))
        # Q10 where(what, when-range) [probe-side range kernel]
        for w in wins[:2]:
            run(f"where({c}, t in {w[0]:.0f}:{w[1]:.0f})", "pos", "where",
                dict(what=c, when=w), ("where", dict(what=c, when=w)))
    if "chair" in tr.classes:
        p = table.where(what="chair")
        # Q3 what(where)  [marginal what_where]
        run(f"what({_fmt(p)})", "class", "what", dict(where=p),
            ("what", dict(near_xy=p, radius=0.8)))
        # Q9 what(where, when-point) [event trace]
        tq = table.when(what="chair", near_xy=p)
        run(f"what({_fmt(p)}, t={tq:.0f})", "class", "what",
            dict(where=p, when=tq),
            ("what", dict(near_xy=p, when=(tq - 60, tq + 60), radius=0.8)))
        # Q8 when(what, where) [event trace]
        run(f"when(chair, {_fmt(p)})", "time", "when",
            dict(what="chair", where=p),
            ("when", dict(what="chair", near_xy=p, radius=0.8)))
    # Q6 where(when-point)  [marginal where_when]
    tq = float(t_hi * 0.12)
    run(f"where(t={tq:.0f})", "pos", "where", dict(when=tq),
        ("where", dict(when=(tq - 20, tq + 20))))
    # Q5 what(when-range)   [marginal what_when + range kernel]
    run(f"what(t in {wins[0][0]:.0f}:{wins[0][1]:.0f})", "class", "what",
        dict(when=wins[0]), ("what", dict(when=wins[0])))
    return rows


def _gt_bandwidth_report(rows, table, bw_list):
    """FIX 1 (tracker entry (r)): the ground truth was estimator-matched —
    ExactEventTable's KDE bandwidths equal the FPE length scales, so table
    and memory could mode-flip together. Re-score the SAME 21 answers (a)
    against ground-truth tables at each position bandwidth in bw_list (time
    bandwidth scaled proportionally: time_bw = GT_TIME_BW * bw/GT_POS_BW)
    and (b) under a bandwidth-free 'supported' criterion (within POS_OK_M /
    TIME_OK_FR of ANY event satisfying the query constraints)."""
    print(f"\n=== ground-truth bandwidth sensitivity "
          f"(same answers, re-scored) ===")
    per_bw = {}
    for bw in bw_list:
        tb = GT_TIME_BW * bw / GT_POS_BW
        per_bw[bw] = []
        for r in rows:
            exact = _gt_answer(table, r["gt"], pos_bw=bw, time_bw=tb)
            _, correct = _score_answer(r["kind"], r["ans"], exact)
            per_bw[bw].append(correct)
    supp = [table.supported(r["kind"], r["ans"], r["gt"][1]) for r in rows]

    hdr = (["query", "conf"] + [f"bw={bw:g}" for bw in bw_list] + ["supported"])
    out = []
    flips = []
    for i, r in enumerate(rows):
        vals = [per_bw[bw][i] for bw in bw_list]
        marks = ["-" if v is None else ("ok" if v else "WRONG") for v in vals]
        sm = "ok" if supp[i] else "UNSUP"
        conf = "-" if r["confident"] is None else ("y" if r["confident"] else "abst")
        out.append([r["desc"], conf] + marks + [sm])
        scored = [v for v in vals if v is not None]
        if len(set(scored)) > 1:
            flips.append(r["desc"])
    widths = [max(len(str(x[i])) for x in out + [hdr]) for i in range(len(hdr))]
    line = "  ".join(h.ljust(w) for h, w in zip(hdr, widths))
    print(line); print("-" * len(line))
    for r in out:
        print("  ".join(str(v).ljust(w) for v, w in zip(r, widths)))

    n_scored = sum(1 for v in per_bw[bw_list[0]] if v is not None)
    print("\naccuracy by ground-truth bandwidth (same answers):")
    for bw in bw_list:
        tb = GT_TIME_BW * bw / GT_POS_BW
        nc = sum(1 for v in per_bw[bw] if v)
        print(f"  pos bw {bw:4g} m / time bw {tb:4.1f} fr: "
              f"{nc}/{n_scored} correct ({100.0 * nc / max(n_scored, 1):.0f}%)")
    ns = sum(1 for s in supp if s)
    print(f"  bandwidth-free supported criterion:  {ns}/{len(rows)} supported "
          f"({100.0 * ns / len(rows):.0f}%)")
    if flips:
        print(f"  correctness FLIPS across bandwidths ({len(flips)}): "
              + "; ".join(flips))
    else:
        print("  no query flips correctness across these bandwidths")
    return per_bw, supp


def _abstention_stats(rows):
    """(n_conf, n_conf_correct, n_abst, n_abst_wrong) over scoreable rows."""
    scored = [r for r in rows if r["correct"] is not None and r["confident"] is not None]
    conf = [r for r in scored if r["confident"]]
    abst = [r for r in scored if not r["confident"]]
    return (len(conf), sum(1 for r in conf if r["correct"]),
            len(abst), sum(1 for r in abst if not r["correct"]))


def _print_calib(tr):
    if not tr.null_calib:
        print("(no null calibration stored; rebuild to enable z/abstention)")
        return
    if any("|" in d for d in tr.null_calib):  # v2 per-(trace, probe-kind)
        print("null calib (v2, per decode|trace|probe-kind):")
        for d, c in sorted(tr.null_calib.items()):
            if d == "_meta":
                continue
            print(f"  {d:22s}: mean={c['mean']:+.5f} std={c['std']:.5f}")
        return
    parts = [f"{d}: mean={c['mean']:+.5f} std={c['std']:.5f}"
             for d, c in tr.null_calib.items()]
    print("null calib (LEGACY v1, one null per decode type)  " + " | ".join(parts))


def cmd_bench(args):
    ev = load_events(args.out_dir, args.events_name)
    table = ExactEventTable(ev)
    if getattr(args, "compare", False):
        return _bench_compare(args, ev, table)
    if getattr(args, "bias_bench", False):
        return _bias_bench(args, ev, table)

    tr = TraceSet.load(os.path.join(args.out_dir, args.traces_name), grid=args.grid)
    router = QueryRouter(tr, z_threshold=args.z_threshold)
    rows = _bench_battery(router, table, ev)

    hdr = ["query", "trace", "vsa", "exact", "err", "peak sim", "z", "abstain", "ms"]
    out = [[r["desc"], r["trace"], _fmt(r["ans"]), _fmt(r["exact"]), r["err"],
            f"{r['sim']:+.4f}",
            "-" if r["z"] is None else f"{r['z']:+.1f}",
            "-" if r["confident"] is None else ("" if r["confident"] else "ABSTAIN"),
            f"{r['ms']:.1f}"] for r in rows]
    widths = [max(len(str(x[i])) for x in out + [hdr]) for i in range(len(hdr))]
    line = "  ".join(h.ljust(w) for h, w in zip(hdr, widths))
    print(line); print("-" * len(line))
    for r in out:
        print("  ".join(str(v).ljust(w) for v, w in zip(r, widths)))

    _print_calib(tr)
    nc, ncc, na, naw = _abstention_stats(rows)
    if nc + na:
        print(f"abstention (z>={args.z_threshold:g}): confident {ncc}/{nc} correct"
              + (f" ({100*ncc/max(nc,1):.0f}%)" if nc else "")
              + f"; abstained {na}, of which wrong {naw}"
              + (f" (abstention precision {100*naw/max(na,1):.0f}%)" if na else ""))

    if getattr(args, "gt_bandwidths", None):
        bw_list = [float(v) for v in args.gt_bandwidths.split(",") if v.strip()]
        _gt_bandwidth_report(rows, table, bw_list)

    print("\nvocabulary scan: which_moved(first-third, last-third), top 5")
    t_hi = ev["t"].max()
    wins = [(0.0, t_hi / 3), (t_hi / 3, 2 * t_hi / 3), (2 * t_hi / 3, t_hi)]
    for r in router.which_moved(wins[0], wins[2], min_sim=0.002)[:5]:
        print(f"  {r['class']:>14}: ({r['a'][0]:+.1f},{r['a'][1]:+.1f}) -> "
              f"({r['b'][0]:+.1f},{r['b'][1]:+.1f})  {r['moved_m']:.2f} m  "
              f"(sim {r['sim_a']:+.3f}/{r['sim_b']:+.3f})")

    mem = tr.memory_bytes()
    print(f"\nmap state: {tr.n_events} events in 4 traces of hd={tr.hd} "
          f"({mem['traces']/1024:.0f} KB traces; {mem['total']/1e6:.1f} MB "
          f"total incl. codebooks/bases/decoder grids)")


def _bench_compare(args, ev, table):
    """Build single-scale / multi-bundle / multi-bind from the same events
    (same seed), run the battery on each, print a compact comparison, and
    save the when-query similarity-curve figure."""
    scales = _parse_scales(args.compare_scales)
    configs = [
        ("single", None, "bundle"),
        ("bundle", scales, "bundle"),
        ("bind", scales, "bind"),
    ]
    all_rows, calibs, curves, mems = {}, {}, {}, {}
    # worked example for the figure: when(chair, near (-1.1,-0.2))
    fig_what, fig_xy = "chair", (-1.1, -0.2)

    for name, sc, comb in configs:
        t0 = time.perf_counter()
        tr = _build_traceset(ev, args.hd_dim, args.seed, args.pos_length_scale,
                             args.time_length_scale, args.grid,
                             time_scales=sc, time_combine=comb,
                             n_null=args.null_samples)
        router = QueryRouter(tr, z_threshold=args.z_threshold)
        all_rows[name] = _bench_battery(router, table, ev)
        calibs[name] = tr.null_calib
        mems[name] = tr.memory_bytes()
        if fig_what in tr.classes:
            res = tr.M["event"] / tr.C[fig_what] / tr.enc.ctx_pos(*fig_xy).values
            curves[name] = (np.real(tr.T_axis @ np.conj(res).astype(np.complex64))
                            / tr.hd)
        print(f"[{name:6s}] built+benched in {time.perf_counter()-t0:.0f}s "
              f"(scales={tr.time_scales}, combine={tr.time_combine})")
        del tr, router  # keep peak memory ~1 decoder grid

    names = [c[0] for c in configs]
    print(f"\n=== bench --compare (hd={args.hd_dim}, seed={args.seed}, "
          f"multi-scales={scales}) ===")
    hdr = ["query"] + [f"{n}: err|sim|ms|conf" for n in names]
    out = []
    for i, r0 in enumerate(all_rows[names[0]]):
        row = [r0["desc"]]
        for n in names:
            r = all_rows[n][i]
            conf = "-" if r["confident"] is None else ("y" if r["confident"] else "ABST")
            row.append(f"{r['err']}|{r['sim']:+.4f}|{r['ms']:.0f}|{conf}")
        out.append(row)
    widths = [max(len(str(x[i])) for x in out + [hdr]) for i in range(len(hdr))]
    line = "  ".join(h.ljust(w) for h, w in zip(hdr, widths))
    print(line); print("-" * len(line))
    for r in out:
        print("  ".join(str(v).ljust(w) for v, w in zip(r, widths)))

    for n in names:
        rows = all_rows[n]
        scored = [r for r in rows if r["correct"] is not None]
        ncorr = sum(1 for r in scored if r["correct"])
        nc, ncc, na, naw = _abstention_stats(rows)
        ms_med = float(np.median([r["ms"] for r in rows]))
        print(f"[{n:6s}] correct {ncorr}/{len(scored)}  median ms {ms_med:.1f}  "
              f"confident {ncc}/{nc} correct  abstained {na} ({naw} wrong)")
        if calibs[n]:
            print("         null " + " | ".join(
                f"{d}: {c['mean']:+.5f}+/-{c['std']:.5f}"
                for d, c in calibs[n].items() if d != "_meta"))

    if curves:
        _save_compare_figure(curves, ev, table, fig_what, fig_xy, args.out_dir)


def _null_key_for(tr, decode, trace, kind):
    """Resolve a null-calibration cell across calibration versions.

    v2 keys are "decode|trace|kind"; v1 stored a single flat key per decode
    type. Returns the best available key, or None if uncalibrated."""
    nc = getattr(tr, "null_calib", None)
    if not nc:
        return None
    for k in (f"{decode}|{trace}|{kind}", decode):
        if k in nc:
            return k
    return next((k for k in nc if k != "_meta" and k.startswith(f"{decode}|")), None)


def _bias_bench(args, ev, table):
    """Class-frequency bias bench (the sqrt-N magnitude-bias limitation).

    Question: does --class-weighting balanced/log lift rare classes above
    the calibrated noise floor WITHOUT sinking frequent ones below it?

    Picks a LOW/MID/HIGH frequency tercile (3 classes each, all with
    >= --bias-min-events events) from the event stream, builds conf /
    balanced / log tracesets from the SAME events and seed, and runs the
    all-time where(what=c) query (marginal what_where trace) for each class:
    peak sim, calibrated z, confident flag, and position error vs the exact
    table's modal answer."""
    counts = Counter(ev["class"])
    elig = sorted((c for c in counts if counts[c] >= args.bias_min_events),
                  key=lambda c: counts[c])
    if len(elig) < 9:
        print(f"only {len(elig)} classes with >= {args.bias_min_events} events; "
              "need 9 (3 per tercile)")
        return
    mid0 = (len(elig) - 3) // 2
    terciles = [("LOW", elig[:3]), ("MID", elig[mid0:mid0 + 3]),
                ("HIGH", elig[-3:])]
    picks = [(t, c) for t, cl in terciles for c in cl]
    weightings = ("conf", "balanced", "log")

    results, calibs = {}, {}
    for wmode in weightings:
        t0 = time.perf_counter()
        tr = _build_traceset(ev, args.hd_dim, args.seed, args.pos_length_scale,
                             args.time_length_scale, args.grid,
                             n_null=args.null_samples, class_weighting=wmode)
        router = QueryRouter(tr, z_threshold=args.z_threshold)
        res = {}
        for _, c in picks:
            ans, info = router.query("where", what=c)
            exact = table.where(what=c)
            err = float(np.hypot(ans[0] - exact[0], ans[1] - exact[1]))
            res[c] = {"sim": info["sim"], "z": info["z"],
                      "confident": info["confident"], "err": err, "ans": ans}
        results[wmode] = res
        # v2 nulls are keyed "decode|trace|kind"; these are all-time where(what)
        # marginal queries, so report the matching cell (v1 fallback: "where").
        wkey = _null_key_for(tr, "where", "what_where", "none")
        calibs[wmode] = tr.null_calib[wkey] if wkey else None
        print(f"[{wmode:8s}] built+queried in {time.perf_counter() - t0:.0f}s"
              + ("" if not calibs[wmode] else
                 f"  null({wkey}): mean={calibs[wmode]['mean']:+.5f} "
                 f"std={calibs[wmode]['std']:.5f}"))
        del tr, router

    print(f"\n=== bias bench: all-time where(what) per frequency tercile "
          f"(hd={args.hd_dim}, seed={args.seed}, z>={args.z_threshold:g}) ===")
    hdr = ["terc", "class (K events)"] + [f"{w}: sim|z|err" for w in weightings]
    out = []
    for terc, c in picks:
        row = [terc, f"{c} ({counts[c]})"]
        for w in weightings:
            r = results[w][c]
            conf = "-" if r["confident"] is None else \
                ("ok" if r["confident"] else "ABST")
            row.append(f"{r['sim']:+.4f}|{r['z']:+5.1f} {conf}|{r['err']:.2f} m")
        out.append(row)
    widths = [max(len(str(x[i])) for x in out + [hdr]) for i in range(len(hdr))]
    line = "  ".join(h.ljust(w) for h, w in zip(hdr, widths))
    print(line); print("-" * len(line))
    for r in out:
        print("  ".join(str(v).ljust(w) for v, w in zip(r, widths)))

    print("\nper-tercile summary (confident count / mean peak sim / mean pos err):")
    for terc, cl in terciles:
        parts = []
        for w in weightings:
            rs = [results[w][c] for c in cl]
            nconf = sum(1 for r in rs if r["confident"])
            parts.append(f"{w}: {nconf}/3 conf, sim {np.mean([r['sim'] for r in rs]):+.4f}, "
                         f"err {np.mean([r['err'] for r in rs]):.2f} m")
        print(f"  {terc:4s}  " + "  |  ".join(parts))


def _save_compare_figure(curves, ev, table, what, xy, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cls_arr = np.array(ev["class"])
    m = (cls_arr == what) & (np.hypot(ev["x"] - xy[0], ev["y"] - xy[1]) <= 1.0)
    gt_times = ev["t"][m]
    t_exact = table.when(what=what, near_xy=xy, radius=0.8)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    styles = {"single": ("C0", "-"), "bundle": ("C1", "-"), "bind": ("C2", "-")}
    for name, s in curves.items():
        c, ls = styles.get(name, ("k", "-"))
        ax.plot(np.arange(len(s)), s, ls, color=c, lw=1.2,
                label=f"{name} (peak {s.max():+.4f} @ t={int(np.argmax(s))})")
    if len(gt_times):
        ax.plot(gt_times, np.full(len(gt_times), ax.get_ylim()[0]), "|",
                color="0.4", ms=10, label=f"{what} events within 1 m")
    if t_exact is not None:
        ax.axvline(t_exact, color="k", ls="--", lw=1,
                   label=f"exact modal t={t_exact:.0f}")
    ax.set_xlabel("t (frame)")
    ax.set_ylabel("similarity  Re<T(t), residual>/hd")
    ax.set_title(f"when({what}, near ({xy[0]:+.1f},{xy[1]:+.1f})): "
                 "M_event / C / S correlated with the time axis")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "multiscale_time_comparison.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"\nsaved figure -> {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("export", help="detections + poses -> canonical events csv")
    pe.add_argument("--place-mode", choices=["robot", "object"], default="robot",
                    help="robot = pose at observation time (default); object = "
                         "depth-localized object world position (falls back to "
                         "robot pose when depth is invalid)")
    pe.add_argument("--events-name", default=None,
                    help="output csv name (default events.csv for robot, "
                         "events_object.csv for object)")
    pe.add_argument("--min-class-events", type=int, default=0,
                    help="drop classes with fewer than N events total (0=off)")
    pe.add_argument("--class-blocklist", default="",
                    help='comma list of class names to drop, e.g. "airplane,cat"')

    pb = sub.add_parser("build", help="events csv -> 4-trace map state")
    pb.add_argument("--class-weighting", choices=["conf", "balanced", "log"],
                    default="conf",
                    help="per-event bundle weights: conf (original), balanced "
                         "(conf/sqrt(class conf mass)), log (conf*log(1+K)/K); "
                         "see _class_weights")
    pb.add_argument("--class-seed-mix", type=int, default=None,
                    help="mix this int into every class-atom seed (md5 of "
                         '"name|mix"); default None = historical '
                         "_class_seed(name) atoms (all logged artifacts)")

    pq = sub.add_parser("query", help="single routed query")
    pq.add_argument("--decode", required=True, choices=["where", "when", "what"],
                    help="which factor to decode")
    pq.add_argument("--what", default=None)
    pq.add_argument("--where", type=float, nargs=2, default=None, metavar=("X", "Y"))
    pq.add_argument("--when", default=None, help="t or a:b (range)")

    pn = sub.add_parser("bench", help="query battery vs exact event table")
    pn.add_argument("--compare", action="store_true",
                    help="build single/bundle/bind in memory and compare")
    pn.add_argument("--compare-scales", default="5,20,80",
                    help="time scales for the multi-scale configs")
    pn.add_argument("--bias-bench", action="store_true",
                    help="class-frequency bias bench: build conf/balanced/log "
                         "in memory, report per-tercile where(what) recall")
    pn.add_argument("--bias-min-events", type=int, default=8,
                    help="tercile candidates need at least this many events")
    pn.add_argument("--gt-bandwidths", default=None,
                    help='comma list of ground-truth position KDE bandwidths '
                         '(m), e.g. "0.4,0.75,1.5"; time bandwidth is scaled '
                         "proportionally. Re-scores the same battery answers "
                         "per bandwidth AND under a bandwidth-free supported "
                         "criterion (estimator-matched-GT hardening)")

    for p in (pb, pn):  # build knobs (bench --compare/--bias-bench rebuild in memory)
        p.add_argument("--hd-dim", type=int, default=8192)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--pos-length-scale", type=float, default=0.75)
        p.add_argument("--time-length-scale", type=float, default=20.0)
        p.add_argument("--null-samples", type=int, default=200,
                       help="null probes per decode type for calibration (0=skip)")
    pb.add_argument("--time-scales", default=None,
                    help='comma list, e.g. "5,20,80" (default: single scale '
                         "at --time-length-scale)")
    pb.add_argument("--time-combine", choices=["bundle", "bind"], default="bundle",
                    help="multi-scale combination (see TraceSet docstring)")

    for p in (pq, pn):
        p.add_argument("--z-threshold", type=float, default=3.0,
                       help="confidence threshold on the null-calibrated z-score")

    for p in (pb, pn):  # which event stream to read (export writes it)
        p.add_argument("--events-name", default=EVENTS_CSV,
                       help="events csv to load (e.g. events_object.csv)")
    for p in (pb, pq, pn):
        p.add_argument("--traces-name", default=TRACES_PT,
                       help="trace-state .pt filename in --out-dir")

    for p in (pe, pb, pq, pn):
        p.add_argument("--out-dir", default=DEFAULT_OUT)
        p.add_argument("--grid", type=int, default=72)

    args = ap.parse_args()
    {"export": cmd_export, "build": cmd_build,
     "query": cmd_query, "bench": cmd_bench}[args.cmd](args)


if __name__ == "__main__":
    main()
