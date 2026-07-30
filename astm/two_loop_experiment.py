"""Two-loop pseudo-timeline experiment: revisit/looping machinery test.

HONEST CAVEAT (read first)
--------------------------
The classroom walk is a SINGLE loop: every place is visited once, so no
revisit / loop-closure structure exists in the real data. This script
manufactures a FAKE SECOND LOOP by frame interleaving: original event rows
carry times t = 0..1238 (stride-2 frame rows). Events with EVEN original t
become loop 1 at t' = t/2; events with ODD original t become loop 2 at
t' = 620 + (t-1)/2. Positions are untouched. The pseudo-timeline therefore
visits every location twice, ~620 pseudo-frames apart.

Interleaved frames are ~0.1 s apart in the real world, so the two "visits"
have NEARLY IDENTICAL appearance, lighting, and viewpoint. This is a
MACHINERY test — does the trace algebra express and decode revisit
structure (bimodal occupancy, cross-loop consistency, cross-loop change
detection, content-keyed loop-closure recall)? It is NOT a robustness test
of any of those against real appearance change. Real multi-loop data is
required for that.

Four tests (all run by the CLI; a scored report is printed):

  1. BIMODAL OCCUPANCY — when(place) from M_where_when on the two-loop
     stream shows TWO peaks ~620 frames apart (vs ONE on the single-loop
     stream), gated by a fake-place null (matched decode, random unit
     phasors as position keys).
  2. CROSS-LOOP CONSISTENCY (negative control) — where(class, loop-1) vs
     where(class, loop-2) via probe-side half-window range kernels for the
     6 most frequent classes: displacement ~0 (nothing moved), and the
     null-gated which_moved(loop1, loop2) reports no class above gate with
     displacement > 1 m. Null gate = range-matched fake-class null (random
     unit-phasor classes decoded with the SAME half-window kernels —
     technique from moved_object_synthetic.py; the build-time point-probe
     null does not transfer to range kernels).
  3. MOVED OBJECT ACROSS LOOPS (positive control) — relocate one class's
     LOOP-2 events by an in-bounds ~1.5-2 m offset; the gated
     which_moved(loop1, loop2) must rank it #1 with displacement ~ the
     injected offset.
  4. LOOP-CLOSURE RECALL (content level) — PCA-64-whitened frame embeddings
     (whitening statistics fit on the EVEN frames only and applied to the
     held-out odd frames — fitting on all frames would leak)
     projected to hd=8192 phasors (seed 0), episodic memory
     mean(content_n * ctx_pos_n) over EVEN frames only; every ODD frame
     unbinds by its content and decodes position on a 56-grid. Position
     error stats = the "have I been here before?" signal on held-out
     revisit frames, vs a shuffled-content null.

hd_dim for the trace tests defaults to 16384 (not 8192): half-window range
kernels concentrate probe energy into ~2% of components; at 8192 the
surviving signal sits within ~2.5 std of the fake-class null floor
(measured in moved_object_synthetic.py). Trace builds use log
class-weighting so tv-level classes clear the null gate.

Figure -> outputs/classroom/two_loop_experiment.png.

Usage (from repo root, needs outputs/classroom/events.csv + embeddings.pt):

    python vsa_cognitive_mapping/two_loop_experiment.py
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vsa_cognitive_mapping.astm_traces import (  # noqa: E402
    ExactEventTable, QueryRouter, TraceSet, load_events, _build_traceset,
)
from vsa_cognitive_mapping.classroom_pipeline import (  # noqa: E402
    ClassroomEncoders, _load_embeddings, load_poses_interpolated,
)
from vsa_cognitive_mapping.vsa import (  # noqa: E402
    pca_components, random_project_to_phasor,
)

DEFAULT_OUT = os.path.join("outputs", "classroom")
LOOP_OFFSET = 620          # ceil(1239/2): loop-2 pseudo-times start here
SEP_LO, SEP_HI = 500, 740  # accepted peak-pair separation window


# ==========================================================================
# Stream construction
# ==========================================================================

def remap_two_loop(ev):
    """Even original t -> loop 1 (t' = t/2); odd -> loop 2 (t' = 620+(t-1)/2)."""
    ti = np.rint(ev["t"]).astype(int)
    t_new = np.where(ti % 2 == 0, ti // 2, LOOP_OFFSET + (ti - 1) // 2)
    out = {"class": list(ev["class"]), "x": ev["x"].copy(), "y": ev["y"].copy(),
           "t": t_new.astype(float), "conf": ev["conf"].copy()}
    return out


def build(ev, hd, seed, pos_l, time_l, grid, weighting="log"):
    """In-memory TraceSet, no build-time null calibration (this experiment
    uses its own matched nulls, per decode type)."""
    return _build_traceset(ev, hd, seed, pos_l, time_l, grid,
                           n_null=0, class_weighting=weighting)


# ==========================================================================
# Matched nulls
# ==========================================================================

def fake_place_null(tr, n=30, seed=777):
    """Max of the when-curve for random unit-phasor 'places' not on the
    trajectory: matched null for the M_where_when / S point-probe decode."""
    rng = np.random.RandomState(seed)
    maxima = []
    for _ in range(n):
        Sf = np.exp(1j * rng.uniform(-np.pi, np.pi, tr.hd))
        res = tr.M["where_when"] / Sf
        s = np.real(tr.T_axis @ np.conj(res).astype(np.complex64)) / tr.hd
        maxima.append(float(s.max()))
    m = np.array(maxima)
    return {"mean": float(m.mean()), "std": float(m.std()),
            "gate": float(m.mean() + 3.0 * m.std())}


def fake_class_null(tr, router, windows, n=60, seed=777):
    """Range-matched null (moved_object_synthetic technique): half-window
    'where' decode sims for random unit-phasor classes NOT in the stream."""
    rng = np.random.RandomState(seed)
    sims = []
    for _ in range(n):
        Cf = np.exp(1j * rng.uniform(-np.pi, np.pi, tr.hd))
        res = tr.M["event"] / Cf
        for w in windows:
            _, s = router._decode_grid(res, probe_time=tr.time_range(*w))
            sims.append(s)
    sims = np.array(sims)
    return {"mean": float(sims.mean()), "std": float(sims.std()),
            "min_sim": float(sims.mean() + 3.0 * sims.std())}


# ==========================================================================
# Test 1: bimodal occupancy
# ==========================================================================

def when_curve(tr, xy):
    res = tr.M["where_when"] / tr.enc.ctx_pos(*xy).values
    return np.real(tr.T_axis @ np.conj(res).astype(np.complex64)) / tr.hd


def peak_pair(curve, gate, excl=250):
    """Two local maxima 500-740 apart, both above gate. Returns dict."""
    t1 = int(np.argmax(curve))
    c2 = curve.copy()
    c2[max(0, t1 - excl):min(len(c2), t1 + excl + 1)] = -np.inf
    t2 = int(np.argmax(c2))
    lo, hi = sorted((t1, t2))

    def is_local_max(t):
        left = curve[t - 1] if t > 0 else -np.inf
        right = curve[t + 1] if t < len(curve) - 1 else -np.inf
        return curve[t] >= left and curve[t] >= right
    sep = hi - lo
    ok = (curve[t1] > gate and curve[t2] > gate
          and SEP_LO <= sep <= SEP_HI
          and is_local_max(t1) and is_local_max(t2))
    return {"t_lo": lo, "t_hi": hi, "sep": sep,
            "s_lo": float(curve[lo]), "s_hi": float(curve[hi]),
            "pass": bool(ok)}


def pick_positions(ev, orig_times):
    """Robot xy at (the nearest event row to) each requested original t."""
    ti = np.rint(ev["t"]).astype(int)
    out = []
    for t0 in orig_times:
        k = int(np.argmin(np.abs(ti - t0)))
        out.append((float(ev["x"][k]), float(ev["y"][k]), int(ti[k])))
    return out


def test1(tr_two, tr_single, ev_orig):
    print("\n=== TEST 1: bimodal occupancy — when(place) on the two-loop stream ===")
    null = fake_place_null(tr_two)
    print(f"  fake-place null (when-curve maxima, n=30): mean={null['mean']:.5f} "
          f"std={null['std']:.5f} -> gate={null['gate']:.5f}")
    positions = pick_positions(ev_orig, [150, 400, 600, 850, 1100])
    rows, worked = [], None
    for (px, py, t0) in positions:
        curve = when_curve(tr_two, (px, py))
        r = peak_pair(curve, null["gate"])
        r.update({"xy": (px, py), "t_orig": t0,
                  "expect": (t0 // 2, LOOP_OFFSET + t0 // 2)})
        rows.append(r)
        print(f"  pos ({px:+.2f},{py:+.2f}) [orig t={t0:4d}, expect "
              f"t'~({t0 // 2},{LOOP_OFFSET + t0 // 2})]: peaks t'=({r['t_lo']},"
              f"{r['t_hi']}) sep={r['sep']} sims=({r['s_lo']:.5f},"
              f"{r['s_hi']:.5f}) -> {'PASS' if r['pass'] else 'FAIL'}")
        if worked is None and r["pass"]:
            worked = {"xy": (px, py), "t_orig": t0, "curve": curve, "row": r}
    if worked is None:  # fall back to the first position for the figure
        worked = {"xy": rows[0]["xy"], "t_orig": rows[0]["t_orig"],
                  "curve": when_curve(tr_two, rows[0]["xy"]), "row": rows[0]}
    # single-loop counterpart for the worked example
    sl_curve = when_curve(tr_single, worked["xy"])
    sl_null = fake_place_null(tr_single)
    sl = peak_pair(sl_curve, sl_null["gate"])
    worked["sl_curve"], worked["sl_row"], worked["sl_gate"] = \
        sl_curve, sl, sl_null["gate"]
    worked["gate"] = null["gate"]
    single_unimodal = not sl["pass"]  # single loop must NOT show a valid pair
    print(f"  single-loop counterpart at ({worked['xy'][0]:+.2f},"
          f"{worked['xy'][1]:+.2f}): global peak t={int(np.argmax(sl_curve))} "
          f"(second candidate at sep={sl['sep']}, "
          f"pair-{'FOUND (unexpected)' if sl['pass'] else 'ABSENT as expected'})")
    n_pass = sum(r["pass"] for r in rows)
    ok = n_pass >= 4 and single_unimodal
    print(f"  TEST 1 {'PASS' if ok else 'FAIL'}: {n_pass}/5 positions bimodal; "
          f"single-loop unimodal={single_unimodal}")
    return {"rows": rows, "worked": worked, "n_pass": n_pass, "pass": ok,
            "null": null}


# ==========================================================================
# Test 2: cross-loop consistency (negative control)
# ==========================================================================

def test2(tr, ev_two, wins):
    print("\n=== TEST 2: cross-loop consistency (negative control) ===")
    router = QueryRouter(tr)
    null = fake_class_null(tr, router, wins)
    print(f"  fake-class null (half-window where, n=60x2): "
          f"mean={null['mean']:.5f} std={null['std']:.5f} "
          f"-> min_sim={null['min_sim']:.5f}")
    counts = Counter(ev_two["class"])
    top6 = [c for c, _ in counts.most_common(6)]
    disp = {}
    for c in top6:
        pa, ia = router.query("where", what=c, when=wins[0])
        pb, ib = router.query("where", what=c, when=wins[1])
        d = float(np.hypot(pb[0] - pa[0], pb[1] - pa[1]))
        gated = min(ia["sim"], ib["sim"]) >= null["min_sim"]
        disp[c] = {"a": pa, "b": pb, "d": d, "sim_a": ia["sim"],
                   "sim_b": ib["sim"], "gated": gated}
        print(f"  where({c:>12}): loop1 ({pa[0]:+.2f},{pa[1]:+.2f})  "
              f"loop2 ({pb[0]:+.2f},{pb[1]:+.2f})  displacement {d:.2f} m  "
              f"sims {ia['sim']:.5f}/{ib['sim']:.5f} "
              f"[{'above' if gated else 'BELOW'} gate]")
    ranked = router.which_moved(wins[0], wins[1], min_sim=null["min_sim"])
    print(f"  which_moved(loop1, loop2), {len(ranked)} classes pass gate, top 3:")
    for i, r in enumerate(ranked[:3]):
        print(f"    #{i + 1} {r['class']:>12}: {r['moved_m']:.2f} m "
              f"(sims {r['sim_a']:.5f}/{r['sim_b']:.5f})")
    max_d6 = max(v["d"] for v in disp.values())
    n_small = sum(v["d"] < 0.5 for v in disp.values())
    outliers = [c for c, v in disp.items() if v["d"] >= 0.5]
    movers = [r for r in ranked if r["moved_m"] > 1.0]
    # PASS = the actual change-detection mechanism is clean (no gated class
    # >1 m) AND any raw-displacement outlier among the top 6 is a low-
    # confidence mode-flip (below the range-matched gate), not a confident
    # detection. Raw argmax on multimodal classes (person walks around) can
    # flip between near-tied modes; the gate is what screens those out.
    outlier_confident = [c for c in outliers if disp[c]["gated"]]
    ok = (not movers) and (not outlier_confident)
    print(f"  TEST 2 {'PASS' if ok else 'FAIL'}: {n_small}/6 classes <0.5 m "
          f"(outliers {outliers or 'none'} — all below gate: "
          f"{not outlier_confident}); gated classes >1 m: "
          f"{[r['class'] for r in movers] or 'none'}")
    return {"top6": top6, "disp": disp, "ranked": ranked, "null": null,
            "max_d6": max_d6, "n_small": n_small, "outliers": outliers,
            "pass": ok}


# ==========================================================================
# Test 3: moved object across loops (positive control)
# ==========================================================================

OFFSET_CANDIDATES = [
    (1.5, 0.75), (-1.5, 0.75), (1.5, -0.75), (-1.5, -0.75),
    (1.8, 0.9), (-1.8, 0.9), (1.8, -0.9), (-1.8, -0.9),
    (1.2, 0.9), (-1.2, 0.9), (1.2, -0.9), (-1.2, -0.9),
    (1.0, 0.5), (-1.0, 0.5), (1.0, -0.5), (-1.0, -0.5),
]


def choose_offset(x2, y2, bounds):
    for dx, dy in OFFSET_CANDIDATES:
        if (x2.min() + dx >= bounds[0] and x2.max() + dx <= bounds[1]
                and y2.min() + dy >= bounds[2] and y2.max() + dy <= bounds[3]):
            return float(dx), float(dy)
    raise SystemExit("no candidate offset keeps the moved cluster inside "
                     "the walk")


def test3(ev_two, wins, hd, seed, pos_l, time_l, grid, what="tv",
          radius=2.0):
    """Relocate the class's DOMINANT LOOP-2 SPATIAL CLUSTER (events within
    `radius` of the loop-2 modal position). In the interleaved stream every
    class's loop-2 events span the whole walk (unlike the real first/second
    halves in moved_object_synthetic), so shifting ALL of them cannot stay
    in bounds; moving the modal cluster = 'this instance moved between
    loops' and keeps the modal-peak displacement equal to the injection."""
    print(f"\n=== TEST 3: moved object across loops (positive control, "
          f"class={what}) ===")
    cls = np.array(ev_two["class"])
    in2 = (cls == what) & (ev_two["t"] >= wins[1][0])   # loop-2 events
    n1 = int(((cls == what) & (ev_two["t"] < wins[1][0])).sum())
    bounds = (float(ev_two["x"].min()), float(ev_two["x"].max()),
              float(ev_two["y"].min()), float(ev_two["y"].max()))
    p_star = ExactEventTable(ev_two).where(what=what, when=wins[1])
    m2 = in2 & (np.hypot(ev_two["x"] - p_star[0],
                         ev_two["y"] - p_star[1]) <= radius)
    dx, dy = choose_offset(ev_two["x"][m2], ev_two["y"][m2], bounds)
    inj = float(np.hypot(dx, dy))
    print(f"  {what}: {n1} loop-1 events, {int(in2.sum())} loop-2 events; "
          f"moved cluster = {int(m2.sum())} loop-2 events within {radius} m "
          f"of the loop-2 modal ({p_star[0]:+.2f},{p_star[1]:+.2f})")
    print(f"  injected loop-2 offset ({dx:+.2f},{dy:+.2f}) m = {inj:.2f} m")

    ev_mod = {"class": list(ev_two["class"]), "x": ev_two["x"].copy(),
              "y": ev_two["y"].copy(), "t": ev_two["t"].copy(),
              "conf": ev_two["conf"].copy()}
    ev_mod["x"][m2] += dx
    ev_mod["y"][m2] += dy

    tr = build(ev_mod, hd, seed, pos_l, time_l, grid)
    router = QueryRouter(tr)
    null = fake_class_null(tr, router, wins)
    print(f"  fake-class null: mean={null['mean']:.5f} std={null['std']:.5f} "
          f"-> min_sim={null['min_sim']:.5f}")
    ranked = router.which_moved(wins[0], wins[1], min_sim=null["min_sim"])
    ranked_all = router.which_moved(wins[0], wins[1], min_sim=0.0)
    rank = next((i + 1 for i, r in enumerate(ranked) if r["class"] == what),
                None)
    row = next((r for r in ranked_all if r["class"] == what), None)
    print(f"  which_moved(loop1, loop2), {len(ranked)} classes pass gate, top 3:")
    for i, r in enumerate(ranked[:3]):
        mark = "  <-- injected" if r["class"] == what else ""
        print(f"    #{i + 1} {r['class']:>12}: ({r['a'][0]:+.2f},{r['a'][1]:+.2f})"
              f" -> ({r['b'][0]:+.2f},{r['b'][1]:+.2f})  {r['moved_m']:.2f} m "
              f"(sims {r['sim_a']:.5f}/{r['sim_b']:.5f}){mark}")
    meas = row["moved_m"] if row else float("nan")
    ok = (rank == 1) and abs(meas - inj) < 0.5
    print(f"  TEST 3 {'PASS' if ok else 'FAIL'}: rank={rank} (want 1), "
          f"measured {meas:.2f} m vs injected {inj:.2f} m")
    del tr, router
    return {"what": what, "inj": inj, "offset": (dx, dy), "rank": rank,
            "measured": meas, "ranked": ranked, "pass": ok, "null": null}


# ==========================================================================
# Test 4: loop-closure recall (content level)
# ==========================================================================

def _poses_for_frames(out_dir, ts, ev_orig):
    """load_poses_interpolated (standard path), with an events.csv fallback
    (robot-mode events carry the robot pose per frame row) if the HF dataset
    is unreachable."""
    try:
        x, y, _psi = load_poses_interpolated(ts)
        return np.asarray(x, float), np.asarray(y, float), "lio-sam"
    except Exception as e:  # offline fallback
        print(f"  (load_poses_interpolated failed: {e!r}; "
              "falling back to events.csv robot poses)")
        ti = np.rint(ev_orig["t"]).astype(int)
        n = len(ts)
        xs = np.full(n, np.nan)
        ys = np.full(n, np.nan)
        for k in range(len(ti)):
            xs[ti[k]] = ev_orig["x"][k]
            ys[ti[k]] = ev_orig["y"][k]
        idx = np.arange(n)
        good = ~np.isnan(xs)
        xs = np.interp(idx, idx[good], xs[good])
        ys = np.interp(idx, idx[good], ys[good])
        return xs, ys, "events-fallback"


def test4(out_dir, ev_orig, hd=8192, n_pca=64, seed=0, pos_l=0.75, grid_n=56):
    print("\n=== TEST 4: loop-closure recall (content level) ===")
    _fi, ts, emb = _load_embeddings(out_dir)
    n = emb.shape[0]
    even = np.arange(0, n, 2)
    odd = np.arange(1, n, 2)
    print(f"  {n} frame embeddings {emb.shape}; PCA-{n_pca} whiten "
          f"(fit on EVEN frames only, applied to odd — no test-frame "
          f"leakage) -> random_project_to_phasor(hd={hd}, seed={seed})")
    # PCA whitening fit on the STORED (even) half only; the held-out odd
    # frames are transformed with the even-fit statistics (mean, basis,
    # per-component std) — pca_components on all frames would leak.
    emb64 = emb.astype(np.float64)
    k = min(n_pca, emb64.shape[1], len(even) - 1)
    mu = emb64[even].mean(axis=0, keepdims=True)
    _, _sv, vt = np.linalg.svd(emb64[even] - mu, full_matrices=False)
    scores = (emb64 - mu) @ vt[:k].T
    std_even = scores[even].std(axis=0, keepdims=True) + 1e-8
    scores = scores / std_even
    z, _W = random_project_to_phasor(torch.from_numpy(scores).float(), hd,
                                     seed=seed)
    Z = z.numpy().astype(np.complex64)          # (n, hd) unit-modulus
    x, y, pose_src = _poses_for_frames(out_dir, ts, ev_orig)
    print(f"  poses: {pose_src}")

    enc = ClassroomEncoders(hd, seed, pos_l, 20.0)
    P = np.empty((n, hd), np.complex64)
    for i in range(n):
        P[i] = enc.ctx_pos(float(x[i]), float(y[i])).values
    M = (Z[even].astype(np.complex128) * P[even]).mean(axis=0)
    print(f"  episodic memory: mean over {len(even)} EVEN frames of "
          f"content (x) ctx_pos; querying {len(odd)} ODD (revisit) frames")

    # 56-grid decoder (aspect-matched y, 1 m pad — TraceSet convention)
    pad = 1.0
    bx = (x.min() - pad, x.max() + pad)
    by = (y.min() - pad, y.max() + pad)
    gx = np.linspace(bx[0], bx[1], grid_n)
    gy = np.linspace(by[0], by[1],
                     max(8, int(grid_n * (by[1] - by[0]) / (bx[1] - bx[0]))))
    G = np.empty((len(gy) * len(gx), hd), np.complex64)
    k = 0
    for yy in gy:
        for xx in gx:
            G[k] = enc.ctx_pos(float(xx), float(yy)).values
            k += 1

    def decode_errors(content_rows, frame_ids):
        # residual = M / content = M * conj(content) (unit modulus)
        R = (M[None, :] * np.conj(content_rows)).astype(np.complex64)
        S = np.real(np.conj(R) @ G.T) / hd          # (n_query, n_grid)
        kk = np.argmax(S, axis=1)
        jj, ii = np.divmod(kk, len(gx))
        return np.hypot(gx[ii] - x[frame_ids], gy[jj] - y[frame_ids])

    err = decode_errors(Z[odd], odd)
    rng = np.random.RandomState(123)
    perm = rng.permutation(len(odd))
    err_null = decode_errors(Z[odd][perm], odd)     # shuffled-content null

    def stats(e):
        return (float(e.mean()), float(np.median(e)),
                float(np.percentile(e, 90)),
                float((e <= 0.5).mean()), float((e <= 1.0).mean()))
    sm = stats(err)
    sn = stats(err_null)
    print(f"  recall error (odd frames vs even-frame memory): "
          f"mean={sm[0]:.2f} m median={sm[1]:.2f} m p90={sm[2]:.2f} m; "
          f"within 0.5 m: {100 * sm[3]:.1f}%  within 1.0 m: {100 * sm[4]:.1f}%")
    print(f"  shuffled-content null:                          "
          f"mean={sn[0]:.2f} m median={sn[1]:.2f} m p90={sn[2]:.2f} m; "
          f"within 0.5 m: {100 * sn[3]:.1f}%  within 1.0 m: {100 * sn[4]:.1f}%")
    ok = (sm[1] < 0.5) and (sn[1] > 2.0 * sm[1])
    print(f"  TEST 4 {'PASS' if ok else 'FAIL'}: median {sm[1]:.2f} m "
          f"(<0.5 wanted), null median {sn[1]:.2f} m (>> real wanted)")
    return {"err": err, "err_null": err_null, "stats": sm, "stats_null": sn,
            "pose_src": pose_src, "pass": ok}


# ==========================================================================
# Figure
# ==========================================================================

def figure(r1, r2, r3, r4, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    # (a) bimodal when-curve vs single-loop counterpart
    ax = axes[0]
    w = r1["worked"]
    ax.plot(w["curve"], "C0-", lw=1.1,
            label=f"two-loop (peaks t'={w['row']['t_lo']},{w['row']['t_hi']}, "
                  f"sep {w['row']['sep']})")
    ax.plot(w["sl_curve"], "C1-", lw=1.1, alpha=0.8,
            label=f"single-loop (peak t={int(np.argmax(w['sl_curve']))})")
    ax.axhline(w["gate"], color="C0", ls=":", lw=1, label="two-loop null gate")
    ax.axhline(w["sl_gate"], color="C1", ls=":", lw=1, alpha=0.8,
               label="single-loop null gate")
    for t in (w["row"]["t_lo"], w["row"]["t_hi"]):
        ax.axvline(t, color="C0", ls="--", lw=0.7, alpha=0.5)
    ax.set_xlabel("t' (pseudo-frame)")
    ax.set_ylabel("similarity  Re<T(t'), M_where_when/S>/hd")
    ax.set_title(f"(a) when(place=({w['xy'][0]:+.1f},{w['xy'][1]:+.1f})): "
                 "bimodal under the fake 2nd loop", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # (b) cross-loop displacement bars
    ax = axes[1]
    names = r2["top6"] + [f"{r3['what']}*\n(injected)"]
    vals = [r2["disp"][c]["d"] for c in r2["top6"]] + [r3["measured"]]
    colors = ["C0"] * len(r2["top6"]) + ["C3"]
    ax.bar(range(len(vals)), vals, color=colors)
    ax.axhline(0.5, color="k", ls=":", lw=1, label="0.5 m (control bound)")
    ax.axhline(r3["inj"], color="C3", ls="--", lw=1,
               label=f"injected {r3['inj']:.2f} m")
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(names, fontsize=7, rotation=30, ha="right")
    ax.set_ylabel("loop1 -> loop2 decoded displacement (m)")
    ax.set_title("(b) cross-loop displacement: 6 controls + injected mover",
                 fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")

    # (c) loop-closure recall error histogram
    ax = axes[2]
    bins = np.linspace(0, max(4.0, float(r4["err_null"].max())), 40)
    ax.hist(r4["err"], bins=bins, alpha=0.7, color="C0",
            label=f"revisit recall (median {r4['stats'][1]:.2f} m)")
    ax.hist(r4["err_null"], bins=bins, alpha=0.5, color="0.5",
            label=f"shuffled-content null (median {r4['stats_null'][1]:.2f} m)")
    ax.axvline(0.5, color="k", ls=":", lw=1)
    ax.set_xlabel("position error (m), odd frames vs even-frame memory")
    ax.set_ylabel("frames")
    ax.set_title("(c) loop-closure recall: content (x) ctx_pos episodic memory",
                 fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    fig.suptitle("Two-loop PSEUDO-timeline (frame-interleaved single walk): "
                 "machinery test for revisit structure — interleaved frames "
                 "are ~0.1 s apart, appearance nearly identical; NOT a "
                 "robustness test", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ==========================================================================
# Main
# ==========================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--hd-dim", type=int, default=16384,
                    help="trace tests (range-kernel headroom; see docstring)")
    ap.add_argument("--content-hd", type=int, default=8192,
                    help="test 4 content path")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pos-length-scale", type=float, default=0.75)
    ap.add_argument("--time-length-scale", type=float, default=20.0)
    ap.add_argument("--grid", type=int, default=72)
    ap.add_argument("--moved-class", default="tv")
    args = ap.parse_args()

    t_start = time.time()
    print("TWO-LOOP PSEUDO-TIMELINE EXPERIMENT")
    print("CAVEAT: fake second loop by frame interleaving (even t -> loop 1 "
          "at t/2, odd t -> loop 2 at 620+(t-1)/2). Interleaved frames are "
          "~0.1 s apart -> appearance nearly identical. This is a MACHINERY "
          "test for revisit structure, NOT a robustness test.")

    ev_orig = load_events(args.out_dir)
    ev_two = remap_two_loop(ev_orig)
    t_max = float(ev_two["t"].max())
    wins = [(0.0, float(LOOP_OFFSET)), (float(LOOP_OFFSET), t_max)]
    print(f"\nstream: {len(ev_two['t'])} events; pseudo-time 0..{t_max:.0f}; "
          f"loop 1 = [0,{LOOP_OFFSET}), loop 2 = [{LOOP_OFFSET},{t_max:.0f}]; "
          f"hd={args.hd_dim}, log class-weighting, seed={args.seed}")

    # single-loop reference build (test 1 comparison)
    t0 = time.time()
    tr_single = build(ev_orig, args.hd_dim, args.seed, args.pos_length_scale,
                      args.time_length_scale, args.grid)
    print(f"built single-loop traceset in {time.time() - t0:.0f}s")
    # two-loop build
    t0 = time.time()
    tr_two = build(ev_two, args.hd_dim, args.seed, args.pos_length_scale,
                   args.time_length_scale, args.grid)
    print(f"built two-loop traceset in {time.time() - t0:.0f}s")

    r1 = test1(tr_two, tr_single, ev_orig)
    del tr_single
    r2 = test2(tr_two, ev_two, wins)
    del tr_two
    r3 = test3(ev_two, wins, args.hd_dim, args.seed, args.pos_length_scale,
               args.time_length_scale, args.grid, what=args.moved_class)
    r4 = test4(args.out_dir, ev_orig, hd=args.content_hd, seed=args.seed,
               pos_l=args.pos_length_scale)

    fig_path = os.path.join(args.out_dir, "two_loop_experiment.png")
    figure(r1, r2, r3, r4, fig_path)
    print(f"\nsaved figure -> {fig_path}")

    print("\n=== SCORED REPORT ===")
    print(f"1 bimodal occupancy:      {'PASS' if r1['pass'] else 'FAIL'} "
          f"({r1['n_pass']}/5 positions show two peaks {SEP_LO}-{SEP_HI} "
          f"apart above the fake-place gate; single-loop unimodal)")
    print(f"2 cross-loop consistency: {'PASS' if r2['pass'] else 'FAIL'} "
          f"({r2['n_small']}/6 classes <0.5 m, outliers "
          f"{r2['outliers'] or 'none'} all below gate; no gated class >1 m)")
    print(f"3 moved object:           {'PASS' if r3['pass'] else 'FAIL'} "
          f"(rank {r3['rank']}, measured {r3['measured']:.2f} m vs injected "
          f"{r3['inj']:.2f} m)")
    print(f"4 loop-closure recall:    {'PASS' if r4['pass'] else 'FAIL'} "
          f"(median {r4['stats'][1]:.2f} m, "
          f"{100 * r4['stats'][4]:.0f}% within 1 m; shuffled null median "
          f"{r4['stats_null'][1]:.2f} m)")
    n_ok = sum(r["pass"] for r in (r1, r2, r3, r4))
    print(f"overall: {n_ok}/4 tests pass; total runtime "
          f"{time.time() - t_start:.0f}s")
    print("(reminder: pseudo-loops share appearance — machinery verified, "
          "robustness untested until real multi-loop data)")


if __name__ == "__main__":
    main()
