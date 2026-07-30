"""ASTM D x N sweep: accuracy / latency / memory / calibration per cell.

First rung of the ASTM paper's evaluation matrix (tracker:
``wiki/analysis/2026-07-29-vsa-query-layer-paper-plan.md``, "Common
evaluation protocol"): sweep hyperdimension D and bundled event count N,
run a FIXED query battery per cell against the cell's own exact event
table, and emit the accuracy-latency-bytes frontier evidence.

Design notes (honesty rules):
  * N here = number of events BUNDLED into the traces. Ground truth for a
    cell is the exact table built from the SAME subsampled events, so each
    cell is self-consistent (accuracy is not deflated by events the trace
    never saw).
  * Subsampling = uniform random subset of the event stream per seed
    (class proportions kept natural, no stratification).
  * The decoder grids and time axis are built ONCE per (D, seed) from the
    FULL event stream's bounds / t_max and reused across N cells (constant
    per D; the class codebook + C_mat decoder are rebuilt per cell from the
    subsample's own classes). Decode grid is 48 (not the interactive 72)
    to keep D=16384 tractable.
  * Every planned cell appears in the cells CSV: status ok / skipped /
    failed with a reason. No silent failures.
  * Seed variance is reported as min-max, not just mean.

Correctness thresholds are the SAME as astm_traces bench scoring:
POS_OK_M = 1.0 m, TIME_OK_FR = 40 frames, class exact-match.

Usage (from repo root; requires outputs/classroom/events.csv):

    python vsa_cognitive_mapping/astm_sweep.py
    python vsa_cognitive_mapping/astm_sweep.py --dims 1024,4096 --seeds 0

Outputs -> outputs/classroom/sweep/:
    astm_sweep.csv            one row per cell x query
    astm_sweep_cells.csv      one row per cell (summary + skips/failures)
    fig_accuracy_vs_D.png     lines per N, seed-mean with min-max band
    fig_accuracy_vs_N.png     lines per D
    fig_pareto.png            total map-state MB vs battery accuracy
    fig_calibration_vs_D.png  confident-correct rate + abstention precision
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vsa_cognitive_mapping.vsa import Phasor  # noqa: E402
from vsa_cognitive_mapping.astm_traces import (  # noqa: E402
    ExactEventTable, QueryRouter, TraceSet, load_events,
    POS_OK_M, TIME_OK_FR, _class_seed_mixed,
)

DEFAULT_OUT = os.path.join("outputs", "classroom")


# ==========================================================================
# subsampling
# ==========================================================================

def subsample_events(ev, n, seed):
    """Uniform random subset of n events (no stratification), sorted by
    original order. n >= total -> the full stream unchanged."""
    total = len(ev["t"])
    if n >= total:
        return ev, total
    rng = np.random.RandomState(10_000 * (seed + 1) + n)
    idx = np.sort(rng.choice(total, size=n, replace=False))
    sub = {"class": [ev["class"][i] for i in idx]}
    for k in ("x", "y", "t", "conf"):
        sub[k] = ev[k][idx]
    return sub, n


# ==========================================================================
# shell reuse: one TraceSet per (D, seed); traces + codebook reset per cell
# ==========================================================================

def make_shell(hd_dim, seed, pos_l, time_l, classes_full, bounds_full,
               t_max_full, grid, class_seed_mix=None):
    """TraceSet with full-stream bounds/t_max so decoder grids are built
    once per (D, seed) and are identical across N cells.

    class_seed_mix (FIX 4): mixed into every class-atom seed so replicate
    seeds draw independent class codebooks — previously the sweep seed
    reseeded position/time bases and the subsample but NOT the class atoms,
    so seed variance understated codebook-draw variance."""
    return TraceSet(hd_dim, seed, pos_l, time_l, classes_full, bounds_full,
                    t_max_full, grid=grid, class_seed_mix=class_seed_mix)


def fill_shell(tr, ev_sub, n_null):
    """Reset the shell's per-cell state (codebook, C_mat decoder, traces,
    calibration) and bundle the subsampled events. Decoder grids / time
    axis / encoders are untouched (reused). Class atoms honour the shell's
    class_seed_mix (FIX 4)."""
    tr.classes = sorted(set(ev_sub["class"]))
    mix = getattr(tr, "class_seed_mix", None)
    tr.C = {c: Phasor(dim=tr.hd, seed=_class_seed_mixed(c, mix)).values
            for c in tr.classes}
    tr.C_mat = np.stack([tr.C[c] for c in tr.classes]).astype(np.complex64)
    tr.M = {k: np.zeros(tr.hd, np.complex128)
            for k in ("what_where", "what_when", "where_when", "event")}
    tr.wsum = 0.0
    tr.n_events = 0
    tr.null_calib = None
    for i in range(len(ev_sub["t"])):
        tr.add_event(ev_sub["class"][i], ev_sub["x"][i], ev_sub["y"][i],
                     ev_sub["t"][i], ev_sub["conf"][i])
    tr.finalize()
    t0 = time.perf_counter()
    if n_null:
        tr.calibrate(ev_sub, n_null=n_null)
    return time.perf_counter() - t0


def shell_bytes_estimate(hd_dim, grid, bounds, t_max, n_classes):
    """Pre-flight estimate of one shell's big arrays (complex64 decoders +
    complex128 traces/bases), for the skip-if-too-big guard."""
    pad = 1.0
    ratio = (bounds[3] - bounds[2] + 2 * pad) / (bounds[1] - bounds[0] + 2 * pad)
    n_grid = grid * max(8, int(grid * ratio))
    n_t = int(t_max) + 1
    decoders = (n_grid + n_t + n_classes) * hd_dim * 8
    traces_bases = (4 + 4 + n_classes) * hd_dim * 16
    return decoders + traces_bases


# ==========================================================================
# fixed query battery (same templates as astm_traces._bench_battery)
# ==========================================================================

def sweep_battery(router, table, ev_sub):
    """~16 queries: 3 most-frequent classes x {where all-time, when,
    where@modal-t, where@window} + what@place + what@window + where@t.
    Scored with the astm_traces thresholds. Returns per-query rows."""
    freq = defaultdict(float)
    for c, cf in zip(ev_sub["class"], ev_sub["conf"]):
        freq[c] += cf
    top = sorted(freq, key=freq.get, reverse=True)[:3]
    t_hi = float(ev_sub["t"].max())
    wins = [(0.0, t_hi / 3), (t_hi / 3, 2 * t_hi / 3), (2 * t_hi / 3, t_hi)]

    rows = []

    def run(desc, kind, decode, vsa_args, exact_answer):
        try:
            ans, info = router.query(decode, **vsa_args)
        except Exception as e:  # honesty rule: no silent query failures
            rows.append({"desc": desc, "kind": kind, "decode": decode,
                         "trace": "FAILED", "correct": None,
                         "confident": None, "sim": None, "z": None,
                         "ms_encode": None, "ms_unbind": None,
                         "ms_decode": None, "ms": None,
                         "err": f"QUERY FAILED: {e}"})
            return
        correct = None
        if exact_answer is None:
            err = "n/a (no events)"
        elif kind == "pos":
            e = float(np.hypot(ans[0] - exact_answer[0], ans[1] - exact_answer[1]))
            err, correct = f"{e:.2f} m", bool(e <= POS_OK_M)
        elif kind == "time":
            e = abs(ans - exact_answer)
            err, correct = f"{e:.0f} fr", bool(e <= TIME_OK_FR)
        else:
            correct = ans == exact_answer
            err = "OK" if correct else f"got {ans} vs {exact_answer}"
        rows.append({"desc": desc, "kind": kind, "decode": decode,
                     "trace": info["trace"], "correct": correct,
                     "confident": info["confident"], "sim": info["sim"],
                     "z": info["z"], "ms_encode": info["ms_encode"],
                     "ms_unbind": info["ms_unbind"],
                     "ms_decode": info["ms_decode"], "ms": info["ms_total"],
                     "err": err})

    for i, c in enumerate(top):
        run(f"where({c}) all-time", "pos", "where", dict(what=c),
            table.where(what=c))
        run(f"when({c})", "time", "when", dict(what=c), table.when(what=c))
        tq = table.when(what=c)
        if tq is not None:
            run(f"where({c}, t={tq:.0f})", "pos", "where",
                dict(what=c, when=tq),
                table.where(what=c, when=(tq - 40, tq + 40)))
        w = wins[i % 2]
        run(f"where({c}, t in {w[0]:.0f}:{w[1]:.0f})", "pos", "where",
            dict(what=c, when=w), table.where(what=c, when=w))
    # what@place: modal position of the most frequent class
    p = table.where(what=top[0])
    if p is not None:
        run(f"what(({p[0]:+.1f},{p[1]:+.1f}))", "class", "what",
            dict(where=p), table.what(near_xy=p, radius=0.8))
    # what@window
    run(f"what(t in {wins[0][0]:.0f}:{wins[0][1]:.0f})", "class", "what",
        dict(when=wins[0]), table.what(when=wins[0]))
    # where@t
    tq = float(t_hi * 0.12)
    run(f"where(t={tq:.0f})", "pos", "where", dict(when=tq),
        table.where(when=(tq - 20, tq + 20)))
    return rows


def summarize_rows(rows):
    scored = [r for r in rows if r["correct"] is not None]
    conf = [r for r in scored if r["confident"] is True]
    abst = [r for r in scored if r["confident"] is False]
    ms = [r["ms"] for r in rows if r["ms"] is not None]
    med = lambda key: (float(np.median([r[key] for r in rows
                                        if r[key] is not None]))
                       if any(r[key] is not None for r in rows) else None)
    return {
        "n_queries": len(rows),
        "n_scored": len(scored),
        "n_correct": sum(1 for r in scored if r["correct"]),
        "accuracy": (sum(1 for r in scored if r["correct"]) / len(scored)
                     if scored else None),
        "n_confident": len(conf),
        "n_confident_correct": sum(1 for r in conf if r["correct"]),
        "n_abstained": len(abst),
        "n_abstained_wrong": sum(1 for r in abst if not r["correct"]),
        "med_ms_encode": med("ms_encode"), "med_ms_unbind": med("ms_unbind"),
        "med_ms_decode": med("ms_decode"), "med_ms_total": med("ms"),
        "total_ms_battery": float(np.sum(ms)) if ms else None,
    }


# ==========================================================================
# sweep driver
# ==========================================================================

def run_cell(D, N, seed, ev_full, shell_cache, args, meta):
    """Build (or reuse) the (D, seed) shell, fill it from the (N, seed)
    subsample, calibrate, run the battery. Returns (cell_dict, query_rows)."""
    t_cell = time.perf_counter()
    key = (D, seed)
    timings = {}
    if key not in shell_cache:
        t0 = time.perf_counter()
        mix = None if args.no_class_seed_mix else seed
        shell_cache[key] = make_shell(
            D, seed, args.pos_length_scale, args.time_length_scale,
            meta["classes_full"], meta["bounds_full"], meta["t_max_full"],
            args.grid, class_seed_mix=mix)
        timings["shell_s"] = time.perf_counter() - t0
    else:
        timings["shell_s"] = 0.0
    tr = shell_cache[key]

    ev_sub, n_actual = subsample_events(ev_full, N, seed)
    t0 = time.perf_counter()
    calib_s = fill_shell(tr, ev_sub, args.null_samples)
    timings["build_s"] = time.perf_counter() - t0 - calib_s
    timings["calib_s"] = calib_s

    table = ExactEventTable(ev_sub)
    router = QueryRouter(tr, z_threshold=args.z_threshold)
    t0 = time.perf_counter()
    rows = sweep_battery(router, table, ev_sub)
    timings["battery_s"] = time.perf_counter() - t0

    mem = tr.memory_bytes()
    cell = {"D": D, "N": N, "seed": seed, "status": "ok", "reason": "",
            "n_events": n_actual, "n_classes": len(tr.classes)}
    cell.update(summarize_rows(rows))
    cell.update({"traces_bytes": mem["traces"], "total_bytes": mem["total"],
                 "build_s": round(timings["build_s"], 3),
                 "calib_s": round(timings["calib_s"], 3),
                 "battery_s": round(timings["battery_s"], 3),
                 "shell_s": round(timings["shell_s"], 3),
                 "cell_s": round(time.perf_counter() - t_cell, 3)})
    return cell, rows, timings


CELL_FIELDS = ["D", "N", "seed", "status", "reason", "n_events", "n_classes",
               "n_queries", "n_scored", "n_correct", "accuracy",
               "n_confident", "n_confident_correct", "n_abstained",
               "n_abstained_wrong", "med_ms_encode", "med_ms_unbind",
               "med_ms_decode", "med_ms_total", "total_ms_battery",
               "traces_bytes", "total_bytes", "build_s", "calib_s",
               "battery_s", "shell_s", "cell_s"]

QUERY_FIELDS = ["D", "N", "seed", "query", "kind", "decode", "trace",
                "correct", "confident", "sim", "z", "ms_encode", "ms_unbind",
                "ms_decode", "ms_total", "err"]

CSV_NOTES = [
    "# ASTM D x N sweep (astm_sweep.py).",
    "# N = number of events BUNDLED into the traces (uniform random subsample",
    "#     per seed, natural class proportions). Ground truth per cell is the",
    "#     exact event table built from the SAME subsampled events - each",
    "#     cell is self-consistent.",
    "# Correctness thresholds (same as astm_traces bench): position <= "
    f"{POS_OK_M} m of exact modal answer, time <= {TIME_OK_FR:.0f} frames, "
    "class exact.",
    "# Decoder grids/time-axis are built from the FULL stream bounds/t_max,",
    "# reused across N cells (constant per D); grid=48 for this sweep.",
    "# Every planned cell is present: status ok / skipped / failed + reason.",
]


def write_csvs(sweep_dir, cells, query_rows_all, extra_notes):
    cells_path = os.path.join(sweep_dir, "astm_sweep_cells.csv")
    with open(cells_path, "w", newline="") as f:
        for line in CSV_NOTES + extra_notes:
            f.write(line + "\n")
        w = csv.DictWriter(f, fieldnames=CELL_FIELDS, extrasaction="ignore")
        w.writeheader()
        for c in cells:
            w.writerow(c)
    q_path = os.path.join(sweep_dir, "astm_sweep.csv")
    with open(q_path, "w", newline="") as f:
        for line in CSV_NOTES + extra_notes:
            f.write(line + "\n")
        w = csv.writer(f)
        w.writerow(QUERY_FIELDS)
        for D, N, seed, rows in query_rows_all:
            for r in rows:
                w.writerow([D, N, seed, r["desc"], r["kind"], r["decode"],
                            r["trace"], r["correct"], r["confident"],
                            None if r["sim"] is None else f"{r['sim']:.6f}",
                            None if r["z"] is None else f"{r['z']:.3f}",
                            *(None if r[k] is None else f"{r[k]:.3f}"
                              for k in ("ms_encode", "ms_unbind", "ms_decode",
                                        "ms")),
                            r["err"]])
    return q_path, cells_path


# ==========================================================================
# figures
# ==========================================================================

def _cells_ok(cells):
    return [c for c in cells if c["status"] == "ok" and c["accuracy"] is not None]


def make_figures(sweep_dir, cells, dims, n_list):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok = _cells_ok(cells)
    if not ok:
        print("no successful cells; skipping figures")
        return []
    paths = []

    def agg(D, N):
        accs = [c["accuracy"] for c in ok if c["D"] == D and c["N"] == N]
        if not accs:
            return None
        return float(np.mean(accs)), float(np.min(accs)), float(np.max(accs))

    n_colors = {N: f"C{i}" for i, N in enumerate(n_list)}
    d_colors = {D: f"C{i}" for i, D in enumerate(dims)}

    # -- fig 1: accuracy vs D ------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for N in n_list:
        ds = [D for D in dims if agg(D, N)]
        if not ds:
            continue
        m = [agg(D, N)[0] for D in ds]
        lo = [agg(D, N)[1] for D in ds]
        hi = [agg(D, N)[2] for D in ds]
        ax.plot(ds, m, "o-", color=n_colors[N], label=f"N={N}")
        ax.fill_between(ds, lo, hi, color=n_colors[N], alpha=0.15)
    ax.set_xscale("log", base=2)
    ax.set_xticks(dims)
    ax.set_xticklabels([str(d) for d in dims])
    ax.set_xlabel("hyperdimension D")
    ax.set_ylabel("battery accuracy (scored queries)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("ASTM sweep: accuracy vs D (seed mean, min-max band)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    p = os.path.join(sweep_dir, "fig_accuracy_vs_D.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    paths.append(p)

    # -- fig 2: accuracy vs N ------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for D in dims:
        ns = [N for N in n_list if agg(D, N)]
        if not ns:
            continue
        m = [agg(D, N)[0] for N in ns]
        lo = [agg(D, N)[1] for N in ns]
        hi = [agg(D, N)[2] for N in ns]
        ax.plot(ns, m, "s-", color=d_colors[D], label=f"D={D}")
        ax.fill_between(ns, lo, hi, color=d_colors[D], alpha=0.12)
    ax.set_xlabel("N events bundled")
    ax.set_ylabel("battery accuracy (scored queries)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("ASTM sweep: accuracy vs N (seed mean, min-max band)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    p = os.path.join(sweep_dir, "fig_accuracy_vs_N.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    paths.append(p)

    # -- fig 3: Pareto scatter ----------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    for D in dims:
        for N in n_list:
            sel = [c for c in ok if c["D"] == D and c["N"] == N]
            if not sel:
                continue
            mb = float(np.mean([c["total_bytes"] for c in sel])) / 1e6
            acc = float(np.mean([c["accuracy"] for c in sel]))
            ms = float(np.mean([c["med_ms_total"] for c in sel
                                if c["med_ms_total"] is not None]))
            ax.scatter(mb, acc, s=30 + 18 * ms, color=d_colors[D],
                       alpha=0.75, edgecolors="k", linewidths=0.5)
            ax.annotate(f"{D}/{N}", (mb, acc), fontsize=7,
                        xytext=(4, 4), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("total map-state MB (traces + codebook + bases + decoders)")
    ax.set_ylabel("battery accuracy (seed mean)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("ASTM sweep: accuracy vs memory Pareto "
                 "(point size = median query ms; label D/N)")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    p = os.path.join(sweep_dir, "fig_pareto.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    paths.append(p)

    # -- fig 4: calibration quality vs D ------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True)

    def pooled(D, N=None):
        sel = [c for c in ok if c["D"] == D and (N is None or c["N"] == N)]
        nc = sum(c["n_confident"] for c in sel)
        ncc = sum(c["n_confident_correct"] for c in sel)
        na = sum(c["n_abstained"] for c in sel)
        naw = sum(c["n_abstained_wrong"] for c in sel)
        return (ncc / nc if nc else np.nan, naw / na if na else np.nan,
                nc, na)

    for N in n_list:
        ds = [D for D in dims if any(c["D"] == D and c["N"] == N for c in ok)]
        if not ds:
            continue
        axes[0].plot(ds, [pooled(D, N)[0] for D in ds], "o-",
                     color=n_colors[N], alpha=0.7, label=f"N={N}")
        axes[1].plot(ds, [pooled(D, N)[1] for D in ds], "o-",
                     color=n_colors[N], alpha=0.7, label=f"N={N}")
    ds_all = [D for D in dims if any(c["D"] == D for c in ok)]
    axes[0].plot(ds_all, [pooled(D)[0] for D in ds_all], "k^--", lw=2,
                 label="pooled")
    axes[1].plot(ds_all, [pooled(D)[1] for D in ds_all], "k^--", lw=2,
                 label="pooled")
    for ax, title in zip(axes, ["confident-correct rate\n"
                                "(correct | z >= threshold)",
                                "abstention precision\n"
                                "(wrong | abstained)"]):
        ax.set_xscale("log", base=2)
        ax.set_xticks(dims)
        ax.set_xticklabels([str(d) for d in dims])
        ax.set_xlabel("hyperdimension D")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("ASTM sweep: null-calibrated abstention quality "
                 "(pooled over seeds; small counts per point)", fontsize=11)
    fig.tight_layout()
    p = os.path.join(sweep_dir, "fig_calibration_vs_D.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    paths.append(p)

    return paths


# ==========================================================================
# main
# ==========================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dims", default="1024,2048,4096,8192,16384")
    ap.add_argument("--n-events", default="500,1000,2000,4498")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--grid", type=int, default=48,
                    help="decode grid (reduced from 72 for the sweep)")
    ap.add_argument("--null-samples", type=int, default=100,
                    help="null probes per decode type (reduced for speed)")
    ap.add_argument("--z-threshold", type=float, default=3.0)
    ap.add_argument("--pos-length-scale", type=float, default=0.75)
    ap.add_argument("--time-length-scale", type=float, default=20.0)
    ap.add_argument("--budget-min", type=float, default=40.0,
                    help="runtime budget; grid is trimmed if the profile "
                         "extrapolation exceeds it")
    ap.add_argument("--mem-limit-mb", type=float, default=3000.0,
                    help="skip a cell if its shell estimate exceeds this")
    ap.add_argument("--no-class-seed-mix", action="store_true",
                    help="do NOT mix the cell seed into the class-atom seeds "
                         "(pre-hardening behaviour: replicate seeds shared "
                         "one class codebook, understating seed variance)")
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    args = ap.parse_args()

    dims = [int(v) for v in args.dims.split(",")]
    n_list = [int(v) for v in args.n_events.split(",")]
    seeds = [int(v) for v in args.seeds.split(",")]

    t_start = time.perf_counter()
    ev_full = load_events(args.out_dir)
    total_events = len(ev_full["t"])
    n_list = sorted({min(n, total_events) for n in n_list})
    meta = {
        "classes_full": sorted(set(ev_full["class"])),
        "bounds_full": (ev_full["x"].min(), ev_full["x"].max(),
                        ev_full["y"].min(), ev_full["y"].max()),
        "t_max_full": float(ev_full["t"].max()),
    }
    sweep_dir = os.path.join(args.out_dir, "sweep")
    os.makedirs(sweep_dir, exist_ok=True)
    print(f"event stream: {total_events} events, "
          f"{len(meta['classes_full'])} classes, t_max={meta['t_max_full']:.0f}")
    print(f"planned grid: D={dims} x N={n_list} x seeds={seeds} "
          f"= {len(dims) * len(n_list) * len(seeds)} cells")

    cells, query_rows_all = [], []
    shell_cache = {}
    done = set()

    # ---- profile one mid cell, extrapolate, trim if over budget ----------
    PROF = (4096, 2000, 0)
    prof_D, prof_N, prof_seed = PROF
    notes = []
    if args.no_class_seed_mix:
        mix_note = ("# CLASS-ATOM SEEDING: --no-class-seed-mix (pre-hardening "
                    "behaviour; all seeds share one class codebook).")
    else:
        mix_note = ("# CLASS-ATOM SEEDING (hardening FIX 4, NEW DEFAULT): "
                    "class_seed_mix = cell seed - each replicate draws its "
                    "own class codebook.")
    notes.append(mix_note)
    notes.append("# NULL CALIBRATION (hardening FIX 2, NEW DEFAULT): v2 "
                 "per-(decode, trace, probe-kind) null cells; confident/"
                 "abstained columns are not comparable to pre-hardening runs.")
    print(mix_note.lstrip("# "))
    if prof_D in dims and prof_N in n_list and prof_seed in seeds:
        print(f"\nprofiling cell D={prof_D} N={prof_N} seed={prof_seed} ...")
        cell, rows, tm = run_cell(prof_D, prof_N, prof_seed, ev_full,
                                  shell_cache, args, meta)
        cells.append(cell)
        query_rows_all.append((prof_D, prof_N, prof_seed, rows))
        done.add(PROF)
        print(f"  shell {tm['shell_s']:.1f}s  build {tm['build_s']:.1f}s  "
              f"calib {tm['calib_s']:.1f}s  battery {tm['battery_s']:.1f}s  "
              f"acc {cell['accuracy']:.2f}")

        def est_total(ds, ss):
            tot = 0.0
            for D in ds:
                for s in ss:
                    tot += tm["shell_s"] * D / prof_D  # per (D, seed) shell
                    for N in n_list:
                        if (D, N, s) in done:
                            continue
                        tot += (tm["build_s"] * (N * D) / (prof_N * prof_D)
                                + (tm["calib_s"] + tm["battery_s"]) * D / prof_D)
            return tot

        budget_s = args.budget_min * 60 * 0.85  # safety margin
        est = est_total(dims, seeds)
        print(f"  extrapolated remaining: {est/60:.1f} min "
              f"(budget {args.budget_min:.0f} min)")
        if est > budget_s and len(seeds) > 2:
            seeds = seeds[:2]
            est = est_total(dims, seeds)
            notes.append(f"# BUDGET TRIM: seeds reduced to {seeds} "
                         f"(extrapolation exceeded {args.budget_min:.0f} min).")
            print(f"  trimmed seeds -> {seeds}; new estimate {est/60:.1f} min")
        if est > budget_s and 16384 in dims:
            dims = [d for d in dims if d != 16384]
            notes.append("# BUDGET TRIM: D=16384 dropped "
                         f"(extrapolation exceeded {args.budget_min:.0f} min).")
            print(f"  dropped D=16384; new estimate "
                  f"{est_total(dims, seeds)/60:.1f} min")

    # ---- main loop --------------------------------------------------------
    for D in dims:
        for seed in seeds:
            for N in n_list:
                if (D, N, seed) in done:
                    continue
                est_mb = shell_bytes_estimate(D, args.grid,
                                              meta["bounds_full"],
                                              meta["t_max_full"],
                                              len(meta["classes_full"])) / 1e6
                if est_mb > args.mem_limit_mb:
                    cells.append({"D": D, "N": N, "seed": seed,
                                  "status": "skipped",
                                  "reason": f"shell estimate {est_mb:.0f} MB "
                                            f"> limit {args.mem_limit_mb:.0f} MB"})
                    print(f"[D={D:5d} N={N:4d} s={seed}] SKIPPED "
                          f"({est_mb:.0f} MB > limit)")
                    continue
                try:
                    cell, rows, _ = run_cell(D, N, seed, ev_full,
                                             shell_cache, args, meta)
                    cells.append(cell)
                    query_rows_all.append((D, N, seed, rows))
                    print(f"[D={D:5d} N={N:4d} s={seed}] acc "
                          f"{cell['accuracy']:.2f} "
                          f"({cell['n_correct']}/{cell['n_scored']})  "
                          f"conf {cell['n_confident_correct']}/"
                          f"{cell['n_confident']}  "
                          f"med {cell['med_ms_total']:.1f} ms  "
                          f"{cell['total_bytes']/1e6:.1f} MB  "
                          f"cell {cell['cell_s']:.1f}s")
                except Exception as e:  # honesty rule: log, don't hide
                    cells.append({"D": D, "N": N, "seed": seed,
                                  "status": "failed", "reason": str(e)})
                    print(f"[D={D:5d} N={N:4d} s={seed}] FAILED: {e}")
                done.add((D, N, seed))
            shell_cache.pop((D, seed), None)  # bound peak memory

    # ---- outputs ----------------------------------------------------------
    notes.append(f"# Executed grid: D={dims} x N={n_list} x seeds={seeds}; "
                 f"total wall time {(time.perf_counter()-t_start)/60:.1f} min.")
    q_path, cells_path = write_csvs(sweep_dir, cells, query_rows_all, notes)
    fig_paths = make_figures(sweep_dir, cells, dims, n_list)

    ok = _cells_ok(cells)
    n_fail = sum(1 for c in cells if c["status"] != "ok")
    print(f"\nsweep done in {(time.perf_counter()-t_start)/60:.1f} min: "
          f"{len(ok)} ok cells, {n_fail} skipped/failed")
    print(f"csv:     {q_path}\n         {cells_path}")
    for p in fig_paths:
        print(f"figure:  {p}")

    # compact seed-mean accuracy matrix for the console
    print("\nseed-mean accuracy (min..max):")
    hdr = "D\\N   " + "".join(f"{N:>16d}" for N in n_list)
    print(hdr)
    for D in dims:
        parts = [f"{D:<6d}"]
        for N in n_list:
            accs = [c["accuracy"] for c in ok if c["D"] == D and c["N"] == N]
            parts.append(f"{np.mean(accs):.2f} "
                         f"({np.min(accs):.2f}..{np.max(accs):.2f})"
                         if accs else "-".rjust(16))
        print("  ".join(parts))


if __name__ == "__main__":
    main()
