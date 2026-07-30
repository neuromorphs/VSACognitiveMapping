"""Bounded-state economics: WHERE does the ASTM map state actually beat an
exact event table? (the honest crossover computation the reviewers asked for)

Model + measurement:

  EXACT TABLE (baseline A)
    * storage grows O(N): measured bytes/event from the real events.csv
      (csv on disk) plus a packed in-memory figure (4 float64 columns +
      2-byte class id), both computed by code;
    * linear-scan query: boolean mask over all N rows (measured ms at
      N=4498, extrapolated O(N)); the full modal answer (KDE over matches)
      is also timed for context;
    * indexed query: O(log N) binary search over a sorted column (measured
      np.searchsorted as the proxy) — effectively free at any N that fits
      in memory. An indexed table therefore NEVER loses on latency; the
      honest latency claim is only vs a scan.

  ASTM (fixed-dimensional associative map state)
    * traces are CONSTANT in N: 4 * D * 16 B  (complex128);
    * the rest of the state scales with vocabulary V and decoder grid
      resolution, NOT with N (component formulas printed; measured via
      TraceSet.memory_bytes on the stored D=8192 state and an in-memory
      D=4096 rebuild):
          codebook       V * D * 16 B
          bases          4 * D * 16 B  (+ extra multi-scale time bases)
          decoder grids  (G_cells + T_steps + V) * D * 8 B   (complex64)
    * query latency ~constant in N (state size fixed): measured median over
      a mixed query set (plan entry j: 2.7 ms at D=4096).

  Crossovers reported (D = 4096 and 8192):
    * bytes: N* where table bytes exceed (a) traces-only, (b) full map
      state (traces + codebook + bases + decoder grids);
    * latency: N* where the linear-scan table query exceeds the ~constant
      ASTM query (and the honest note that an INDEXED table never crosses).

Figure -> outputs/classroom/crossover_analysis.png.

Imports from astm_traces are read-only and restricted to the stable names
(TraceSet, QueryRouter, ExactEventTable, load_events); the import is retried
once after 60 s if it fails mid-edit (another agent owns that file).

Usage (from repo root, needs outputs/classroom/events.csv + astm_traces.pt):
    python vsa_cognitive_mapping/crossover_analysis.py
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _import_astm():
    try:
        from vsa_cognitive_mapping.astm_traces import (
            TraceSet, QueryRouter, ExactEventTable, load_events)
    except Exception as e:  # file may be mid-edit by another agent
        print(f"astm_traces import failed ({e!r}); retrying in 60 s ...")
        time.sleep(60)
        from vsa_cognitive_mapping.astm_traces import (
            TraceSet, QueryRouter, ExactEventTable, load_events)
    return TraceSet, QueryRouter, ExactEventTable, load_events


TraceSet, QueryRouter, ExactEventTable, load_events = _import_astm()

DEFAULT_OUT = os.path.join("outputs", "classroom")


# --------------------------------------------------------------------------
# exact-table side (measured)
# --------------------------------------------------------------------------

def table_bytes_per_event(out_dir, ev, events_name="events.csv"):
    n = len(ev["t"])
    csv_b = os.path.getsize(os.path.join(out_dir, events_name)) / n
    # packed in-memory row: x, y, t, conf float64 + uint16 class id
    packed_b = 4 * 8 + 2
    return csv_b, packed_b, n


def time_table_queries(ev, table, reps=200):
    """Median ms for (a) the O(N) boolean-mask linear scan, (b) the full
    modal where() answer, (c) an indexed lookup proxy (searchsorted)."""
    cls_arr = np.array(ev["class"])
    t_arr = ev["t"]
    a, b = float(t_arr.max()) / 3, 2 * float(t_arr.max()) / 3

    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        m = (cls_arr == "chair") & (t_arr >= a) & (t_arr <= b)
        _ = ev["x"][m]
        ts.append((time.perf_counter() - t0) * 1e3)
    scan_ms = float(np.median(ts))

    ts = []
    for _ in range(max(reps // 10, 10)):
        t0 = time.perf_counter()
        table.where(what="chair", when=(a, b))
        ts.append((time.perf_counter() - t0) * 1e3)
    modal_ms = float(np.median(ts))

    t_sorted = np.sort(t_arr)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        np.searchsorted(t_sorted, a)
        np.searchsorted(t_sorted, b)
        ts.append((time.perf_counter() - t0) * 1e3)
    idx_ms = float(np.median(ts))
    return scan_ms, modal_ms, idx_ms


# --------------------------------------------------------------------------
# ASTM side (measured + component formulas)
# --------------------------------------------------------------------------

def astm_components(tr):
    """memory_bytes() split, plus (V, grid cells, time steps) for formulas."""
    mem = tr.memory_bytes()
    V = len(tr.classes)
    g_cells = tr.G_flat.shape[0]
    t_steps = tr.T_axis.shape[0]
    return mem, V, g_cells, t_steps


def build_traceset_at(ev, hd, seed=0, pos_l=0.75, time_l=20.0, grid=72):
    """In-memory TraceSet at a given D from the events (stable API only;
    no null calibration needed for byte/latency accounting)."""
    classes = sorted(set(ev["class"]))
    bounds = (ev["x"].min(), ev["x"].max(), ev["y"].min(), ev["y"].max())
    tr = TraceSet(hd, seed, pos_l, time_l, classes, bounds, ev["t"].max(),
                  grid=grid)
    for i in range(len(ev["t"])):
        tr.add_event(ev["class"][i], ev["x"][i], ev["y"][i], ev["t"][i],
                     ev["conf"][i])
    tr.finalize()
    return tr


def time_astm_queries(tr, ev, reps=30):
    """Median ms over a mixed routed-query set (the bench battery's shapes)."""
    router = QueryRouter(tr)
    t_hi = float(ev["t"].max())
    classes = [c for c in ("chair", "person", "tv") if c in tr.classes] \
        or tr.classes[:3]
    times = []
    for _ in range(reps):
        for c in classes:
            for q in (dict(decode="where", what=c),
                      dict(decode="when", what=c),
                      dict(decode="where", what=c, when=(0.0, t_hi / 3))):
                _, info = router.query(**q)
                times.append(info["ms_total"])
        _, info = router.query("what", where=(0.0, 0.0), when=t_hi / 2)
        times.append(info["ms_total"])
    return float(np.median(times)), float(np.percentile(times, 90))


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--traces-name", default="astm_traces.pt")
    ap.add_argument("--grid", type=int, default=72)
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()

    ev = load_events(args.out_dir)
    table = ExactEventTable(ev)
    csv_b, packed_b, n_ev = table_bytes_per_event(args.out_dir, ev)
    print(f"exact table: N={n_ev} events; measured {csv_b:.1f} B/event (csv "
          f"on disk), {packed_b} B/event packed in-memory "
          f"(4 float64 + uint16 class id)")

    scan_ms, modal_ms, idx_ms = time_table_queries(ev, table)
    scan_ms_per_ev = scan_ms / n_ev
    print(f"exact-table latency at N={n_ev}: linear-scan mask "
          f"{scan_ms:.4f} ms (-> {1e6 * scan_ms_per_ev:.1f} ns/event, "
          f"extrapolated O(N)); full modal answer {modal_ms:.2f} ms "
          f"(KDE over matches, O(M^2)); indexed searchsorted {idx_ms:.4f} ms "
          f"(O(log N) — effectively free at any N)")

    # ---- ASTM states at D=8192 (stored) and D=4096 (rebuilt) --------------
    states = {}
    path = os.path.join(args.out_dir, args.traces_name)
    tr8 = TraceSet.load(path, grid=args.grid)
    states[tr8.hd] = tr8
    print(f"\nloaded {args.traces_name}: D={tr8.hd}, {tr8.n_events} events, "
          f"V={len(tr8.classes)} classes")
    t0 = time.perf_counter()
    tr4 = build_traceset_at(ev, 4096, seed=tr8.seed, pos_l=tr8.enc.pos_l,
                            time_l=tr8.enc.time_l, grid=args.grid)
    print(f"rebuilt D=4096 state in-memory in {time.perf_counter() - t0:.0f}s")
    states[4096] = tr4

    astm = {}
    for D in sorted(states):
        tr = states[D]
        mem, V, g_cells, t_steps = astm_components(tr)
        q_med, q_p90 = time_astm_queries(tr, ev)
        astm[D] = {"mem": mem, "V": V, "g": g_cells, "t": t_steps,
                   "q_med": q_med, "q_p90": q_p90}
        print(f"\nASTM D={D}: query median {q_med:.2f} ms (p90 {q_p90:.2f}) — "
              f"constant in N (state size fixed)")
        print(f"  traces        {mem['traces']:>12,} B   = 4 * D * 16")
        print(f"  codebook      {mem['class_codebook']:>12,} B   = V * D * 16"
              f"  (V={V})")
        print(f"  bases         {mem['bases']:>12,} B   = 4 * D * 16")
        print(f"  decoder grids {mem['decoder_grids']:>12,} B   = "
              f"(G + T + V) * D * 8  (G={g_cells} cells, T={t_steps} steps)")
        print(f"  TOTAL         {mem['total']:>12,} B "
              f"({mem['total'] / 1e6:.1f} MB)")

    # ---- crossovers -------------------------------------------------------
    print("\n---- crossover event-counts (measured csv bytes/event "
          f"{csv_b:.1f} B) ----")
    crossings = {}
    for D in sorted(astm):
        mem = astm[D]["mem"]
        n_traces = mem["traces"] / csv_b
        n_total = mem["total"] / csv_b
        n_lat = astm[D]["q_med"] / scan_ms_per_ev
        crossings[D] = {"n_traces": n_traces, "n_total": n_total,
                        "n_lat": n_lat}
        print(f"  D={D}:")
        print(f"    bytes vs traces-only ({mem['traces'] / 1024:.0f} KB): "
              f"table exceeds at N* = {n_traces:,.0f} events "
              f"({'ALREADY PASSED' if n_ev > n_traces else 'not yet'} "
              f"at N={n_ev})")
        print(f"    bytes vs full map state ({mem['total'] / 1e6:.1f} MB): "
              f"table exceeds at N* = {n_total:,.0f} events "
              f"(= {n_total / n_ev:,.0f}x the demonstrated stream)")
        print(f"    latency vs linear scan: ASTM {astm[D]['q_med']:.2f} ms "
              f"constant; scan reaches that at N* = {n_lat:,.0f} events")
    print("    latency vs INDEXED table: never crosses — searchsorted is "
          f"{idx_ms * 1e3:.1f} us at N={n_ev} and O(log N).")

    print("\nHONEST SENTENCES:")
    for D in sorted(astm):
        c = crossings[D]
        print(f"  At D={D}: the traces alone are smaller than the event csv "
              f"beyond ~{c['n_traces'] / 1e3:.1f}k events, but the FULL map "
              f"state (incl. codebook/bases/decoder grids) only wins on "
              f"bytes beyond ~{c['n_total'] / 1e6:.1f}M events, and on "
              f"latency only vs an unindexed linear scan beyond "
              f"~{c['n_lat'] / 1e6:.1f}M events; an indexed exact table is "
              f"never beaten on latency at feasible N.")

    # ---- figure -----------------------------------------------------------
    if not args.no_figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        Ns = np.logspace(2, 8, 200)
        fig, axs = plt.subplots(1, 2, figsize=(13, 5.2),
                                constrained_layout=True)

        ax = axs[0]
        ax.plot(Ns, Ns * csv_b, "k-", lw=1.8,
                label=f"exact table, csv ({csv_b:.0f} B/event)")
        ax.plot(Ns, Ns * packed_b, "k--", lw=1.2,
                label=f"exact table, packed ({packed_b} B/event)")
        for D, color in ((4096, "#0d7d88"), (8192, "#b9772a")):
            mem = astm[D]["mem"]
            ax.axhline(mem["traces"], color=color, ls=":", lw=1.4,
                       label=f"ASTM D={D} traces only "
                             f"({mem['traces'] / 1024:.0f} KB)")
            ax.axhline(mem["total"], color=color, ls="-", lw=1.4,
                       label=f"ASTM D={D} full state "
                             f"({mem['total'] / 1e6:.0f} MB)")
            for key, ls in (("n_traces", ":"), ("n_total", "-")):
                nstar = crossings[D][key]
                ystar = nstar * csv_b
                ax.plot([nstar], [ystar], "o", color=color, ms=6)
                ax.annotate(f"{nstar:,.0f}", (nstar, ystar), fontsize=7,
                            textcoords="offset points", xytext=(4, -10),
                            color=color)
        ax.axvline(n_ev, color="0.5", lw=0.8)
        ax.text(n_ev, ax.get_ylim()[0] * 2, f" demonstrated N={n_ev}",
                fontsize=7, color="0.4", rotation=90, va="bottom")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("N events"); ax.set_ylabel("bytes")
        ax.set_title("(a) storage crossover: O(N) table vs constant map state",
                     fontsize=10)
        ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7)

        ax = axs[1]
        ax.plot(Ns, Ns * scan_ms_per_ev, "k-", lw=1.8,
                label=f"linear scan ({1e6 * scan_ms_per_ev:.0f} ns/event, "
                      f"measured at N={n_ev})")
        ax.axhline(idx_ms, color="k", ls="--", lw=1.2,
                   label=f"indexed table (searchsorted, "
                         f"{idx_ms * 1e3:.1f} us, O(log N))")
        for D, color in ((4096, "#0d7d88"), (8192, "#b9772a")):
            ax.axhline(astm[D]["q_med"], color=color, ls="-", lw=1.4,
                       label=f"ASTM D={D} query ({astm[D]['q_med']:.1f} ms, "
                             f"constant)")
            nstar = crossings[D]["n_lat"]
            ax.plot([nstar], [astm[D]["q_med"]], "o", color=color, ms=6)
            ax.annotate(f"{nstar:,.0f}", (nstar, astm[D]["q_med"]),
                        fontsize=7, textcoords="offset points",
                        xytext=(4, 4), color=color)
        ax.axvline(n_ev, color="0.5", lw=0.8)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("N events"); ax.set_ylabel("query latency (ms)")
        ax.set_title("(b) latency crossover: scan O(N) vs constant ASTM "
                     "(indexed table never crossed)", fontsize=10)
        ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7)

        fig.suptitle("Bounded-state economics: where the ASTM map state "
                     "actually wins", fontsize=11)
        path = os.path.join(args.out_dir, "crossover_analysis.png")
        fig.savefig(path, dpi=140)
        print(f"saved {path}")


if __name__ == "__main__":
    main()
