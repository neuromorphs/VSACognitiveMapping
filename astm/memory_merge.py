"""Merge by addition: k sessions (or k robots) into ONE fixed-size memory.

This is the capability that has no counterpart in the systems ASTM is compared
against, and it is not a matter of degree — it is structural. Bundling is
commutative and associative, so two memories built over the SAME bases and
symbol codebook satisfy

    M_A + M_B  ==  M_{A union B}          (exactly, to floating point)

Merging therefore requires no data association, no instance re-identification,
no graph alignment, and — the part that matters for the memory argument — no
growth: k sessions merged still occupy D dimensions.

Contrast, for the same k sessions:
  * object/scene graph  — must solve correspondence between two object sets
    before it can merge at all, and the merged graph grows with objects;
  * kNN / vector store   — concatenates (k x bytes), then needs dedup;
  * exact event table    — concatenates (k x bytes).

What this file measures
-----------------------
 1. EXACTNESS      split the event stream in two, build each half separately,
                   add them, and compare against the memory built over the
                   union. Reports max |difference| per trace.
 2. QUERY EQUIVALENCE  the merged memory must answer identically to the union
                   memory (same decoded answers, same similarities).
 3. k-WAY + FOOTPRINT  merge k in {2,4,8,16} parts; error vs k, and bytes of
                   ASTM (flat) against baselines that grow with k.
 4. FRAME RECOVERY  the realistic multi-robot case: robot B's map is expressed
                   in a frame offset by an unknown delta. Because
                   S(x + delta) = S(x) (x) S(delta), the whole of B's memory is
                   B's true memory bound by S(delta) — so delta is recovered by
                   ONE correlation sweep between the two map vectors, then
                   unbound before adding. No point cloud, no ICP, no landmarks.

A constraint this measurement exposes: exact mergeability holds for
confidence weighting, but NOT for the class-frequency weightings
(``balanced``/``log``), whose per-event weight depends on global class counts
that differ between a part and the union. Robots must either agree the
weighting up front or merge on raw confidence. Reported, not hidden.

Run:
    python vsa_cognitive_mapping/memory_merge.py
    python vsa_cognitive_mapping/memory_merge.py --hd-dim 8192 --grid 48
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vsa_cognitive_mapping.astm_traces import (  # noqa: E402
    DEFAULT_OUT, ExactEventTable, QueryRouter, TraceSet, load_events,
)

TRACES = ("what_where", "what_when", "where_when", "event")


# --------------------------------------------------------------------------
# accumulation without rebuilding decoder grids
# --------------------------------------------------------------------------

def accumulate(tr, ev, idx, dx=0.0, dy=0.0):
    """Unnormalised traces + weight mass over ev[idx], using tr's encoders.

    dx/dy shift every position — used to simulate a second robot whose map is
    expressed in an offset frame."""
    M = {k: np.zeros(tr.hd, np.complex128) for k in TRACES}
    wsum = 0.0
    for i in idx:
        c = ev["class"][i]
        S = tr.enc.ctx_pos(float(ev["x"][i]) + dx, float(ev["y"][i]) + dy).values
        T = tr.ctx_time_vec(float(ev["t"][i]))
        C = tr.C[c]
        w = float(ev["conf"][i])
        M["what_where"] += w * C * S
        M["what_when"] += w * C * T
        M["where_when"] += w * S * T
        M["event"] += w * C * S * T
        wsum += w
    return M, wsum


def merge(parts):
    """Add k unnormalised memories and normalise once — the whole operation."""
    out = {k: np.zeros_like(parts[0][0][k]) for k in TRACES}
    tot = 0.0
    for M, w in parts:
        for k in TRACES:
            out[k] += M[k]
        tot += w
    return {k: v / max(tot, 1e-9) for k, v in out.items()}, tot


def install(tr, M):
    tr.M = {k: v.copy() for k, v in M.items()}
    return tr


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--hd-dim", type=int, default=8192)
    ap.add_argument("--grid", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--offset", type=float, nargs=2, default=[1.40, -0.90],
                    metavar=("DX", "DY"), help="frame offset for test 4")
    args = ap.parse_args()

    ev = load_events(args.out_dir)
    n = len(ev["t"])
    classes = sorted(set(ev["class"]))
    bounds = (ev["x"].min(), ev["x"].max(), ev["y"].min(), ev["y"].max())
    print(f"events {n} | classes {len(classes)} | hd {args.hd_dim} | grid {args.grid}")

    t0 = time.perf_counter()
    tr = TraceSet(args.hd_dim, args.seed, 0.75, 20.0, classes, bounds,
                  ev["t"].max(), grid=args.grid)
    print(f"reference TraceSet built in {time.perf_counter()-t0:.0f}s "
          f"(encoders + decoders shared by every condition below)")

    # --- split: two "robots", interleaved so both see overlapping places -----
    idx = np.arange(n)
    A, B = idx[0::2], idx[1::2]

    # ---------------- 1. exactness ------------------------------------------
    print("\n=== 1. exactness: M_A + M_B  vs  M_union ===")
    MA, wA = accumulate(tr, ev, A)
    MB, wB = accumulate(tr, ev, B)
    MU, wU = accumulate(tr, ev, idx)
    merged, wM = merge([(MA, wA), (MB, wB)])
    union = {k: v / max(wU, 1e-9) for k, v in MU.items()}
    print(f"  |A|={len(A)} (w {wA:.1f})  |B|={len(B)} (w {wB:.1f})  "
          f"merged w {wM:.1f}  union w {wU:.1f}")
    worst = 0.0
    for k in TRACES:
        d = float(np.max(np.abs(merged[k] - union[k])))
        rel = d / max(float(np.max(np.abs(union[k]))), 1e-12)
        worst = max(worst, rel)
        print(f"  {k:12s} max|diff| {d:.3e}   relative {rel:.3e}")
    print(f"  -> {'EXACT (float roundoff only)' if worst < 1e-12 else 'NOT exact'}")

    # ---------------- 2. query equivalence ----------------------------------
    print("\n=== 2. query equivalence: merged vs union ===")
    table = ExactEventTable(ev)
    freq = {}
    for c, cf in zip(ev["class"], ev["conf"]):
        freq[c] = freq.get(c, 0.0) + cf
    top = sorted(freq, key=freq.get, reverse=True)[:4]
    t_mid = float(ev["t"].max() * 0.5)

    def battery(M):
        install(tr, M)
        r = QueryRouter(tr)
        out = []
        for c in top:
            out.append(("where(%s)" % c, r.query("where", what=c)))
            out.append(("when(%s)" % c, r.query("when", what=c)))
            out.append(("where(%s,t=%.0f)" % (c, t_mid),
                        r.query("where", what=c, when=t_mid)))
        return out

    bm, bu = battery(merged), battery(union)
    agree, dsim = 0, 0.0
    for (nm, (am, im)), (_, (au, iu)) in zip(bm, bu):
        same = (np.allclose(am, au, atol=1e-9) if isinstance(am, tuple)
                else abs(float(am) - float(au)) < 1e-9)
        agree += bool(same)
        dsim = max(dsim, abs(float(im["sim"]) - float(iu["sim"])))
        if not same:
            print(f"  DIFFER {nm}: merged {am} vs union {au}")
    print(f"  {agree}/{len(bm)} queries identical  (max |sim diff| {dsim:.2e})")

    # ---------------- 3. k-way merge + footprint ----------------------------
    print("\n=== 3. k-way merge (k sessions -> one fixed vector) ===")
    ks, errs = [], []
    for k in (2, 4, 8, 16):
        parts = [accumulate(tr, ev, idx[j::k]) for j in range(k)]
        mk, _ = merge(parts)
        e = max(float(np.max(np.abs(mk[t] - union[t]))) for t in TRACES)
        ks.append(k); errs.append(e)
        print(f"  k={k:3d}  max|diff vs union| {e:.3e}   ASTM bytes "
              f"{4*args.hd_dim*16/1024:.0f} KB (unchanged)")
    ev_bytes = 40.0  # PQ-kNN, ~40 B per stored observation (per stress-test)
    print(f"  for comparison, a kNN/vector store over the same k sessions grows "
          f"linearly: {n*ev_bytes/1024:.0f} KB at k=1 -> {n*ev_bytes/1024:.0f} KB "
          f"per session added (merge = concatenate)")

    # ---------------- 4. frame recovery -------------------------------------
    dx, dy = args.offset
    print(f"\n=== 4. offset-frame merge: robot B's map shifted by "
          f"({dx:+.2f}, {dy:+.2f}) m, delta unknown ===")
    MB_off, wB_off = accumulate(tr, ev, B, dx=dx, dy=dy)
    # S(x+d) = S(x) (x) S(d)  ->  M_B_off = M_B (x) S(d); recover d by sweeping
    # Sweep is centred on ZERO, not on the answer: the receiving robot does not
    # know the offset. Grid spacing therefore bounds the achievable error.
    span, npts = 2.5, 51
    gx = np.linspace(-span, span, npts)
    gy = np.linspace(-span, span, npts)
    step = float(gx[1] - gx[0])
    a_vec = MA["what_where"] / max(wA, 1e-9)
    b_vec = MB_off["what_where"] / max(wB_off, 1e-9)
    best, bxy = -1e9, (0.0, 0.0)
    surf = np.zeros((len(gy), len(gx)))
    for j, yy in enumerate(gy):
        for i, xx in enumerate(gx):
            Sd = tr.enc.ctx_pos(float(xx), float(yy)).values
            s = float(np.real(np.sum(np.conj(a_vec) * (b_vec / Sd))) / tr.hd)
            surf[j, i] = s
            if s > best:
                best, bxy = s, (float(xx), float(yy))
    err = float(np.hypot(bxy[0] - dx, bxy[1] - dy))
    print(f"  swept {npts}x{npts} offsets over +/-{span} m centred on ZERO "
          f"(grid step {step:.3f} m)")
    print(f"  recovered delta ({bxy[0]:+.3f}, {bxy[1]:+.3f}) m  "
          f"vs true ({dx:+.2f}, {dy:+.2f})  error {err:.3f} m "
          f"({err/step:.2f} grid cells)  peak corr {best:+.4f}")
    Sd = tr.enc.ctx_pos(*bxy).values
    MB_alu = {k: (MB_off[k] / Sd if k in ("what_where",) else MB_off[k]) for k in TRACES}
    realigned, _ = merge([(MA, wA), (MB_alu, wB_off)])
    d_al = float(np.max(np.abs(realigned["what_where"] - union["what_where"])))
    print(f"  after unbinding the recovered offset, merged what_where matches "
          f"the union to {d_al:.3e}")
    print("  (no ICP, no landmarks, no point cloud — one correlation sweep and "
          "one unbind)")

    # ---------------- figure ------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    axs[0].semilogy(ks, np.maximum(errs, 1e-18), "o-", color="#0d7d88")
    axs[0].set_xlabel("k sessions merged"); axs[0].set_ylabel("max |diff| vs union")
    axs[0].set_title("k-way merge stays exact")
    axs[0].grid(alpha=.3)

    astm = [4 * args.hd_dim * 16 / 1024] * len(ks)
    knn = [n * ev_bytes / 1024 * k for k in ks]
    axs[1].plot(ks, astm, "o-", label="ASTM traces (fixed)", color="#0d7d88")
    axs[1].plot(ks, knn, "s--", label="kNN store (concatenate)", color="#b9772a")
    axs[1].set_yscale("log"); axs[1].set_xlabel("k sessions merged")
    axs[1].set_ylabel("KB"); axs[1].set_title("footprint after merge")
    axs[1].legend(fontsize=8); axs[1].grid(alpha=.3)

    im = axs[2].imshow(surf, origin="lower", cmap="coolwarm",
                       extent=[gx[0], gx[-1], gy[0], gy[-1]], aspect="equal")
    axs[2].scatter([dx], [dy], s=150, facecolor="none", edgecolor="k",
                   linewidth=2, label="true offset")
    axs[2].scatter([bxy[0]], [bxy[1]], marker="x", s=120, color="k",
                   linewidth=2.2, label="recovered")
    axs[2].set_title("frame offset recovered by correlation")
    axs[2].set_xlabel("dx (m)"); axs[2].set_ylabel("dy (m)"); axs[2].legend(fontsize=8)
    fig.colorbar(im, ax=axs[2], fraction=.046)
    path = os.path.join(args.out_dir, "memory_merge.png")
    fig.savefig(path, dpi=140)
    print(f"\nsaved {path}")
    print("\nNOTE: exact mergeability holds for confidence weighting. The "
          "class-frequency weightings (balanced/log) depend on global class "
          "counts, which differ per part — robots must agree the weighting or "
          "merge on raw confidence.")


if __name__ == "__main__":
    main()
