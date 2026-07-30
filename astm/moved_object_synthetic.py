"""ASTM moved-object detection on a SYNTHETIC relocation (pending real data).

The classroom walk is a single loop, so place and time are confounded: no
object actually moves. This script deliberately breaks that confound: it
picks a mid-frequency class (default 'tv', >= 30 events required in each
half) and RELOCATES its second-half events (t > t_max/2) by a fixed offset,
keeping every relocated event inside the original walk bounds (the
requested +2.5/+1.5 m offset is auto-shrunk/flipped to the first feasible
candidate if the class's second-half footprint would leave the bounds).
All other events are untouched. A TraceSet is then built in-memory from the
modified stream, and THREE detections are verified (plus the same three on
the UNMODIFIED stream as control):

  (a) ``QueryRouter.which_moved(first-half, second-half)`` ranks the
      relocated class #1 with displacement ~ the injected offset. Ranking
      is gated by a RANGE-MATCHED fake-class null: 60 random unit-phasor
      "classes" that are NOT in the stream are unbound from M_event and
      decoded with the same half-window kernels; min_sim = mean + 3 std of
      that null. (The build-time point-probe null does not transfer to
      half-window range kernels — the kernel is far from unit-modulus, so
      absolute sims shrink for signal and null alike; measured here.)
  (b) conditional queries where(class, t in first half) vs where(class,
      t in second half) decode the two distinct positions;
  (c) the timeless marginal where(class) (M_what_where) cannot: it returns
      one answer for a two-place ground truth — the event trace is what
      makes the distinction possible.

Default hd_dim is 16384 (not the usual 8192): half-window range kernels
concentrate the probe energy into ~2% of the components, and at 8192 the
surviving signal for a ~600-event class sits within ~2.5 std of the
fake-class null floor (measured); 16384 gives the needed headroom.

Figure -> outputs/classroom/moved_object_synthetic.png: control and
modified similarity fields for the class (timeless marginal + the two
conditional half-windows), injected truth marked.

Usage (from repo root, needs outputs/classroom/events.csv):

    python vsa_cognitive_mapping/moved_object_synthetic.py
    python vsa_cognitive_mapping/moved_object_synthetic.py --what suitcase --hd-dim 8192
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vsa_cognitive_mapping.astm_traces import (  # noqa: E402
    ExactEventTable, QueryRouter, TraceSet, load_events,
)

DEFAULT_OUT = os.path.join("outputs", "classroom")

OFFSET_CANDIDATES = [
    (2.5, 1.5), (2.5, -1.5), (-2.5, 1.5), (-2.5, -1.5),
    (2.0, 1.2), (-2.0, 1.2), (2.0, -1.2), (-2.0, -1.2),
    (1.5, 0.75), (-1.5, 0.75), (1.5, -0.75), (-1.5, -0.75),
    (1.0, 0.5), (-1.0, 0.5), (1.0, -0.5), (-1.0, -0.5),
]


def pick_class(ev, prefer, min_half=30):
    """First preferred class with >= min_half events in EACH half."""
    cls = np.array(ev["class"])
    th = ev["t"].max() / 2.0
    counts = Counter(ev["class"])
    for c in prefer:
        n1 = int(((cls == c) & (ev["t"] <= th)).sum())
        n2 = int(((cls == c) & (ev["t"] > th)).sum())
        if min(n1, n2) >= min_half:
            return c, counts[c], n1, n2
    raise SystemExit(f"no class in {prefer} has >= {min_half} events per half")


def choose_offset(x2, y2, bounds, requested):
    """First candidate offset keeping ALL relocated events inside bounds."""
    for dx, dy in [tuple(requested)] + OFFSET_CANDIDATES:
        if (x2.min() + dx >= bounds[0] and x2.max() + dx <= bounds[1]
                and y2.min() + dy >= bounds[2] and y2.max() + dy <= bounds[3]):
            return float(dx), float(dy)
    raise SystemExit("no candidate offset keeps the class inside the walk")


def build_traceset(ev, hd, seed, pos_l, time_l, grid):
    """In-memory TraceSet from an event dict (no null calibration — the
    range-matched fake-class null below is what this experiment needs)."""
    classes = sorted(set(ev["class"]))
    bounds = (ev["x"].min(), ev["x"].max(), ev["y"].min(), ev["y"].max())
    tr = TraceSet(hd, seed, pos_l, time_l, classes, bounds, ev["t"].max(),
                  grid=grid)
    for i in range(len(ev["t"])):
        tr.add_event(ev["class"][i], ev["x"][i], ev["y"][i],
                     ev["t"][i], ev["conf"][i])
    tr.finalize()
    return tr


def fake_class_null(tr, router, windows, n=60, seed=777):
    """Half-window 'where' decode sims for classes NOT in the stream."""
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


def field(tr, residual, probe_time=None):
    """Full similarity map over the decoder grid (gy, gx)."""
    v = np.conj(residual)
    if probe_time is not None:
        v = v * probe_time
    s = np.real(tr.G_flat @ v.astype(np.complex64)) / tr.hd
    return s.reshape(len(tr.gy), len(tr.gx))


def analyse(tag, tr, ev, cname, wins, truth_a, truth_b, null_n):
    """Run detections (a),(b),(c) on one traceset. Returns results dict."""
    router = QueryRouter(tr)
    table = ExactEventTable(ev)
    out = {"tag": tag}

    null = fake_class_null(tr, router, wins, n=null_n)
    out["null"] = null
    print(f"  fake-class null (half-window where): mean={null['mean']:.5f} "
          f"std={null['std']:.5f} -> min_sim={null['min_sim']:.5f}")

    # (a) which_moved, gated by the range-matched null
    ranked = router.which_moved(wins[0], wins[1], min_sim=null["min_sim"])
    out["ranked"] = ranked
    ranked_all = router.which_moved(wins[0], wins[1], min_sim=0.0)
    rank = next((i + 1 for i, r in enumerate(ranked) if r["class"] == cname),
                None)
    rank_all = next((i + 1 for i, r in enumerate(ranked_all)
                     if r["class"] == cname), None)
    out["rank"], out["rank_all"] = rank, rank_all
    row = next((r for r in ranked_all if r["class"] == cname), None)
    out["row"] = row
    print(f"  (a) which_moved halves, {len(ranked)} classes pass the null "
          f"gate (of {len(ranked_all)}):")
    for i, r in enumerate(ranked[:5]):
        mark = "  <-- " + cname if r["class"] == cname else ""
        print(f"      #{i+1} {r['class']:>12}: ({r['a'][0]:+.2f},{r['a'][1]:+.2f})"
              f" -> ({r['b'][0]:+.2f},{r['b'][1]:+.2f})  {r['moved_m']:.2f} m"
              f"  sims {r['sim_a']:.5f}/{r['sim_b']:.5f}{mark}")
    print(f"      {cname}: rank {rank} of {len(ranked)} gated "
          f"(rank {rank_all} of {len(ranked_all)} ungated), "
          f"measured displacement {row['moved_m']:.2f} m")

    # (b) conditional where(class, half-window)
    out["cond"] = {}
    for w, truth, half in ((wins[0], truth_a, "first"),
                           (wins[1], truth_b, "second")):
        ans, info = router.query("where", what=cname, when=w)
        exact = table.where(what=cname, when=w)
        e_tr = float(np.hypot(ans[0] - truth[0], ans[1] - truth[1]))
        e_ex = float(np.hypot(ans[0] - exact[0], ans[1] - exact[1]))
        out["cond"][half] = {"ans": ans, "exact": exact, "err_truth": e_tr,
                             "err_exact": e_ex, "sim": info["sim"]}
        print(f"  (b) where({cname}, t in {half} half) = "
              f"({ans[0]:+.2f},{ans[1]:+.2f})  err vs truth {e_tr:.2f} m, "
              f"vs exact table {e_ex:.2f} m  (sim {info['sim']:.5f})")

    # (c) timeless marginal where(class)
    ans, info = router.query("where", what=cname)
    da = float(np.hypot(ans[0] - truth_a[0], ans[1] - truth_a[1]))
    db = float(np.hypot(ans[0] - truth_b[0], ans[1] - truth_b[1]))
    out["marginal"] = {"ans": ans, "d_a": da, "d_b": db, "sim": info["sim"]}
    print(f"  (c) where({cname}) timeless = ({ans[0]:+.2f},{ans[1]:+.2f}) -- "
          f"ONE answer for a two-place truth: {da:.2f} m from first-half "
          f"pos, {db:.2f} m from second-half pos (sim {info['sim']:.5f})")

    # fields for the figure
    C = tr.C[cname]
    out["fields"] = {
        "marginal": field(tr, tr.M["what_where"] / C),
        "first": field(tr, tr.M["event"] / C,
                       probe_time=tr.time_range(*wins[0])),
        "second": field(tr, tr.M["event"] / C,
                        probe_time=tr.time_range(*wins[1])),
    }
    out["grid"] = (tr.gx.copy(), tr.gy.copy())

    # bimodality of the timeless marginal: field value at the two truths
    fm = out["fields"]["marginal"]
    def _at(p):
        i = int(np.argmin(np.abs(tr.gx - p[0])))
        j = int(np.argmin(np.abs(tr.gy - p[1])))
        return float(fm[j, i])
    sa_, sb_, pk = _at(truth_a), _at(truth_b), float(fm.max())
    out["marginal"]["field_a"], out["marginal"]["field_b"] = sa_, sb_
    print(f"      marginal field: {sa_:.5f} at first-half truth, "
          f"{sb_:.5f} at second-half truth (peak {pk:.5f}) -- both places "
          f"carry mass, but argmax reports only one")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--what", default=None,
                    help="class to relocate (default: first of tv, suitcase "
                         "with >= 30 events per half)")
    ap.add_argument("--offset", type=float, nargs=2, default=(2.5, 1.5),
                    metavar=("DX", "DY"),
                    help="requested offset; auto-adjusted to stay in bounds")
    ap.add_argument("--hd-dim", type=int, default=16384,
                    help="16384 default: see docstring (range-kernel headroom)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pos-length-scale", type=float, default=0.75)
    ap.add_argument("--time-length-scale", type=float, default=20.0)
    ap.add_argument("--grid", type=int, default=72)
    ap.add_argument("--null-samples", type=int, default=60)
    args = ap.parse_args()

    t_start = time.time()
    ev = load_events(args.out_dir)
    t_max = float(ev["t"].max())
    th = t_max / 2.0
    wins = [(0.0, th), (th, t_max)]
    bounds = (float(ev["x"].min()), float(ev["x"].max()),
              float(ev["y"].min()), float(ev["y"].max()))

    prefer = [args.what] if args.what else ["tv", "suitcase"]
    cname, n_ev, n1, n2 = pick_class(ev, prefer)
    cls = np.array(ev["class"])
    m2 = (cls == cname) & (ev["t"] > th)
    dx, dy = choose_offset(ev["x"][m2], ev["y"][m2], bounds, args.offset)
    inj = float(np.hypot(dx, dy))
    print(f"class: {cname} ({n_ev} events; {n1} first half, {n2} second half)")
    print(f"injected offset: ({dx:+.2f}, {dy:+.2f}) m = {inj:.2f} m "
          f"displacement (requested ({args.offset[0]:+.2f},"
          f"{args.offset[1]:+.2f}); walk bounds x=[{bounds[0]:.2f},"
          f"{bounds[1]:.2f}] y=[{bounds[2]:.2f},{bounds[3]:.2f}])")

    # modified stream: relocate the class's second-half events
    ev_mod = {"class": list(ev["class"]), "x": ev["x"].copy(),
              "y": ev["y"].copy(), "t": ev["t"].copy(),
              "conf": ev["conf"].copy()}
    ev_mod["x"][m2] += dx
    ev_mod["y"][m2] += dy

    # ground truth (exact modal positions per half)
    table0 = ExactEventTable(ev)
    truth_a = table0.where(what=cname, when=wins[0])
    truth_b_ctl = table0.where(what=cname, when=wins[1])
    truth_b_mod = (truth_b_ctl[0] + dx, truth_b_ctl[1] + dy)
    nat = float(np.hypot(truth_b_ctl[0] - truth_a[0],
                         truth_b_ctl[1] - truth_a[1]))
    print(f"truth: first-half modal ({truth_a[0]:+.2f},{truth_a[1]:+.2f}); "
          f"second-half modal control ({truth_b_ctl[0]:+.2f},"
          f"{truth_b_ctl[1]:+.2f}) (natural displacement {nat:.2f} m), "
          f"relocated ({truth_b_mod[0]:+.2f},{truth_b_mod[1]:+.2f})")

    results = {}
    for tag, stream, tb in (("control", ev, truth_b_ctl),
                            ("modified", ev_mod, truth_b_mod)):
        print(f"\n=== {tag.upper()} stream (hd={args.hd_dim}) ===")
        t0 = time.time()
        tr = build_traceset(stream, args.hd_dim, args.seed,
                            args.pos_length_scale, args.time_length_scale,
                            args.grid)
        print(f"  built {tr.n_events} events, {len(tr.classes)} classes "
              f"in {time.time() - t0:.0f}s")
        results[tag] = analyse(tag, tr, stream, cname, wins, truth_a, tb,
                               args.null_samples)
        del tr  # free the decoder grid before the next build

    # summary
    rm, rc = results["modified"], results["control"]
    print(f"\n=== SUMMARY ({cname}, injected {inj:.2f} m) ===")
    print(f"(a) modified: rank {rm['rank']} (gated) / measured "
          f"{rm['row']['moved_m']:.2f} m vs injected {inj:.2f} m "
          f"(+{nat:.2f} m natural); control: rank {rc['rank']} / "
          f"{rc['row']['moved_m']:.2f} m")
    for half in ("first", "second"):
        c = rm["cond"][half]
        print(f"(b) modified where({cname},{half} half): "
              f"({c['ans'][0]:+.2f},{c['ans'][1]:+.2f}) "
              f"err {c['err_truth']:.2f} m vs truth")
    mg = rm["marginal"]
    print(f"(c) modified timeless where({cname}): "
          f"({mg['ans'][0]:+.2f},{mg['ans'][1]:+.2f}) -- {mg['d_a']:.2f} m / "
          f"{mg['d_b']:.2f} m from the two true places (cannot express both; "
          f"field {mg['field_a']:.5f} vs {mg['field_b']:.5f} at the truths)")

    fig_path = os.path.join(args.out_dir, "moved_object_synthetic.png")
    _figure(results, ev, ev_mod, cname, th, truth_a, truth_b_ctl,
            truth_b_mod, fig_path)
    print(f"saved figure -> {fig_path}")
    print(f"total runtime {time.time() - t_start:.0f}s")


def _figure(results, ev, ev_mod, cname, th, truth_a, truth_b_ctl,
            truth_b_mod, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("marginal", "timeless marginal  M_what_where / C"),
              ("first", "first half  M_event / C, T[0,t/2]"),
              ("second", "second half  M_event / C, T[t/2,t]")]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharex=True,
                             sharey=True)
    for i, (tag, stream, tb) in enumerate(
            (("control", ev, truth_b_ctl), ("modified", ev_mod, truth_b_mod))):
        r = results[tag]
        gx, gy = r["grid"]
        cls = np.array(stream["class"])
        m1 = (cls == cname) & (stream["t"] <= th)
        m2 = (cls == cname) & (stream["t"] > th)
        for j, (key, title) in enumerate(panels):
            ax = axes[i, j]
            f = r["fields"][key]
            im = ax.imshow(f, origin="lower", aspect="equal",
                           extent=[gx[0], gx[-1], gy[0], gy[-1]],
                           cmap="viridis")
            plt.colorbar(im, ax=ax, fraction=0.046)
            if key in ("marginal", "first"):
                ax.scatter(stream["x"][m1], stream["y"][m1], s=3, c="w",
                           alpha=0.5, linewidths=0)
            if key in ("marginal", "second"):
                ax.scatter(stream["x"][m2], stream["y"][m2], s=3, c="r",
                           alpha=0.5, linewidths=0)
            ax.plot(*truth_a, "wx", ms=11, mew=2.5)
            ax.plot(*tb, "r+", ms=13, mew=2.5)
            ans = {"marginal": r["marginal"]["ans"],
                   "first": r["cond"]["first"]["ans"],
                   "second": r["cond"]["second"]["ans"]}[key]
            ax.plot(*ans, "o", ms=11, mfc="none", mec="orange", mew=2.5)
            ax.set_title(f"{tag}: {title}\npeak sim {f.max():.5f}",
                         fontsize=9)
            if j == 0:
                ax.set_ylabel("y (m)")
            if i == 1:
                ax.set_xlabel("x (m)")
    fig.suptitle(
        f"Synthetic moved object: '{cname}' second-half events relocated "
        f"({truth_b_mod[0]-truth_b_ctl[0]:+.1f},"
        f"{truth_b_mod[1]-truth_b_ctl[1]:+.1f}) m\n"
        "white x = true first-half modal, red + = true second-half modal "
        "(relocated in bottom row), orange o = VSA decode; "
        "white dots = first-half events, red dots = second-half events",
        fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
