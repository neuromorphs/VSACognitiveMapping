"""ASTM abstention calibration: proper metrics on a large labeled battery.

Upgrades the z-threshold abstention check to real selective-prediction
metrics. A ~500-query labeled battery is generated from the canonical event
stream (``outputs/classroom/events.csv``) + the exact event table:

  * answerable queries across ALL classes with >= 3 events (not just the
    frequent ones): where(c) all-time, where(c, t-point), where(c,
    t-window), when(c), when(c, place), what(place), what(place, t);
  * ~100 deliberately unanswerable queries: class x time windows with zero
    events of that class in the window, probe positions outside the walk,
    and when(class, place) pairings at places the class never appears.

GUARD BAND (FIX 3, evaluation-hardening 2026-07-30, default ON): the
original "unanswerable" labels had NO guard band — an event of the class
sitting one frame outside the window (or just past the place radius) made
the query "unanswerable" while the memory's smooth kernels legitimately
responded, counting correct behaviour as error (deflating our own AUROC).
Now a class x time window is unanswerable only if no event of the class
lies within window +/- 2*time_l (40 fr at the default scales), and a
place-mismatch when(c, place) only if no event of c lies within
radius + 2*pos_l (= 2.5 m) of the probe. ``--no-guard-band`` reproduces
the old (noisy) labels exactly.

Each query is labeled against the exact table (position correct = within
1.0 m of the modal answer, time within 40 fr, class exact; any answer to an
unanswerable query counts as an error). Confidence scores are the router's
null-calibrated z AND the raw peak similarity. Reported:

  * AUROC (correct vs incorrect+unanswerable) for z and for raw sim,
    overall and per decode type;
  * ECE (10 equal-width probability bins) and Brier score. Choice
    documented here: both z and raw sim are mapped to P(correct) through a
    per-score logistic (Platt scaling, IRLS, fitted IN-SAMPLE on the same
    battery — an upper bound on calibrated-ness, stated as such);
  * risk-coverage curves (sweep the score threshold; x = coverage =
    fraction answered, y = selective risk = error rate among answered).

Figure -> outputs/classroom/calibration_eval.png (reliability diagram +
risk-coverage, z vs raw sim).

Usage (from repo root; needs events.csv and astm_traces.pt from
``python vsa_cognitive_mapping/astm_traces.py export`` / ``build``):

    python vsa_cognitive_mapping/calibration_eval.py
    python vsa_cognitive_mapping/calibration_eval.py --min-class-events 3 --seed 0
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vsa_cognitive_mapping.astm_traces import (  # noqa: E402
    ExactEventTable, QueryRouter, TraceSet, load_events,
)

DEFAULT_OUT = os.path.join("outputs", "classroom")
POS_OK_M = 1.0     # position correct = within 1 m of exact modal answer
TIME_OK_FR = 40.0  # time correct = within 40 frames


# ==========================================================================
# Battery generation
# ==========================================================================

def build_battery(ev, table, rng, min_class_events=3,
                  n_tpoint=5, n_twin=4, n_what=60,
                  n_gapwin=2, n_outside=30, n_mismatch=25,
                  t_guard=0.0, place_guard=2.0):
    """Labeled query battery. Each item:
    {desc, decode, kind, kwargs, exact, answerable}
    kind in {pos, time, class} decides the correctness rule.

    t_guard / place_guard implement the FIX 3 guard band on the
    unanswerable labels (see module docstring). t_guard=0, place_guard=2.0
    reproduces the original (unguarded) battery exactly."""
    cls_arr = np.array(ev["class"])
    t_arr, x_arr, y_arr = ev["t"], ev["x"], ev["y"]
    t_max = float(t_arr.max())
    bx = (float(x_arr.min()), float(x_arr.max()))
    by = (float(y_arr.min()), float(y_arr.max()))
    counts = defaultdict(int)
    for c in cls_arr:
        counts[c] += 1
    classes = sorted(c for c in counts if counts[c] >= min_class_events)

    Q = []

    def add(desc, decode, kind, kwargs, exact, answerable=True):
        Q.append({"desc": desc, "decode": decode, "kind": kind,
                  "kwargs": kwargs, "exact": exact, "answerable": answerable})

    # ---- answerable ------------------------------------------------------
    for c in classes:
        idx = np.where(cls_arr == c)[0]
        # where(c) all-time
        add(f"where({c})", "where", "pos", dict(what=c), table.where(what=c))
        # when(c)
        add(f"when({c})", "when", "time", dict(what=c), table.when(what=c))
        # when(c, place) at the class's modal position
        p = table.where(what=c)
        add(f"when({c},@modal)", "when", "time", dict(what=c, where=p),
            table.when(what=c, near_xy=p, radius=1.0))
        # where(c, t-point) at sampled event times
        for t in rng.choice(t_arr[idx], size=min(n_tpoint, len(idx)),
                            replace=False):
            t = float(t)
            add(f"where({c},t={t:.0f})", "where", "pos",
                dict(what=c, when=t),
                table.where(what=c, when=(t - TIME_OK_FR, t + TIME_OK_FR)))
        # where(c, t-window): 120-frame windows centred on sampled events
        for t in rng.choice(t_arr[idx], size=min(n_twin, len(idx)),
                            replace=False):
            a = max(0.0, float(t) - 60.0)
            w = (a, min(t_max, a + 120.0))
            add(f"where({c},t in {w[0]:.0f}:{w[1]:.0f})", "where", "pos",
                dict(what=c, when=w), table.where(what=c, when=w))

    # what(place) and what(place, t) at sampled event locations
    sel = rng.choice(len(t_arr), size=min(n_what, len(t_arr)), replace=False)
    for i in sel:
        p = (float(x_arr[i]), float(y_arr[i]))
        add(f"what(({p[0]:+.1f},{p[1]:+.1f}))", "what", "class",
            dict(where=p), table.what(near_xy=p, radius=1.0))
    sel = rng.choice(len(t_arr), size=min(n_what, len(t_arr)), replace=False)
    for i in sel:
        p = (float(x_arr[i]), float(y_arr[i]))
        t = float(t_arr[i])
        add(f"what(({p[0]:+.1f},{p[1]:+.1f}),t={t:.0f})", "what", "class",
            dict(where=p, when=t),
            table.what(near_xy=p, when=(t - 60.0, t + 60.0), radius=1.0))

    # ---- unanswerable ----------------------------------------------------
    # (1) class x time window with ZERO events of that class in the window
    #     +/- t_guard (guard band, FIX 3: without it an event 1 frame past
    #     the window edge made a legitimately-answered query count as error)
    for c in classes:
        tc = t_arr[cls_arr == c]
        got = 0
        for _ in range(200):
            if got >= n_gapwin:
                break
            a = float(rng.uniform(0.0, t_max - 80.0))
            w = (a, a + 80.0)
            if np.any((tc >= w[0] - t_guard) & (tc <= w[1] + t_guard)):
                continue
            add(f"where({c},t in {w[0]:.0f}:{w[1]:.0f}) [UNANS]", "where",
                "pos", dict(what=c, when=w), None, answerable=False)
            got += 1
    # (2) probe positions outside the walk (1.5-3 m beyond the bounds)
    for _ in range(n_outside):
        side = rng.randint(4)
        d = float(rng.uniform(1.5, 3.0))
        if side == 0:
            p = (bx[0] - d, float(rng.uniform(*by)))
        elif side == 1:
            p = (bx[1] + d, float(rng.uniform(*by)))
        elif side == 2:
            p = (float(rng.uniform(*bx)), by[0] - d)
        else:
            p = (float(rng.uniform(*bx)), by[1] + d)
        add(f"what(({p[0]:+.1f},{p[1]:+.1f})) [UNANS]", "what", "class",
            dict(where=p), None, answerable=False)
    # (3) when(class, place) at in-bounds places the class never appears
    #     (place_guard = scoring radius + 2*pos_l under FIX 3)
    got = 0
    for _ in range(2000):
        if got >= n_mismatch:
            break
        c = classes[rng.randint(len(classes))]
        m = cls_arr == c
        p = (float(rng.uniform(*bx)), float(rng.uniform(*by)))
        if np.any(np.hypot(x_arr[m] - p[0], y_arr[m] - p[1]) <= place_guard):
            continue
        add(f"when({c},@({p[0]:+.1f},{p[1]:+.1f})) [UNANS]", "when", "time",
            dict(what=c, where=p), None, answerable=False)
        got += 1

    return Q


def run_battery(router, battery):
    """Execute every query; attach answer, sim, z, y (1 = correct)."""
    rows = []
    for q in battery:
        ans, info = router.query(q["decode"], **q["kwargs"])
        exact = q["exact"]
        if not q["answerable"] or exact is None:
            status, y = "unanswerable", 0
        elif q["kind"] == "pos":
            e = float(np.hypot(ans[0] - exact[0], ans[1] - exact[1]))
            y = int(e <= POS_OK_M)
            status = "correct" if y else "incorrect"
        elif q["kind"] == "time":
            e = abs(float(ans) - float(exact))
            y = int(e <= TIME_OK_FR)
            status = "correct" if y else "incorrect"
        else:
            y = int(ans == exact)
            status = "correct" if y else "incorrect"
        rows.append({**q, "ans": ans, "sim": float(info["sim"]),
                     "z": float(info["z"]) if info["z"] is not None else np.nan,
                     "y": y, "status": status})
    return rows


# ==========================================================================
# Metrics (numpy only)
# ==========================================================================

def auroc(scores, y):
    """Rank-based AUROC with tie handling (Mann-Whitney U)."""
    scores, y = np.asarray(scores, float), np.asarray(y, int)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), float)
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):  # average ranks over ties
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def platt_fit(scores, y, iters=50):
    """Logistic p = sigmoid(a*s + b) by IRLS (Newton). Returns (a, b)."""
    s = np.asarray(scores, float)
    mu, sd = s.mean(), max(s.std(), 1e-12)
    sn = (s - mu) / sd
    X = np.stack([sn, np.ones_like(sn)], axis=1)
    w = np.zeros(2)
    yv = np.asarray(y, float)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-X @ w))
        g = X.T @ (p - yv)
        W = p * (1 - p)
        H = X.T @ (X * W[:, None]) + 1e-9 * np.eye(2)
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-10:
            break
    a, b = w[0] / sd, w[1] - w[0] * mu / sd  # un-standardize
    return float(a), float(b)


def platt_apply(scores, ab):
    a, b = ab
    return 1.0 / (1.0 + np.exp(-(a * np.asarray(scores, float) + b)))


def ece(p, y, n_bins=10):
    """Expected calibration error, equal-width bins on p."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    e, bins = 0.0, []
    for k in range(n_bins):
        m = (p >= edges[k]) & (p < edges[k + 1] if k < n_bins - 1 else p <= 1.0)
        if not m.any():
            bins.append(None)
            continue
        conf, acc = float(p[m].mean()), float(y[m].mean())
        e += m.sum() / len(p) * abs(acc - conf)
        bins.append((conf, acc, int(m.sum())))
    return float(e), bins


def risk_coverage(scores, y):
    """Sweep the score threshold. Returns (coverage, risk) arrays."""
    scores, y = np.asarray(scores, float), np.asarray(y, int)
    order = np.argsort(-scores, kind="mergesort")
    err = (1 - y)[order]
    cum_err = np.cumsum(err)
    n = len(y)
    k = np.arange(1, n + 1)
    return k / n, cum_err / k


# ==========================================================================
# Main
# ==========================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--traces", default=None,
                    help="path to astm_traces.pt (default: <out-dir>/astm_traces.pt)")
    ap.add_argument("--grid", type=int, default=72)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-class-events", type=int, default=3)
    ap.add_argument("--z-threshold", type=float, default=3.0)
    ap.add_argument("--no-guard-band", action="store_true",
                    help="reproduce the ORIGINAL unguarded unanswerable "
                         "labels (pre-hardening; deflates AUROC — an event "
                         "just outside the window/radius counts a legitimate "
                         "answer as error)")
    args = ap.parse_args()

    t_start = time.time()
    ev = load_events(args.out_dir)
    table = ExactEventTable(ev)
    path = args.traces or os.path.join(args.out_dir, "astm_traces.pt")
    tr = TraceSet.load(path, grid=args.grid)
    if not tr.null_calib:
        print("(no stored null calibration; calibrating now)")
        tr.calibrate(ev)
    router = QueryRouter(tr, z_threshold=args.z_threshold)

    rng = np.random.RandomState(args.seed)
    if args.no_guard_band:
        t_guard, place_guard = 0.0, 2.0  # original (pre-hardening) labels
        print("GUARD BAND OFF (--no-guard-band): reproducing the ORIGINAL "
              "noisy unanswerable labels")
    else:
        t_guard = 2.0 * float(tr.enc.time_l)          # window +/- 40 fr
        place_guard = 1.0 + 2.0 * float(tr.enc.pos_l)  # radius + 1.5 = 2.5 m
        print(f"guard band ON (FIX 3 default): unanswerable windows require "
              f"no class event within +/-{t_guard:.0f} fr; place mismatches "
              f"require none within {place_guard:.1f} m")
    battery = build_battery(ev, table, rng,
                            min_class_events=args.min_class_events,
                            t_guard=t_guard, place_guard=place_guard)
    n_ans = sum(1 for q in battery if q["answerable"] and q["exact"] is not None)
    n_un = len(battery) - n_ans
    print(f"battery: {len(battery)} queries ({n_ans} answerable, "
          f"{n_un} unanswerable) over "
          f"{len(set(q['kwargs'].get('what') for q in battery) - {None})} classes")

    rows = run_battery(router, battery)
    y = np.array([r["y"] for r in rows])
    z = np.array([r["z"] for r in rows])
    sim = np.array([r["sim"] for r in rows])
    n_corr = int(y.sum())
    n_inc = sum(1 for r in rows if r["status"] == "incorrect")
    n_una = sum(1 for r in rows if r["status"] == "unanswerable")
    print(f"outcomes: {n_corr} correct, {n_inc} incorrect, "
          f"{n_una} unanswerable (any answer to these = error); "
          f"base error rate {(1 - y.mean()) * 100:.1f}%")

    # ---- AUROC ----------------------------------------------------------
    au_z, au_s = auroc(z, y), auroc(sim, y)
    print(f"\nAUROC (correct vs incorrect+unanswerable):")
    print(f"  z (null-calibrated): {au_z:.4f}")
    print(f"  raw peak sim       : {au_s:.4f}")
    print("  per decode type (z | sim | n | acc):")
    for d in ("where", "when", "what"):
        m = np.array([r["decode"] == d for r in rows])
        if m.sum():
            print(f"    {d:5s}: {auroc(z[m], y[m]):.4f} | "
                  f"{auroc(sim[m], y[m]):.4f} | n={int(m.sum()):3d} | "
                  f"acc={y[m].mean():.2f}")

    # ---- ECE / Brier (Platt-scaled, in-sample; documented in docstring) --
    ab_z, ab_s = platt_fit(z, y), platt_fit(sim, y)
    p_z, p_s = platt_apply(z, ab_z), platt_apply(sim, ab_s)
    ece_z, bins_z = ece(p_z, y)
    ece_s, bins_s = ece(p_s, y)
    brier_z = float(np.mean((p_z - y) ** 2))
    brier_s = float(np.mean((p_s - y) ** 2))
    print(f"\nECE (10 bins, Platt-scaled in-sample) / Brier:")
    print(f"  z  : ECE={ece_z:.4f}  Brier={brier_z:.4f} "
          f"(logistic a={ab_z[0]:.3g}, b={ab_z[1]:.3g})")
    print(f"  sim: ECE={ece_s:.4f}  Brier={brier_s:.4f} "
          f"(logistic a={ab_s[0]:.3g}, b={ab_s[1]:.3g})")

    # ---- risk-coverage --------------------------------------------------
    cov_z, risk_z = risk_coverage(z, y)
    cov_s, risk_s = risk_coverage(sim, y)
    print(f"\nrisk-coverage (selective risk at coverage):")
    for c_target in (0.5, 0.7, 0.8, 0.9, 1.0):
        iz = int(np.searchsorted(cov_z, c_target))
        isim = int(np.searchsorted(cov_s, c_target))
        iz, isim = min(iz, len(risk_z) - 1), min(isim, len(risk_s) - 1)
        print(f"  cov={c_target:.0%}:  z-risk={risk_z[iz]:.3f}  "
              f"sim-risk={risk_s[isim]:.3f}")
    # default operating point
    m_conf = z >= args.z_threshold
    if m_conf.any():
        print(f"  default z>={args.z_threshold:g}: coverage "
              f"{m_conf.mean():.1%}, selective risk "
              f"{(1 - y[m_conf].mean()):.3f}")

    verdict = "z beats raw sim" if au_z > au_s + 0.01 else \
        ("raw sim beats z" if au_s > au_z + 0.01 else "z ~ raw sim (tie)")
    print(f"\nverdict: {verdict} as a confidence signal "
          f"(AUROC {au_z:.3f} vs {au_s:.3f})")

    # ---- figure ---------------------------------------------------------
    fig_path = os.path.join(args.out_dir, "calibration_eval.png")
    _figure(bins_z, bins_s, ece_z, ece_s, cov_z, risk_z, cov_s, risk_s,
            au_z, au_s, args.z_threshold, z, y, fig_path)
    print(f"saved figure -> {fig_path}")
    print(f"total runtime {time.time() - t_start:.0f}s")


def _figure(bins_z, bins_s, ece_z, ece_s, cov_z, risk_z, cov_s, risk_s,
            au_z, au_s, z_thr, z, y, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

    # reliability diagram
    ax1.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    for bins, ecev, col, name in ((bins_z, ece_z, "C0", "z (null-calibrated)"),
                                  (bins_s, ece_s, "C1", "raw peak sim")):
        xs = [b[0] for b in bins if b]
        ys = [b[1] for b in bins if b]
        ns = [b[2] for b in bins if b]
        ax1.plot(xs, ys, "o-", color=col, label=f"{name} (ECE={ecev:.3f})")
        for xx, yy, nn in zip(xs, ys, ns):
            ax1.annotate(str(nn), (xx, yy), textcoords="offset points",
                         xytext=(0, 6), fontsize=6, color=col, ha="center")
    ax1.set_xlabel("predicted P(correct) (Platt-scaled, in-sample)")
    ax1.set_ylabel("empirical accuracy")
    ax1.set_title("Reliability diagram (10 bins; bin counts annotated)")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # risk-coverage
    ax2.plot(cov_z, risk_z, "-", color="C0",
             label=f"z sweep (AUROC {au_z:.3f})")
    ax2.plot(cov_s, risk_s, "-", color="C1",
             label=f"raw sim sweep (AUROC {au_s:.3f})")
    m = z >= z_thr
    if m.any():
        ax2.plot([m.mean()], [1 - y[m].mean()], "k*", ms=12,
                 label=f"default z>={z_thr:g}")
    ax2.set_xlabel("coverage (fraction answered)")
    ax2.set_ylabel("selective risk (error rate among answered)")
    ax2.set_title("Risk-coverage (threshold sweep)")
    ax2.set_xlim(0, 1.02)
    ax2.set_ylim(bottom=0)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.suptitle("ASTM abstention calibration: z vs raw peak similarity",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
