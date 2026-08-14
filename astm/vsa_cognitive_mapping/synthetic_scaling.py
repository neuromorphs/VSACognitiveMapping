"""Synthetic scaling: validate the crosstalk decomposition with CONTROLLED
feature geometry, at N far beyond any real sequence.

The real data gave 16 measured (encoder x variant) crosstalk cells at N=2,429 —
enough to *fit* a decomposition (mean / spectrum / redundancy), not enough to
claim a law. Here the three dials are set by hand and swept independently:

    x_t = m * u  +  L g_t                       features, d_in dims
    g_t = rho * g_{t-1} + sqrt(1 - rho^2) e_t   AR(1) temporal redundancy
    L: diagonal, eigvals lambda_k ∝ k^(-alpha)  spectrum (alpha=0 -> isotropic)

then the EXACT existing pipeline: random_project_to_phasor (fixed W) -> bind to
a position phasor at a random (x, y) in a 7x7 m box -> streaming bundle ->
chi(N) = N * mean|offtarget| over a retained probe subset — the same metric as
``crosstalk_scaling``.

One knob at a time from a base config:

    A  low-rank, centred      m=0, alpha tuned to eff-rank ~7/256, rho=0
    B  A + mean               m set so the mean carries ~80% of energy (yolo-like)
    C  isotropic i.i.d.       alpha=0, m=0, rho=0     (the textbook assumption)
    D  C + redundancy         rho=0.9                 (the "scene" term alone)
    E  A whitened post-hoc    PCA fit on a sample, applied streaming

Expectations, from the measured law-in-progress: B >> A (coherent mean term,
slope ~1); A > C (spectral coherence, slope ~1); C slope ~0.5 (incoherent);
D lifts C's floor with slope between 0.5 and 1; E ~ C-ish. An inconsistency is
a reportable finding, not a failure.

Nothing is ever materialised at full N: the bundle streams in chunks, and only
~``--probes`` (content, position) rows are retained for the readout.

    python -m vsa_cognitive_mapping.synthetic_scaling            # default sweep
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vsa_cognitive_mapping.vsa import random_project_to_phasor  # noqa: E402
from vsa_cognitive_mapping.classroom_pipeline import ClassroomEncoders  # noqa: E402

# per-config N ceilings: every config covers 1e2..3e5; the two cheapest
# spectra also run 1e6 so the figure spans four decades where it matters
NS_BASE = [100, 300, 1000, 3000, 10_000, 30_000, 100_000, 300_000]
NS_TOP = [1_000_000]
CONFIGS = {
    "A_lowrank":        dict(m_frac=0.0, alpha=1.7, rho=0.0, whiten=False, top=False),
    "B_lowrank_mean":   dict(m_frac=0.8, alpha=1.7, rho=0.0, whiten=False, top=False),
    "C_isotropic":      dict(m_frac=0.0, alpha=0.0, rho=0.0, whiten=False, top=True),
    "D_iso_redundant":  dict(m_frac=0.0, alpha=0.0, rho=0.9, whiten=False, top=True),
    "E_lowrank_whit":   dict(m_frac=0.0, alpha=1.7, rho=0.0, whiten=True,  top=False),
    # Persistence hypothesis (added after the first sweep): static low rank
    # (A) and short AR redundancy (D, corr time ~10) both kept the sqrt-N
    # exponent — only the MEAN (B) went O(N). If the exponent is governed by
    # correlation PERSISTENCE rather than the spectrum, near-unit rho (corr
    # time ~2000, mimicking minutes-long place clusters in a real walk) should
    # push the slope toward 1 over the measured range. This is what real
    # centred crops do (slope ~1.0), and what the synthetic configs above
    # failed to reproduce.
    "D2_iso_persistent":    dict(m_frac=0.0, alpha=0.0, rho=0.9995, whiten=False, top=True),
    "F_lowrank_persistent": dict(m_frac=0.0, alpha=1.7, rho=0.9995, whiten=False, top=False),
}


def spectrum(d_in, alpha):
    lam = np.arange(1, d_in + 1, dtype=np.float64) ** (-alpha)
    lam *= d_in / lam.sum()                       # normalise total variance
    pr = lam.sum() ** 2 / (lam ** 2).sum()
    return np.sqrt(lam), pr


class Stream:
    """Chunked generator of (features, positions) with AR(1) carry."""

    def __init__(self, cfg, d_in, seed, box=7.0):
        self.rng = np.random.RandomState(seed)
        self.sd, self.pr = spectrum(d_in, cfg["alpha"])
        u = self.rng.randn(d_in)
        self.u = u / np.linalg.norm(u)
        # m_frac = share of E||x||^2 carried by the mean: ||m u||^2 = f/(1-f) * tr(Sigma)
        f = cfg["m_frac"]
        self.m = 0.0 if f <= 0 else np.sqrt(f / (1 - f) * d_in)
        self.rho = cfg["rho"]
        self.carry = self.rng.randn(d_in)
        self.box = box
        self.d_in = d_in
        self.W = None                             # PCA whitener (mu, comps, scale)

    def fit_whitener(self, n=5000):
        # Full-rank whitening (all d_in components) so the output dimension is
        # unchanged and the one shared projection W applies to every config.
        # A rank-reducing whitener here crashed on the shared W (4096x64 vs
        # 256x8192); dimension-preserving whitening still flattens the
        # spectrum, which is the property config E exists to isolate.
        X = self._chunk_raw(n)
        mu = X.mean(0, keepdims=True)
        Xc = X - mu
        _, s, vt = np.linalg.svd(Xc, full_matrices=False)
        sc = (s / np.sqrt(len(X) - 1)) + 1e-9
        self.W = (mu, vt, sc)
        self.carry = self.rng.randn(self.d_in)    # restart the AR chain

    def _chunk_raw(self, n):
        if self.rho > 0:
            g = np.empty((n, self.d_in))
            c, r, q = self.carry, self.rho, np.sqrt(1 - self.rho ** 2)
            eps = self.rng.randn(n, self.d_in)
            for i in range(n):                    # sequential by construction
                c = r * c + q * eps[i]
                g[i] = c
            self.carry = c
        else:
            g = self.rng.randn(n, self.d_in)
        return self.m * self.u[None, :] + g * self.sd[None, :]

    def chunk(self, n):
        X = self._chunk_raw(n)
        if self.W is not None:
            mu, comps, sc = self.W
            X = ((X - mu) @ comps.T) / sc
        xy = self.rng.uniform(-self.box / 2, self.box / 2, size=(n, 2))
        return X, xy


def run_config(name, cfg, args, enc, phx, phy, Wproj):
    Ns = NS_BASE + (NS_TOP if cfg["top"] and args.full_top else [])
    Ns = [n for n in Ns if n <= args.max_n]
    out_rows = []
    for seed in args.seeds:
        st = Stream(cfg, args.d_in, 10_000 + seed)
        if cfg["whiten"]:
            st.fit_whitener()
        n_max = Ns[-1]
        # LOG-uniform probe positions: uniform sampling puts ~none of the 200
        # probes below N=1e3 when n_max=1e6, so every small-N checkpoint fell
        # under the >=10-probe guard and the curve lost its left half. Log
        # spacing gives each decade a similar probe count.
        prng = np.random.RandomState(77 + seed)
        raw_idx = np.exp(prng.uniform(0, np.log(n_max),
                                      size=args.probes * 3)).astype(np.int64)
        probe_idx = np.unique(np.clip(raw_idx, 0, n_max - 1))[:args.probes * 2]
        probe_idx = np.sort(prng.choice(probe_idx,
                                        size=min(args.probes, len(probe_idx)),
                                        replace=False))
        # Chunk sizes are clipped to land EXACTLY on each checkpoint: advancing
        # in fixed chunk strides skipped every intermediate N (done was only
        # ever a multiple of the chunk size), so a whole sweep produced one
        # endpoint per config and no curve.
        cps = sorted(Ns)
        M = np.zeros(args.hd, np.complex128)
        pc, pp = [], []                            # retained probe content / position
        done, t0, pi = 0, time.time(), 0
        results = {}
        while done < n_max:
            next_cp = next(c for c in cps if c > done)
            n = min(args.chunk, next_cp - done)
            X, xy = st.chunk(n)
            z, _ = random_project_to_phasor(
                torch.from_numpy(np.ascontiguousarray(X)).float(), d=args.hd, W=Wproj)
            C = z.numpy().astype(np.complex128)
            P = np.exp(1j * (phx[None, :] * (xy[:, 0:1] / enc.pos_l)
                             + phy[None, :] * (xy[:, 1:2] / enc.pos_l)))
            M += (C * P).sum(axis=0)
            while pi < len(probe_idx) and probe_idx[pi] < done + n:
                k = probe_idx[pi] - done
                pc.append(C[k]); pp.append(P[k]); pi += 1
            done += n
            if done in cps and len(pc) >= 10:
                Cp, Pp = np.asarray(pc), np.asarray(pp)
                Mb = M / done
                S = np.real((Mb[None, :] * np.conj(Pp)) @ np.conj(Cp).T) / args.hd
                m_ = len(Cp)
                off = np.abs(S[~np.eye(m_, dtype=bool)].reshape(m_, m_ - 1)).mean()
                results[done] = done * off
                row = dict(config=name, seed=seed, N=done, chi=float(done * off),
                           probes=m_, pr_spectrum=float(st.pr),
                           mean_frac=cfg["m_frac"], rho=cfg["rho"],
                           whiten=cfg["whiten"], elapsed_s=round(time.time() - t0, 1))
                out_rows.append(row)
                with open(args.rows_path, "a") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"  {name} seed{seed} N={done:>8}: chi={done*off:9.2f}x "
                      f"({time.time()-t0:6.1f}s)", flush=True)
        ns = sorted(results)
        if len(ns) >= 3:
            lx, ly = np.log(ns), np.log([results[n] for n in ns])
            m_fit = np.polyfit(lx, ly, 1)[0]
            print(f"  {name} seed{seed}: chi slope {m_fit:+.3f}", flush=True)
            out_rows.append(dict(config=name, seed=seed, N=-1, chi_slope=float(m_fit)))
            with open(args.rows_path, "a") as f:
                f.write(json.dumps(out_rows[-1]) + "\n")
    return out_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="outputs/synthetic")
    ap.add_argument("--hd", type=int, default=4096)
    ap.add_argument("--d-in", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--probes", type=int, default=200)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    ap.add_argument("--max-n", type=int, default=1_000_000)
    ap.add_argument("--full-top", action="store_true", default=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    args.rows_path = os.path.join(args.out_dir, "rows.jsonl")

    enc = ClassroomEncoders(args.hd, 100, 0.75, 20.0)
    phx = np.angle(enc.Bx.values)
    phy = np.angle(enc.By.values)
    # one fixed projection W shared by every config (as the pipeline does)
    _z, Wproj = random_project_to_phasor(torch.zeros(2, args.d_in), d=args.hd, seed=0)

    for name in args.configs:
        cfg = CONFIGS[name]
        sd, pr = spectrum(args.d_in, cfg["alpha"])
        print(f"== {name}: alpha={cfg['alpha']} (spectrum PR={pr:.1f}/{args.d_in}), "
              f"mean_frac={cfg['m_frac']}, rho={cfg['rho']}, whiten={cfg['whiten']}",
              flush=True)
        run_config(name, cfg, args, enc, phx, phy, Wproj)

    # ---- figure ----------------------------------------------------------
    rows = [json.loads(l) for l in open(args.rows_path)]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.5, 5.6), constrained_layout=True)
    colors = {"A_lowrank": "#b9772a", "B_lowrank_mean": "#b02a24",
              "C_isotropic": "#0d7d88", "D_iso_redundant": "#7a4bb0",
              "E_lowrank_whit": "#0f7a4b"}
    for name in CONFIGS:
        pts = sorted([(r["N"], r["chi"]) for r in rows
                      if r.get("config") == name and r.get("N", -1) > 0])
        if not pts:
            continue
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-",
                color=colors.get(name, "0.4"), label=name, lw=1.6, ms=4)
    for sref, ls in ((1.0, ":"), (0.5, "-.")):
        ax.plot([1e2, 1e6], [0.5 * (n / 1e2) ** sref for n in (1e2, 1e6)],
                ls, color="0.6", lw=0.9, label=f"slope {sref:g}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("N bundled (content ⊗ position) pairs")
    ax.set_ylabel("chi = interference vs one stored item's signal")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    ax.set_title(f"Synthetic scaling, hd={args.hd}, d_in={args.d_in}: "
                 f"one geometry knob per curve", fontsize=10)
    path = os.path.join(args.out_dir, "synthetic_scaling.png")
    fig.savefig(path, dpi=140)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
