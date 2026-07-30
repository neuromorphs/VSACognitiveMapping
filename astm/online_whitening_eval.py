"""Online (causal) whitening evaluation for the classroom VSA pipeline.

The pipeline's load-bearing preprocessing step whitens the YOLO content
embeddings with statistics of the WHOLE walk (``pca_components`` over all
1,239 frames) -- an oracle a deployed robot does not have. This script
measures how quickly CAUSAL (streaming) whitening statistics become good
enough, and what the transition costs in recall.

Conditions (same random phasor projection W, hd 8192 seed 0; same position
codes, ClassroomEncoders seed 100, pos length scale 0.75):

    oracle    offline whitening with whole-walk stats (the current pipeline:
              ``pca_components`` top-64 standardized scores)
    causal    frame t's content whitened with W_t built from frames <= t only
              (streaming Welford mean + covariance, per-frame 256x256 eigh,
              top-64 components, eigenvector signs aligned to the previous
              frame for temporal continuity; frames before --min-frames fall
              back to centering-only through a fixed orthonormal 256->64
              projection, since the covariance is still rank-deficient)
    ema       secondary causal variant: EMA mean/covariance with half-life
              --ema-halflife frames instead of Welford (exact, all-history)
    frozen-K  whitening frozen after the first K frames ("calibration lap"):
              Welford stats at frame K, applied to every frame

Metric: the pipeline's episodic position-recall protocol -- bundle every 3rd
frame's content (x) ctx_pos into one memory (mean of bound pairs), then for
every stored frame (a) unbind by its own position code, top-1 content match
against the stored codebook -> position error between true and matched
frame; (b) reverse: unbind by content, decode position on a 56x56 metric
grid -> position error to the grid argmax. Causal is additionally split into
early/mid/late thirds of the walk (the transition curve).

Isotropy trajectory: mean |off-diagonal| cosine of the whitened features in
a sliding 200-frame window, causal vs oracle (~0.067 whole-walk) vs raw
(~0.94) -- does causal whitening converge to oracle isotropy, and by when?

Writes outputs/classroom/online_whitening.png. Read-only with respect to the
rest of the package: imports from classroom_pipeline / vsa at call time and
touches no existing file.

Run:
    python vsa_cognitive_mapping/online_whitening_eval.py
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_OUT = os.path.join("outputs", "classroom")


# --------------------------------------------------------------------------
# streaming whitening
# --------------------------------------------------------------------------

def _top_k_whitener(cov, mean, k, prev_V, eps_rel=1e-8):
    """Eigendecompose cov, keep top-k components, sign-align to prev_V.

    Returns (V, inv_std, mean): whitened feature of x is
    ((x - mean) @ V) * inv_std."""
    lam, V = np.linalg.eigh(cov)          # ascending
    order = np.argsort(lam)[::-1][:k]
    lam_k = lam[order]
    V_k = V[:, order]
    if prev_V is not None:                # temporal continuity of the basis
        signs = np.sign(np.sum(V_k * prev_V, axis=0))
        signs[signs == 0] = 1.0
        V_k = V_k * signs
    floor = max(eps_rel * float(lam_k.max()), 1e-12)
    inv_std = 1.0 / np.sqrt(np.maximum(lam_k, floor))
    return V_k, inv_std, mean


def causal_whiten_welford(emb, k, min_frames, frozen_ks, seed):
    """One causal pass. Returns (feats (N,k), frozen: {K: (V, inv_std, mean)}).

    At frame t the transform uses frames <= t only (Welford mean + M2).
    Before min_frames: centering-only through a fixed orthonormal 256->k
    projection (covariance still rank-deficient). The phasor projection is
    scale-invariant per (i, q) pair, so the fallback's raw scale is harmless.
    """
    N, D = emb.shape
    rng = np.random.RandomState(seed)
    P_fallback, _ = np.linalg.qr(rng.randn(D, k))
    mean = np.zeros(D)
    M2 = np.zeros((D, D))
    prev_V = None
    feats = np.zeros((N, k))
    frozen = {}
    for t in range(N):
        x = emb[t]
        n = t + 1
        delta = x - mean
        mean = mean + delta / n
        M2 = M2 + np.outer(delta, x - mean)
        if n >= min_frames:
            V, inv_std, mu = _top_k_whitener(M2 / (n - 1), mean.copy(), k, prev_V)
            prev_V = V
            feats[t] = ((x - mu) @ V) * inv_std
        else:
            # frame 0 centers to the exact zero vector (mean == x): use the
            # raw embedding so the phasor projection stays defined
            c = (x - mean) if t > 0 else x
            feats[t] = c @ P_fallback
        if n in frozen_ks:
            frozen[n] = _top_k_whitener(M2 / (n - 1), mean.copy(), k, None)
    # sign-aligned to the causal chain so the code-drift diagnostic compares
    # like with like (eigenvector sign is otherwise arbitrary)
    frozen["final"] = _top_k_whitener(M2 / (N - 1), mean.copy(), k, prev_V)
    return feats, frozen


def causal_whiten_ema(emb, k, min_frames, halflife, seed):
    """EMA-statistics causal variant (same fallback/alignment as Welford)."""
    N, D = emb.shape
    a = 1.0 - 0.5 ** (1.0 / halflife)
    rng = np.random.RandomState(seed)
    P_fallback, _ = np.linalg.qr(rng.randn(D, k))
    mean = emb[0].copy()
    C = np.zeros((D, D))
    prev_V = None
    feats = np.zeros((N, k))
    for t in range(N):
        x = emb[t]
        if t > 0:
            delta = x - mean
            mean = mean + a * delta
            C = (1.0 - a) * (C + a * np.outer(delta, delta))
        if t + 1 >= min_frames:
            V, inv_std, mu = _top_k_whitener(C, mean.copy(), k, prev_V)
            prev_V = V
            feats[t] = ((x - mu) @ V) * inv_std
        else:
            c = (x - mean) if t > 0 else x
            feats[t] = c @ P_fallback
    return feats


# --------------------------------------------------------------------------
# recall metric (the pipeline's episodic protocol, vectorized)
# --------------------------------------------------------------------------

def project_content(feats, hd, W, seed):
    """Fixed random phasor projection shared by every condition."""
    from vsa_cognitive_mapping.vsa import random_project_to_phasor
    x = torch.from_numpy(np.ascontiguousarray(feats)).float()
    if W is None:
        z, W = random_project_to_phasor(x, d=hd, seed=seed)
    else:
        z, W = random_project_to_phasor(x, d=hd, W=W)
    return z.numpy().astype(np.complex128), W


def recall_eval(z_S, ctx_S, xs, ys, grid_flat, grid_xy, thirds=None):
    """Bundle mean(content (x) ctx_pos) over the stride split, then:
    (a) unbind by position -> top-1 content match -> error to matched frame;
    (b) unbind by content -> grid-decode position -> error to grid argmax.

    Returns a dict; if `thirds` (list of index arrays into S) is given, adds
    per-third numbers."""
    hd = z_S.shape[1]
    M = (z_S * ctx_S).mean(axis=0)

    # (a) position -> content
    R = M[None, :] / ctx_S
    sims = np.real(R @ np.conj(z_S).T) / hd
    best = np.argmax(sims, axis=1)
    acc = best == np.arange(len(z_S))
    err_c = np.hypot(xs - xs[best], ys - ys[best])

    # (b) content -> position on the metric grid
    Rp = (M[None, :] / z_S).astype(np.complex64)
    simsg = np.real(np.conj(Rp) @ grid_flat.T) / hd
    pk = np.argmax(simsg, axis=1)
    err_p = np.hypot(xs - grid_xy[pk, 0], ys - grid_xy[pk, 1])

    out = {"top1_acc": float(acc.mean()),
           "L1_content_m": float(err_c.mean()),
           "L1_grid_m": float(err_p.mean())}
    if thirds is not None:
        for name, idx in thirds:
            out[name] = {"top1_acc": float(acc[idx].mean()),
                         "L1_content_m": float(err_c[idx].mean()),
                         "L1_grid_m": float(err_p[idx].mean())}
    return out


def sliding_isotropy(feats, win, step):
    """Mean |off-diagonal| cosine in sliding windows -> (centers, values)."""
    U = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12)
    centers, vals = [], []
    iu = np.triu_indices(win, k=1)
    for s in range(0, len(U) - win + 1, step):
        C = U[s:s + win] @ U[s:s + win].T
        vals.append(float(np.abs(C[iu]).mean()))
        centers.append(s + win // 2)
    return np.array(centers), np.array(vals)


def overall_isotropy(feats):
    U = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12)
    C = U @ U.T
    return float(np.abs(C[np.triu_indices(len(U), k=1)]).mean())


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--hd-dim", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pos-length-scale", type=float, default=0.75)
    ap.add_argument("--time-length-scale", type=float, default=20.0)
    ap.add_argument("--n-components", type=int, default=64)
    ap.add_argument("--min-frames", type=int, default=100,
                    help="causal frames before this: centering-only fallback")
    ap.add_argument("--frozen-k", type=int, nargs="*", default=[100, 300, 600])
    ap.add_argument("--ema-halflife", type=float, default=200.0)
    ap.add_argument("--subset-stride", type=int, default=3)
    ap.add_argument("--grid", type=int, default=56)
    ap.add_argument("--window", type=int, default=200)
    ap.add_argument("--window-step", type=int, default=10)
    args = ap.parse_args()

    # imports at call time (classroom_pipeline is being edited concurrently;
    # the four names below are the frozen API surface)
    from vsa_cognitive_mapping.classroom_pipeline import (
        ClassroomEncoders, _load_embeddings, load_poses_interpolated)
    from vsa_cognitive_mapping.vsa import pca_components

    frame_idx, ts, emb32 = _load_embeddings(args.out_dir)
    emb = emb32.astype(np.float64)
    N = len(ts)
    x, y, _psi = load_poses_interpolated(ts)
    k = min(args.n_components, emb.shape[1], N - 1)
    print(f"{N} frames, {emb.shape[1]}-d embeddings, k={k} components, "
          f"hd={args.hd_dim}")

    # ---- whitened features per condition ---------------------------------
    print("oracle: offline pca_components (whole-walk stats) ...")
    feats_oracle, _ = pca_components(emb, n_components=k)
    print("causal: streaming Welford pass (per-frame eigh) ...")
    feats_causal, frozen = causal_whiten_welford(
        emb, k, args.min_frames, set(args.frozen_k), args.seed)
    print(f"ema: streaming EMA pass (half-life {args.ema_halflife:g}) ...")
    feats_ema = causal_whiten_ema(emb, k, args.min_frames,
                                  args.ema_halflife, args.seed)
    feats_frozen = {}
    for K in args.frozen_k:
        V, inv_std, mu = frozen[K]
        feats_frozen[K] = ((emb - mu) @ V) * inv_std
    Vf, inv_stdf, muf = frozen["final"]
    feats_final = ((emb - muf) @ Vf) * inv_stdf

    # ---- shared phasor projection + position codes -----------------------
    z_oracle, W = project_content(feats_oracle, args.hd_dim, None, args.seed)
    z_causal, _ = project_content(feats_causal, args.hd_dim, W, args.seed)
    z_ema, _ = project_content(feats_ema, args.hd_dim, W, args.seed)
    z_frozen = {K: project_content(f, args.hd_dim, W, args.seed)[0]
                for K, f in feats_frozen.items()}
    z_final, _ = project_content(feats_final, args.hd_dim, W, args.seed)

    enc = ClassroomEncoders(args.hd_dim, args.seed + 100,
                            args.pos_length_scale, args.time_length_scale)
    # vectorized ctx_pos: Bx**v is values**v = exp(v * principal log), exactly
    logx = np.log(enc.Bx.values)
    logy = np.log(enc.By.values)
    l = args.pos_length_scale
    S = np.arange(0, N, args.subset_stride)
    ctx_S = np.exp(np.outer(x[S] / l, logx) + np.outer(y[S] / l, logy))

    pad = 0.5
    gx = np.linspace(x.min() - pad, x.max() + pad, args.grid)
    gy = np.linspace(y.min() - pad, y.max() + pad, args.grid)
    GX, GY = np.meshgrid(gx, gy)
    grid_xy = np.stack([GX.ravel(), GY.ravel()], axis=1)
    grid_flat = np.exp(np.outer(grid_xy[:, 0] / l, logx) +
                       np.outer(grid_xy[:, 1] / l, logy)).astype(np.complex64)

    thirds = [(f"third_{name}", np.where((S >= lo) & (S < hi))[0])
              for name, lo, hi in [("early", 0, N // 3),
                                   ("mid", N // 3, 2 * N // 3),
                                   ("late", 2 * N // 3, N + 1)]]

    # ---- recall per condition --------------------------------------------
    conds = [("oracle", z_oracle, None), ("causal", z_causal, thirds),
             ("ema", z_ema, thirds)]
    conds += [(f"frozen-{K}", z_frozen[K], thirds) for K in args.frozen_k]
    results = {}
    for name, z, th in conds:
        r = recall_eval(z[S], ctx_S, x[S], y[S], grid_flat, grid_xy, th)
        results[name] = r
        print(f"{name:>10}:  top1={r['top1_acc']:.3f}  "
              f"L1(content-match)={r['L1_content_m']:.3f} m  "
              f"L1(grid-decode)={r['L1_grid_m']:.3f} m")
        if th is not None:
            for tname, _ in th:
                rt = r[tname]
                print(f"{'':>12}{tname}: top1={rt['top1_acc']:.3f}  "
                      f"L1(content)={rt['L1_content_m']:.3f} m  "
                      f"L1(grid)={rt['L1_grid_m']:.3f} m")

    # code drift: causal code at storage time vs re-encoded with final stats
    drift = np.real(np.sum(z_causal * np.conj(z_final), axis=1)) / args.hd_dim
    print("code drift (sim of causal code at t vs re-encoded with final stats): "
          f"early={drift[:N // 3].mean():.3f}  "
          f"mid={drift[N // 3:2 * N // 3].mean():.3f}  "
          f"late={drift[2 * N // 3:].mean():.3f}")

    # ---- isotropy trajectory --------------------------------------------
    iso_raw_all = overall_isotropy(emb)
    iso_oracle_all = overall_isotropy(feats_oracle)
    c_raw, v_raw = sliding_isotropy(emb, args.window, args.window_step)
    c_cau, v_cau = sliding_isotropy(feats_causal, args.window, args.window_step)
    c_ora, v_ora = sliding_isotropy(feats_oracle, args.window, args.window_step)
    print(f"isotropy (mean |off-diag| cosine, whole walk): raw={iso_raw_all:.3f}  "
          f"oracle={iso_oracle_all:.3f}")
    gap = v_cau - v_ora
    conv = None                     # absolute: within 0.02 of oracle, sustained
    for i in range(len(c_cau)):
        if np.all(np.abs(gap[i:]) < 0.02):
            conv = int(c_cau[i])
            break
    conv_rel = None                 # relative: below 1.5x oracle, sustained
    for i in range(len(c_cau)):
        if np.all(v_cau[i:] < 1.5 * v_ora[i:]):
            conv_rel = int(c_cau[i])
            break
    print(f"causal isotropy convergence: within |0.02| of oracle from "
          f"window-centre t={conv} (sustained); below 1.5x oracle from "
          f"t={conv_rel}; final windowed causal={v_cau[-1]:.3f} vs "
          f"oracle={v_ora[-1]:.3f} (gap {gap[-1]:+.4f})")
    qs = np.linspace(0, len(c_cau) - 1, 6).astype(int)
    print("windowed isotropy samples (t: raw / causal / oracle): "
          + "  ".join(f"{c_cau[i]}: {v_raw[i]:.2f}/{v_cau[i]:.3f}/{v_ora[i]:.3f}"
                      for i in qs))

    # ---- figure ----------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2, figsize=(13.5, 4.6), constrained_layout=True)
    ax = axs[0]
    ax.plot(c_raw, v_raw, color="0.55", lw=1.6, label="raw embeddings")
    ax.plot(c_cau, v_cau, color="#b9772a", lw=1.8, label="causal whitening")
    ax.plot(c_ora, v_ora, color="#0d7d88", lw=1.8, label="oracle whitening")
    ax.axhline(iso_oracle_all, color="#0d7d88", lw=0.9, ls="--", alpha=0.6)
    ax.axvline(args.min_frames, color="0.75", lw=1.0, ls=":",
               label=f"min-frames={args.min_frames}")
    if conv is not None:
        ax.axvline(conv, color="#b9772a", lw=1.0, ls=":",
                   label=f"converged ~t={conv}")
    ax.set_yscale("log")
    ax.set_xlabel("frame t (sliding 200-frame window centre)")
    ax.set_ylabel("mean |off-diag| cosine (log)")
    ax.set_title("isotropy of whitened content over the walk")
    ax.legend(fontsize=8)

    ax = axs[1]
    names = ["oracle", "causal", "causal\nearly", "causal\nmid", "causal\nlate"]
    vals = [results["oracle"]["L1_grid_m"], results["causal"]["L1_grid_m"],
            results["causal"]["third_early"]["L1_grid_m"],
            results["causal"]["third_mid"]["L1_grid_m"],
            results["causal"]["third_late"]["L1_grid_m"]]
    accs = [results["oracle"]["top1_acc"], results["causal"]["top1_acc"],
            results["causal"]["third_early"]["top1_acc"],
            results["causal"]["third_mid"]["top1_acc"],
            results["causal"]["third_late"]["top1_acc"]]
    colors = ["#0d7d88", "#b9772a", "#d9a05b", "#d9a05b", "#d9a05b"]
    for K in args.frozen_k:
        names.append(f"frozen\nK={K}")
        vals.append(results[f"frozen-{K}"]["L1_grid_m"])
        accs.append(results[f"frozen-{K}"]["top1_acc"])
        colors.append("#4a7b4f")
    names.append("ema")
    vals.append(results["ema"]["L1_grid_m"])
    accs.append(results["ema"]["top1_acc"])
    colors.append("#7a5aa0")
    xs_bar = np.arange(len(names))
    ax.bar(xs_bar, vals, color=colors, alpha=0.9)
    for xi, v, a in zip(xs_bar, vals, accs):
        ax.text(xi, v + 0.02 * max(vals), f"{v:.2f} m\nacc {a:.2f}",
                ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(xs_bar)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylim(0, max(vals) * 1.25)
    ax.set_ylabel("L1 position error, grid decode (m)")
    ax.set_title("episodic position recall per whitening condition "
                 f"(stride-{args.subset_stride} split, |S|={len(S)})")

    path = os.path.join(args.out_dir, "online_whitening.png")
    fig.savefig(path, dpi=130)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
