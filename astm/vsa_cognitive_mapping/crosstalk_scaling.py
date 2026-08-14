"""Crosstalk scaling: define the public 365x/27x figure and measure its
N-dependence (coherent O(N) vs incoherent O(sqrt N) claim).

Metric (THE definition — the quoted 365x/27x had none in code)
--------------------------------------------------------------
Bundle N bound pairs from the real classroom walk (YOLO frame embeddings ->
content phasors c_i; LIO-SAM robot pose -> position phasors p_i, the same
ClassroomEncoders FPE codes the pipeline uses):

    M_N = (1/N) * sum_{i in S_N}  c_i (x) p_i ,   S_N = random N-subsample

For each probe i in a fixed-size probe subset of S_N, unbind its position
(positions are unit-modulus, so (/) p_i = * conj(p_i)) and correlate the
residual with every stored content:

    r_i         = M_N (/) p_i
    ontarget_i  = Re< r_i, c_i >/D           (the recalled signal)
    offtarget_i = mean_{j != i} | Re< r_i, c_j >/D |   (interference)

    crosstalk(N) = mean_i offtarget_i / mean_i ontarget_i

(ratio of means, the stable "simpler equivalent": under heavy interference
individual ontarget_i can cross zero and a mean-of-ratios diverges).

Secondary metric — because the MEASURED ontarget also contains the coherent
interference mass, the growth law is cleanest against the exact per-item
signal, which for the mean bundle is exactly 1/N (Re<c_i,c_i>/D = 1):

    chi(N) = mean_i offtarget_i / (1/N) = N * mean_i offtarget_i

chi(N) is the interference amplitude in units of one stored item's signal;
this is the quantity the O(N)-coherent vs O(sqrt N)-incoherent claim is
actually about:

  * RAW content: the frame embeddings are anisotropic (mean pairwise cosine
    ~0.9), so after projection the content phasors share a coherent
    component; interference terms add IN PHASE -> chi ~ N^1 * rho
    (coherent).
  * PCA-WHITENED content (the pipeline's fix): content phasors are
    quasi-orthogonal; interference terms add with random phases ->
    chi ~ sqrt(N)-ish (incoherent), if residual content correlations
    (consecutive frames genuinely look alike) are negligible.

Both metrics are computed and fitted; the figure shows both panels.

Both content variants come from outputs/classroom/embeddings.pt through the
exact pipeline path (PCA-64 standardized scores vs raw embeddings, then
random_project_to_phasor at hd=8192, seed 0). Whitening statistics are fit
on the FULL walk once (as the pipeline's build stage does), then rows are
subsampled.

NOTE on the sweep upper end: the task/plan quotes "2478 (all frames)", but
embeddings.pt holds the STRIDE-2 embed of the 2478-frame walk = 1239
embedded frames. 1239 is therefore the honest "all frames" endpoint; the
endpoint ratio is reported at N=1239.

Outputs: fitted log-log slopes per content type, the N=1239 endpoint
crosstalk values (compare against the quoted 365x/27x), and
outputs/classroom/crosstalk_scaling.png.

Usage (from repo root, needs outputs/classroom/embeddings.pt + HF pose cache):
    python vsa_cognitive_mapping/crosstalk_scaling.py
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vsa_cognitive_mapping.vsa import (  # noqa: E402
    pca_components, random_project_to_phasor,
)
from vsa_cognitive_mapping.classroom_pipeline import (  # noqa: E402  (read-only)
    ClassroomEncoders, _load_embeddings, load_poses_interpolated,
)

DEFAULT_OUT = os.path.join("outputs", "classroom")


def content_variants(emb, hd, seed, n_pca):
    """Ordered dict of content phasor arrays, each (N, hd) complex128,
    all through the SAME random_project_to_phasor(d=hd, seed) path.

    Decomposes the whitening gain into its ingredients:
      raw      : emb as-is (the pre-fix path)
      centred  : emb - column mean  (covariance spectrum bit-identical to raw)
      zscored  : centred / per-dim std  (diagonal scaling, no rotation)
      whitened : PCA-n_pca standardized scores (the pipeline's fix)
    """
    def proj(X):
        z, _ = random_project_to_phasor(
            torch.from_numpy(np.ascontiguousarray(X)).float(), d=hd, seed=seed)
        return z.numpy().astype(np.complex128)

    centred = emb - emb.mean(axis=0, keepdims=True)
    zscored = centred / (emb.std(axis=0, keepdims=True) + 1e-9)
    k = min(n_pca, emb.shape[1], emb.shape[0] - 1)
    scores, _ = pca_components(emb.astype(np.float64), n_components=k)
    return {"raw": proj(emb),
            "centred": proj(centred),
            "zscored": proj(zscored),
            "whitened": proj(scores)}


def crop_frame_descriptors(out_dir, fname):
    """Per-frame content from a per-DETECTION crop-embedding file.

    The 365x/22x figures were all measured on ``embeddings.pt`` -- YOLOv8n's
    whole-frame penultimate feature, which the encoder comparison showed is the
    most anisotropic representation available. Quoting a whitening multiplier
    measured on the worst encoder overstates it, so the metric has to be
    re-runnable on any backbone.

    A frame's descriptor is the mean of its crops, L2-normalised, matching how
    the video exporter and the localisation memory build content. Timestamps
    come from ``crop_embeddings.pt`` (the YOLO crop pass records them; the
    other encoders' files carry only ``frame_idx``).
    """
    d = torch.load(os.path.join(out_dir, fname), weights_only=False)
    X = d["embedding"].numpy().astype(np.float64)
    fr = d["frame_idx"].numpy()

    base = torch.load(os.path.join(out_dir, "crop_embeddings.pt"), weights_only=False)
    ts_of = {}
    for f, t in zip(base["frame_idx"].numpy(), base["timestamp_ns"].numpy()):
        ts_of.setdefault(int(f), int(t))

    uf, inv = np.unique(fr, return_inverse=True)
    Dsc = np.zeros((len(uf), X.shape[1]), dtype=np.float64)
    np.add.at(Dsc, inv, X)
    Dsc /= np.maximum(np.bincount(inv, minlength=len(uf)).astype(np.float64)[:, None], 1)
    Dsc /= (np.linalg.norm(Dsc, axis=1, keepdims=True) + 1e-12)

    keep = np.array([i for i, f in enumerate(uf) if int(f) in ts_of])
    ts = np.array([ts_of[int(uf[i])] for i in keep], dtype=np.int64)
    o = np.argsort(ts)
    return ts[o], Dsc[keep][o]


def mean_offdiag_cos(C, n_sub, rng):
    """Mean off-diagonal Re<c_i,c_j>/D over an n_sub-row subsample."""
    idx = rng.choice(C.shape[0], size=min(n_sub, C.shape[0]), replace=False)
    S = np.real(C[idx] @ np.conj(C[idx]).T) / C.shape[1]
    off = S[~np.eye(len(idx), dtype=bool)]
    return float(off.mean()), float(np.abs(off).mean())


def crosstalk(C, P, sel, n_probe, rng):
    """crosstalk(N) for one subsample sel (indices into C/P rows).

    Returns (ratio, mean_on, mean_off)."""
    D = C.shape[1]
    M = (C[sel] * P[sel]).mean(axis=0)                      # (D,)
    probes = sel if len(sel) <= n_probe else \
        sel[rng.choice(len(sel), size=n_probe, replace=False)]
    R = M[None, :] * np.conj(P[probes])                      # residuals (m, D)
    S = np.real(R @ np.conj(C[probes]).T) / D                # (m, m) sims
    on = np.diag(S).copy()
    off = np.abs(S[~np.eye(len(probes), dtype=bool)]
                 .reshape(len(probes), len(probes) - 1)).mean(axis=1)
    mean_on, mean_off = float(on.mean()), float(off.mean())
    return mean_off / mean_on, mean_on, mean_off


def fit_slope(Ns, vals):
    """Least-squares slope of log(vals) vs log(Ns)."""
    lx, ly = np.log(np.asarray(Ns, float)), np.log(np.asarray(vals, float))
    A = np.vstack([lx, np.ones_like(lx)]).T
    (m, b), *_ = np.linalg.lstsq(A, ly, rcond=None)
    return float(m), float(b)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--hd-dim", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-pca", type=int, default=64)
    ap.add_argument("--pos-length-scale", type=float, default=0.75)
    ap.add_argument("--n-probe", type=int, default=200,
                    help="probes per (N, seed) config")
    ap.add_argument("--sub-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--no-figure", action="store_true")
    ap.add_argument("--content", default="frames",
                    help="'frames' = embeddings.pt (YOLOv8n whole-frame feature, "
                         "what the 365x/22x figures were measured on), or a "
                         "crop-embedding filename to use per-frame descriptors "
                         "from another backbone, e.g. crop_embeddings_dinov2.pt")
    ap.add_argument("--tag", default=None,
                    help="label for printed output and the figure filename")
    ap.add_argument("--odom-config", default="odometry_lio_sam",
                    help="pass school_run1_odometry_lio_sam when --out-dir "
                         "points at the scale-up sequence")
    args = ap.parse_args()

    if args.content == "frames":
        _fi, ts, emb = _load_embeddings(args.out_dir)
        src = "embeddings.pt (YOLOv8n whole-frame, stride-2 of the 2478-frame walk)"
    else:
        ts, emb = crop_frame_descriptors(args.out_dir, args.content)
        src = f"{args.content} (per-detection crops -> per-frame mean descriptor)"
    N_all = len(ts)
    x, y, _psi = load_poses_interpolated(ts, odom_config=args.odom_config)
    print(f"{N_all} frames of content from {src}; hd={args.hd_dim}, "
          f"input dim={emb.shape[1]}")

    variants = content_variants(emb, args.hd_dim, args.seed, args.n_pca)
    enc = ClassroomEncoders(args.hd_dim, args.seed + 100,
                            args.pos_length_scale, 20.0)
    P = np.empty((N_all, args.hd_dim), np.complex128)
    for i in range(N_all):
        P[i] = enc.ctx_pos(float(x[i]), float(y[i])).values

    rng0 = np.random.RandomState(999)
    for name, C in variants.items():
        mo, moa = mean_offdiag_cos(C, 300, rng0)
        print(f"  {name:8s} content phasors: mean off-diag cos {mo:+.4f} "
              f"(mean |cos| {moa:.4f})")
    # embedding-space coherence for reference (the quoted ~0.88)
    e = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    idx = rng0.choice(N_all, 300, replace=False)
    Se = e[idx] @ e[idx].T
    print(f"  raw EMBEDDING mean off-diag cos: "
          f"{Se[~np.eye(300, dtype=bool)].mean():+.4f}")

    # extended past 1600 so school_run1 (~10k frame descriptors) is sampled
    # along the way rather than jumping straight from 1600 to the endpoint
    Ns = [n for n in (50, 100, 200, 400, 800, 1600, 3200, 6400) if n < N_all] + [N_all]
    results = {name: {} for name in variants}
    for name, C in variants.items():
        for n in Ns:
            vals = []
            for s in args.sub_seeds:
                rng = np.random.RandomState(1000 * s + n)
                sel = (np.arange(N_all) if n >= N_all
                       else np.sort(rng.choice(N_all, size=n, replace=False)))
                ratio, on, off = crosstalk(C, P, sel, args.n_probe, rng)
                vals.append((ratio, on, off))
            r = np.array([v[0] for v in vals])
            chi = np.array([n * v[2] for v in vals])
            results[name][n] = {"ratio_mean": float(r.mean()),
                                "ratio_std": float(r.std()),
                                "chi_mean": float(chi.mean()),
                                "chi_std": float(chi.std()),
                                "on": float(np.mean([v[1] for v in vals])),
                                "off": float(np.mean([v[2] for v in vals]))}
            print(f"  {name:8s} N={n:5d}: ratio={r.mean():6.2f}x "
                  f"(+/-{r.std():.2f})  chi=N*off={chi.mean():8.2f}x "
                  f"(+/-{chi.std():.2f})  mean_on={results[name][n]['on']:+.3e}")

    print("\n---- log-log fits ~ N^slope ----")
    slopes = {}
    for metric in ("ratio_mean", "chi_mean"):
        tag = "ratio (off/measured-on)" if metric == "ratio_mean" else \
            "chi (off/per-item-signal)"
        for name in variants:
            m, b = fit_slope(Ns, [results[name][n][metric] for n in Ns])
            slopes[(name, metric)] = m
            print(f"  {tag:26s} {name:8s}: slope {m:+.3f}  "
                  f"(coherent claim ~1.0, incoherent claim ~0.5)")

    n_end = N_all
    print(f"\n---- endpoint N={n_end} (all embedded frames; quoted figures "
          f"were 365x raw / 27x whitened) ----")
    for metric, tag in (("ratio_mean", "ratio"), ("chi_mean", "chi")):
        r_raw = results["raw"][n_end][metric]
        parts = []
        for name in variants:
            v = results[name][n_end][metric]
            part = f"{name} {v:8.2f}x"
            if name != "raw":
                part += f" (raw/{name} {r_raw / v:.1f}x)"
            parts.append(part)
        print(f"  {tag:6s}: " + "   ".join(parts))

    if not args.no_figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
        panels = [("ratio_mean", "ratio_std",
                   "crosstalk ratio = mean|offtarget| / mean measured ontarget",
                   "(a) task metric: recall-relative (measured ontarget "
                   "absorbs the coherent mass)"),
                  ("chi_mean", "chi_std",
                   "chi(N) = N * mean|offtarget|  (per-item-signal units)",
                   "(b) growth-law metric: interference vs one stored "
                   "item's signal")]
        colors = {"raw": "#b9772a", "centred": "#8a5fbf",
                  "zscored": "#3d7dbf", "whitened": "#0d7d88"}
        for ax, (mk, sk, ylab, title) in zip(axs, panels):
            for name in variants:
                color = colors[name]
                mu = np.array([results[name][n][mk] for n in Ns])
                sd = np.array([results[name][n][sk] for n in Ns])
                ax.errorbar(Ns, mu, yerr=sd, fmt="o-", color=color, lw=1.6,
                            capsize=3,
                            label=f"{name}: slope {slopes[(name, mk)]:+.2f} "
                                  f"(endpoint {mu[-1]:.2f}x)")
                m, b = fit_slope(Ns, mu)
                xs = np.array([Ns[0], Ns[-1]], float)
                ax.plot(xs, np.exp(b) * xs ** m, "--", color=color, lw=0.9,
                        alpha=0.6)
            for sref, ls in ((1.0, ":"), (0.5, "-.")):
                anchor = results["whitened"][Ns[0]][mk]
                xs = np.array([Ns[0], Ns[-1]], float)
                ax.plot(xs, anchor * (xs / Ns[0]) ** sref, ls, color="0.6",
                        lw=0.9, label=f"reference slope {sref:g}")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("N bundled (content (x) position) pairs")
            ax.set_ylabel(ylab)
            ax.set_title(title, fontsize=9)
            ax.grid(alpha=0.3, which="both")
            ax.legend(fontsize=8)
        lbl = args.tag or ("yolov8n frames" if args.content == "frames"
                           else args.content.replace("crop_embeddings_", "")
                                            .replace(".pt", "") + " crops")
        fig.suptitle(f"Crosstalk scaling — content: {lbl}, hd={args.hd_dim}, "
                     f"3 subsample seeds; endpoint N={n_end}", fontsize=10)
        # tag the filename: a fixed name silently overwrote the YOLO figure when
        # the same metric was re-run on a different backbone
        suffix = "" if args.content == "frames" else "_" + lbl.replace(" ", "_")
        path = os.path.join(args.out_dir, f"crosstalk_scaling{suffix}.png")
        fig.savefig(path, dpi=140)
        print(f"saved {path}")


if __name__ == "__main__":
    main()
