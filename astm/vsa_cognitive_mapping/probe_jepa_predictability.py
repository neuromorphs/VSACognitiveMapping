"""VSA-JEPA predictability probe: is the next frame's VSA content atom
predictable from the current one via algebraic path integration?

The planned VSA-JEPA architecture replaces a learned predictor with a bind:

    z_hat_{t+1} = z_t (x) T(Delta_pose),   z_t = s_t (x) ctx_pos(p_t)

where s_t is the whitened-content phasor of frame t (PCA-64 -> random
phasor projection, identical to the classroom pipeline's build stage) and
T(Delta) = Bx**(dx/l) * By**(dy/l) is the FPE transition operator built from
the SAME position bases the pipeline uses (ClassroomEncoders, seed 100).
Because ctx_pos is a world-frame FPE group homomorphism,
ctx_pos(p_{t+1}) = ctx_pos(p_t) (x) T(Delta_world) holds exactly — world
deltas, no body-frame rotation needed.

Measurements over all consecutive pairs of outputs/classroom/embeddings.pt
(1,239 stride-2 frames, poses via LIO-SAM interpolation):

    (a) content persistence      cos(s_t, s_{t+1})          [baseline]
    (b) place-code prediction    cos(ctx_t (x) T_t, ctx_{t+1})   [~1 sanity]
    (c) bound-state prediction   cos(z_t (x) T_t, z_{t+1})  [THE question]
    (d) content-only null        cos(s_t, s_{t'}), same derangement of t'
    (e) horizon decay            (c) and cos(s_t, s_{t+h}) for h=1,2,4,8,16

Note the algebra: with unit-modulus ctx and exact transport (b)=1, (c)
collapses to (a) IDENTICALLY — so (c) can never "beat" a null by more than
persistence does. The honest framing is the persistence baseline: (c) vs
(a) shows the bind adds nothing beyond persistence, and (d) — the
content-only null under the same derangement — shows how much of the
persistence is walk-wide content coherence rather than temporal adjacency.
(The previous shuffled BOUND-STATE null was vacuous: it mostly measured
position-code decorrelation, guaranteeing a large margin by construction.)

Everything is vectorized over frames and hd dims; the whole run is seconds.

Example:
    python vsa_cognitive_mapping/probe_jepa_predictability.py
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
from vsa_cognitive_mapping.classroom_pipeline import (  # noqa: E402
    ClassroomEncoders, _load_embeddings, load_poses_interpolated,
)

DEFAULT_OUT = os.path.join("outputs", "classroom")


# --------------------------------------------------------------------------
# building blocks
# --------------------------------------------------------------------------

def content_phasors(emb: np.ndarray, hd_dim: int, seed: int, n_pca: int) -> np.ndarray:
    """PCA-whiten (standardized scores, as the pipeline's build stage) then
    random-project to unit-modulus phasors. Returns (N, hd_dim) complex128."""
    k = min(n_pca, emb.shape[1], emb.shape[0] - 1)
    scores, _ = pca_components(emb.astype(np.float64), n_components=k)
    z, _ = random_project_to_phasor(
        torch.from_numpy(np.ascontiguousarray(scores)).float(), d=hd_dim, seed=seed)
    return z.numpy().astype(np.complex128)


def pos_phase_table(enc: ClassroomEncoders, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """theta[t] such that ctx_pos(x_t, y_t) = exp(1j * theta[t]).

    Uses the principal argument of the Bx/By base phasors, which is exactly
    what numpy's complex ** (Phasor.fpe) exponentiates, so
    exp(1j * theta[t]) == enc.ctx_pos(x[t], y[t]).values to float precision.
    Shape (N, hd_dim), real float64.
    """
    phx = np.angle(enc.Bx.values)
    phy = np.angle(enc.By.values)
    return (np.outer(x / enc.pos_l, phx) + np.outer(y / enc.pos_l, phy))


def row_cos(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between complex arrays (M, D) -> (M,)."""
    num = np.real(np.einsum("ij,ij->i", np.conj(u), v))
    return num / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1))


def stats(v: np.ndarray) -> dict:
    return {"mean": float(np.mean(v)), "median": float(np.median(v)),
            "p10": float(np.percentile(v, 10))}


# --------------------------------------------------------------------------
# probe
# --------------------------------------------------------------------------

def run_probe(args):
    frame_idx, ts, emb = _load_embeddings(args.out_dir)
    N = len(ts)
    x, y, psi = load_poses_interpolated(ts)
    print(f"{N} frames; hd={args.hd_dim}; pca={args.n_pca}; "
          f"pos_length_scale={args.pos_length_scale}")

    # content atoms s_t (pipeline-identical whitened content, seed 0)
    s = content_phasors(emb, args.hd_dim, args.seed, args.n_pca)

    # position phases -> ctx and transition operators come from the same table
    enc = ClassroomEncoders(args.hd_dim, args.enc_seed,
                            args.pos_length_scale, args.time_length_scale)
    theta = pos_phase_table(enc, x, y)          # (N, hd)
    ctx = np.exp(1j * theta)                    # ctx_pos(p_t)
    z = s * ctx                                 # bound state z_t

    rng = np.random.RandomState(args.shuffle_seed)
    results = {}

    # ---- (a) content-only persistence ------------------------------------
    a = row_cos(s[:-1], s[1:])
    results["a_content_persistence"] = stats(a)

    # ---- (b) place-code prediction (algebra sanity) -----------------------
    # T_t = exp(1j*(theta_{t+1}-theta_t)); ctx_t (x) T_t vs ctx_{t+1}
    T1 = np.exp(1j * (theta[1:] - theta[:-1]))
    b = row_cos(ctx[:-1] * T1, ctx[1:])
    results["b_place_code"] = stats(b)

    # ---- (c) bound-state prediction ---------------------------------------
    z_hat = z[:-1] * T1
    c = row_cos(z_hat, z[1:])
    results["c_bound_state"] = stats(c)

    # ---- (d) content-only null (same derangement) --------------------------
    # cos(s_t, s_t') for a deranged pairing t' != t: how similar is the
    # content atom to a WRONG frame's content? This is the null persistence
    # must beat; a bound-state null would only measure position-code
    # decorrelation and is vacuous by construction.
    n_pairs = N - 1
    perm = rng.permutation(n_pairs)
    fixed = np.where(perm == np.arange(n_pairs))[0]
    perm[fixed] = (perm[fixed] + 1) % n_pairs   # derangement: no t' == t
    d = row_cos(s[:-1], s[1:][perm])
    results["d_content_null"] = stats(d)

    # ---- (e) horizon decay -------------------------------------------------
    horizons = args.horizons
    curve_pred, curve_content, curve_null = [], [], []
    for h in horizons:
        Th = np.exp(1j * (theta[h:] - theta[:-h]))          # T(p_{t+h}-p_t)
        ch = row_cos(z[:-h] * Th, z[h:])                     # bound prediction
        ah = row_cos(s[:-h], s[h:])                          # content persistence
        m = N - h
        p = rng.permutation(m)
        fx = np.where(p == np.arange(m))[0]
        p[fx] = (p[fx] + 1) % m
        nh = row_cos(s[:-h], s[h:][p])          # content-only null at horizon h
        curve_pred.append(stats(ch))
        curve_content.append(stats(ah))
        curve_null.append(stats(nh))
    results["e_horizon"] = {"h": list(horizons),
                            "pred": curve_pred, "content": curve_content,
                            "null": curve_null}

    # ---- conclusion (persistence-baseline framing ONLY) ---------------------
    ma, mc, md = results["a_content_persistence"]["mean"], \
        results["c_bound_state"]["mean"], results["d_content_null"]["mean"]
    concl = (f"persistence baseline: transport is lossless (b mean="
             f"{results['b_place_code']['mean']:.4f}), so bound-state "
             f"prediction inherits content persistence by identity "
             f"(c={mc:.3f} vs a={ma:.3f}) — the bind adds NO predictive "
             f"information beyond persistence. Content-only null (same "
             f"derangement): {md:.3f}; adjacency contributes "
             f"{ma - md:.3f} of persistence above walk-wide content "
             f"coherence.")
    results["conclusion"] = concl

    return results, (a, b, c, d), (horizons, curve_pred, curve_content, curve_null)


def make_figure(dists, horizon_data, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a, b, c, d = dists
    horizons, curve_pred, curve_content, curve_null = horizon_data

    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.4), constrained_layout=True)

    bins = np.linspace(min(d.min(), a.min()) - 0.02, 1.02, 60)
    axs[0].hist(a, bins=bins, alpha=0.55, color="#0d7d88",
                label=f"(a) content persistence  mean={a.mean():.3f}")
    axs[0].hist(c, bins=bins, alpha=0.55, color="#b9772a",
                label=f"(c) bound-state prediction  mean={c.mean():.3f}")
    axs[0].hist(d, bins=bins, alpha=0.55, color="0.5",
                label=f"(d) content-only null  mean={d.mean():.3f}")
    axs[0].axvline(b.mean(), color="black", lw=1.2, ls="--",
                   label=f"(b) place-code sanity  mean={b.mean():.4f}")
    axs[0].set_xlabel("cosine similarity")
    axs[0].set_ylabel("pair count")
    axs[0].set_title("h=1 predictability: prediction vs baseline vs null")
    axs[0].legend(fontsize=8, loc="upper left")

    hp = np.array([s["mean"] for s in curve_pred])
    hc = np.array([s["mean"] for s in curve_content])
    hn = np.array([s["mean"] for s in curve_null])
    hp10 = np.array([s["p10"] for s in curve_pred])
    axs[1].plot(horizons, hp, "o-", color="#b9772a", lw=1.8,
                label="bound-state prediction cos(z_hat, z_{t+h})")
    axs[1].plot(horizons, hc, "s--", color="#0d7d88", lw=1.4,
                label="content persistence cos(s_t, s_{t+h})")
    axs[1].fill_between(horizons, hp10, hp, color="#b9772a", alpha=0.15,
                        label="prediction p10..mean")
    axs[1].plot(horizons, hn, "^:", color="0.4", lw=1.2,
                label="content-only null")
    axs[1].set_xscale("log", base=2)
    axs[1].set_xticks(list(horizons))
    axs[1].set_xticklabels([str(h) for h in horizons])
    axs[1].set_xlabel("horizon h (frames, stride-2 capture)")
    axs[1].set_ylabel("mean cosine similarity")
    axs[1].set_title("predictability decay vs horizon")
    axs[1].legend(fontsize=8)

    fig.suptitle("VSA-JEPA predictability probe — classroom walk, hd=8192",
                 fontsize=11)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"saved {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--hd-dim", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=0,
                    help="content projection seed (pipeline default)")
    ap.add_argument("--enc-seed", type=int, default=100,
                    help="ClassroomEncoders seed (pipeline uses seed+100=100)")
    ap.add_argument("--n-pca", type=int, default=64)
    ap.add_argument("--pos-length-scale", type=float, default=0.75)
    ap.add_argument("--time-length-scale", type=float, default=20.0)
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--shuffle-seed", type=int, default=12345)
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()

    results, dists, horizon_data = run_probe(args)

    print("\n---- distributions (mean / median / p10) ----")
    for key in ("a_content_persistence", "b_place_code",
                "c_bound_state", "d_content_null"):
        st = results[key]
        print(f"{key:>24}: {st['mean']:+.4f} / {st['median']:+.4f} / {st['p10']:+.4f}")
    print("\n---- horizon decay (mean cos) ----")
    e = results["e_horizon"]
    print(f"{'h':>4} {'prediction':>11} {'content':>9} {'null':>8}")
    for i, h in enumerate(e["h"]):
        print(f"{h:>4} {e['pred'][i]['mean']:>11.4f} "
              f"{e['content'][i]['mean']:>9.4f} {e['null'][i]['mean']:>8.4f}")
    print(f"\nCONCLUSION: {results['conclusion']}")

    if not args.no_figure:
        make_figure(dists, horizon_data,
                    os.path.join(args.out_dir, "jepa_predictability.png"))


if __name__ == "__main__":
    main()
