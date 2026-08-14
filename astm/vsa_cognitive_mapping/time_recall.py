"""Time recall + two-robot merge on a sequence whose POSITIONS are unusable.

`school_run1`'s LIO-SAM positions are void (23 km of "path" in 455 s), but its
timestamps are fine — so the honest through-memory test there is temporal:
store ``content ⊗ ctx_time(t)`` and ask, with a held-out view, **when** it was
seen. Same blocked-split discipline, same chance/oracle/constant brackets as
`heldout_eval`, decoded over a time grid instead of a position grid.

This also carries the merge demo on real data with a valid factor: split the
stored frames into "robot A" (first half of the walk) and "robot B" (second
half), build each memory separately, add them, and check (a) bit-level
agreement with the jointly-built memory and (b) that every query returns the
identical answer either way.

    python -m vsa_cognitive_mapping.time_recall --out-dir outputs/school_run1
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vsa_cognitive_mapping.vsa import random_project_to_phasor  # noqa: E402
from vsa_cognitive_mapping.classroom_pipeline import ClassroomEncoders  # noqa: E402
from vsa_cognitive_mapping.heldout_eval import frame_descriptors, treat, make_split  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="outputs/school_run1")
    ap.add_argument("--encoder", default="crop_embeddings_dinov2.pt")
    ap.add_argument("--treatments", nargs="+",
                    default=["raw", "centred", "zscored", "whitened128"])
    ap.add_argument("--hd-dim", type=int, default=4096)
    ap.add_argument("--time-cells", type=int, default=400)
    ap.add_argument("--time-l", type=float, default=20.0,
                    help="the pipeline's time length scale (seconds-equivalent)")
    ap.add_argument("--frac", type=float, default=0.3)
    ap.add_argument("--gap", type=int, default=45)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()

    uf, E = frame_descriptors(args.out_dir, args.encoder)
    # Timestamps come from the SAME file as the embeddings. Reading them from
    # crop_embeddings.pt broke for frame-level frontends (KeyError): the crop
    # file only covers frames WITH detections, while e.g. the VPR file covers
    # every frame. Our schema always carries timestamp_ns, so use it directly.
    base = torch.load(os.path.join(args.out_dir, args.encoder),
                      weights_only=False)
    ts_of = {}
    for f, t in zip(base["frame_idx"].numpy(), base["timestamp_ns"].numpy()):
        ts_of.setdefault(int(f), int(t))
    tsec = np.array([ts_of[int(f)] for f in uf], np.float64) / 1e9
    tsec -= tsec.min()
    n, span = len(uf), tsec.max()
    print(f"{n} frames over {span:.0f} s ({args.encoder}); positions on this "
          f"sequence are VOID — time is the valid factor. hd={args.hd_dim}")

    enc = ClassroomEncoders(args.hd_dim, 0, 0.75, args.time_l)
    pht = np.angle(enc.Bt.values)
    T = np.exp(1j * pht[None, :] * (tsec[:, None] / args.time_l))    # (n, hd)
    tg = np.linspace(0, span, args.time_cells)
    Tg = np.exp(1j * pht[None, :] * (tg[:, None] / args.time_l)).astype(np.complex64)

    print(f"\n=== held-out TIME recall (blocked, {len(args.seeds)} seeds; "
          f"'when was this view seen?') ===")
    hdr = (f"{'treatment':<13}{'oracle s':>9}{'const s':>9}{'median s':>10}"
           f"{'signal':>9}{'sig_sd':>8}")
    print(hdr); print("-" * len(hdr))
    results = []
    for tr in args.treatments:
        Et = treat(E, tr)
        z, _ = random_project_to_phasor(
            torch.from_numpy(np.ascontiguousarray(Et)).float(), d=args.hd_dim, seed=0)
        Z = z.numpy().astype(np.complex128)
        sigs, meds, last = [], [], None
        for s in args.seeds:
            rng = np.random.RandomState(1000 + s)
            mem, qry = make_split(uf, "blocked", args.frac, args.gap, rng)
            M = (Z[mem] * T[mem]).mean(axis=0)
            M /= max(np.abs(M).max(), 1e-12)
            R = M[None, :] / Z[qry]
            S = (R @ np.conj(Tg).T).real
            t_hat = tg[S.argmax(1)]
            err = np.abs(t_hat - tsec[qry])
            tm, tq = tsec[mem], tsec[qry]
            oracle = np.median(np.abs(tq[:, None] - tm[None, :]).min(axis=1))
            const = np.median(np.abs(tq - np.median(tm)))
            med = float(np.median(err))
            sigs.append((const - med) / max(const - oracle, 1e-9))
            meds.append(med)
            last = (oracle, const)
        results.append(dict(treatment=tr, oracle_s=last[0], const_s=last[1],
                            median_s=float(np.mean(meds)),
                            signal=float(np.mean(sigs)),
                            signal_std=float(np.std(sigs))))
        print(f"{tr:<13}{last[0]:>9.1f}{last[1]:>9.1f}{np.mean(meds):>10.1f}"
              f"{np.mean(sigs):>9.1%}{np.std(sigs):>8.1%}")

    # ---- two-robot merge, on the best treatment --------------------------
    best = max(results, key=lambda r: r["signal"])["treatment"]
    print(f"\n=== two-robot merge (treatment = {best}) ===")
    Et = treat(E, best)
    z, _ = random_project_to_phasor(
        torch.from_numpy(np.ascontiguousarray(Et)).float(), d=args.hd_dim, seed=0)
    Z = z.numpy().astype(np.complex128)
    rng = np.random.RandomState(1000)
    mem, qry = make_split(uf, "blocked", args.frac, args.gap, rng)
    mi = np.where(mem)[0]
    half = len(mi) // 2
    A, B = mi[:half], mi[half:]              # first / second half of the walk
    print(f"robot A: {len(A)} frames, t {tsec[A].min():.0f}-{tsec[A].max():.0f} s; "
          f"robot B: {len(B)} frames, t {tsec[B].min():.0f}-{tsec[B].max():.0f} s")

    def build(idx):
        return (Z[idx] * T[idx]).sum(axis=0), len(idx)

    SA, nA = build(A)
    SB, nB = build(B)
    SJ, nJ = build(mi)
    M_merged = (SA + SB) / (nA + nB)
    M_joint = SJ / nJ
    exact = float(np.abs(M_merged - M_joint).max())
    print(f"max |merged − joint| over {args.hd_dim} dims: {exact:.2e}  "
          f"(floating-point noise)")

    qsel = np.where(qry)[0][np.linspace(0, qry.sum() - 1, 24).astype(int)]
    for name, M in (("joint", M_joint), ("merged", M_merged)):
        Mn = M / max(np.abs(M).max(), 1e-12)
        S = ((Mn[None, :] / Z[qsel]) @ np.conj(Tg).T).real
        if name == "joint":
            S_j, t_j = S, tg[S.argmax(1)]
        else:
            same = int((tg[S.argmax(1)] == t_j).sum())
            print(f"query battery ({len(qsel)} held-out views): identical answers "
                  f"{same}/{len(qsel)}; max field difference "
                  f"{np.abs(S - S_j).max():.2e}")
    err_j = np.abs(t_j - tsec[qsel])
    print(f"merged-memory time recall on the battery: median "
          f"{np.median(err_j):.1f} s over a {span:.0f} s walk")

    out = os.path.join(args.out_dir, "time_recall.json")
    with open(out, "w") as f:
        json.dump(dict(encoder=args.encoder, n=n, span_s=float(span),
                       results=results, merge_exactness=exact,
                       merge_treatment=best), f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
