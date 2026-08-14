"""Absolute-terms place recall: CDF fractions, kNN ceiling, VSA argmax, top-k.

The standing metric after tracker (as): every recall claim reports the error
CDF (<=0.5 / <=1 / <=2 m and worst case) alongside any relative "signal"
number, and always next to the kNN ceiling on the SAME descriptors — the kNN
row is what separates "the memory is the bottleneck" from "the descriptor is".

    python -m vsa_cognitive_mapping.recall_cdf --out-dir outputs/classroom \
        --encoder crop_embeddings_dinov2.pt --treatment zscored --bind-yaw
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vsa_cognitive_mapping.sequences import load_sequence  # noqa: E402
from vsa_cognitive_mapping.heldout_eval import (  # noqa: E402
    frame_descriptors, treat, make_split)
from vsa_cognitive_mapping.classroom_pipeline import ClassroomEncoders  # noqa: E402
from vsa_cognitive_mapping.vsa import random_project_to_phasor  # noqa: E402


def topk_err(S, XYq, gx, gy, grid, k=3, sep=4):
    """Best error over the top-k well-separated modes of each similarity field."""
    errs = []
    for r, s in enumerate(S):
        idx = np.argsort(s)[::-1]
        picks, taken = [], []
        for j in idx:
            c = np.array([j % grid, j // grid])
            if all(np.abs(c - t).max() > sep for t in taken):
                taken.append(c); picks.append(j)
            if len(picks) == k:
                break
        pts = np.stack([gx[np.array(picks) % grid], gy[np.array(picks) // grid]], 1)
        errs.append(np.hypot(*(pts - XYq[r]).T).min())
    return np.array(errs)


def cdf_row(e):
    return dict(median=float(np.median(e)),
                le05=float(np.mean(e <= 0.5)), le1=float(np.mean(e <= 1)),
                le2=float(np.mean(e <= 2)), worst=float(e.max()))


def fmt(name, e):
    r = cdf_row(e)
    return (f"{name:<26} median {r['median']:5.2f} m | <=0.5m {r['le05']:5.1%} "
            f"| <=1m {r['le1']:5.1%} | <=2m {r['le2']:5.1%} | worst {r['worst']:4.1f} m")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset",
                    default="vsa_cognitive_mapping/configs/classroom_seq.json")
    ap.add_argument("--out-dir", default=None,
                    help="defaults to the dataset config's out_dir")
    ap.add_argument("--encoder", default="crop_embeddings_dinov2.pt")
    ap.add_argument("--treatment", default="zscored")
    ap.add_argument("--bind-yaw", action="store_true")
    ap.add_argument("--hd-dim", type=int, default=4096)
    ap.add_argument("--grid", type=int, default=48)
    ap.add_argument("--frac", type=float, default=0.3)
    ap.add_argument("--gap", type=int, default=45)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--battery-max", type=int, default=100,
                    help="strided battery size (matches the inspector's rule)")
    args = ap.parse_args()

    seq = load_sequence(args.dataset)
    out_dir = args.out_dir or seq.out_dir
    q = seq.pose_quality()
    if q is None or not q.usable:
        raise SystemExit("dataset has no usable poses")
    px, py, pyaw = seq.poses()
    pose_of = {int(f): (px[i], py[i], pyaw[i])
               for i, f in enumerate(seq.frame_ids())}

    uf, E = frame_descriptors(out_dir, args.encoder)
    keep = np.array([int(f) in pose_of for f in uf])
    uf, E = uf[keep], E[keep]
    XY = np.array([pose_of[int(f)][:2] for f in uf])
    PSI = np.array([pose_of[int(f)][2] for f in uf])

    Et = treat(E, args.treatment)
    z, _ = random_project_to_phasor(
        torch.from_numpy(np.ascontiguousarray(Et)).float(), d=args.hd_dim, seed=0)
    Z = z.numpy().astype(np.complex128)
    rng = np.random.RandomState(1000 + args.split_seed)
    mem, qry = make_split(uf, "blocked", args.frac, args.gap, rng)
    mb = np.where(mem)[0]
    qi = np.where(qry)[0][::max(1, int(np.ceil(qry.sum() / args.battery_max)))]

    enc = ClassroomEncoders(args.hd_dim, 0, 0.75, 20.0)
    ext = [XY[:, 0].min() - 1, XY[:, 0].max() + 1,
           XY[:, 1].min() - 1, XY[:, 1].max() + 1]
    gx = np.linspace(ext[0], ext[1], args.grid)
    gy = np.linspace(ext[2], ext[3], args.grid)
    G = np.empty((args.grid ** 2, args.hd_dim), np.complex64)
    k = 0
    for yy in gy:
        for xx in gx:
            G[k] = enc.ctx_pos(float(xx), float(yy)).values.astype(np.complex64)
            k += 1
    P = np.stack([enc.ctx_pos(float(a), float(b)).values for a, b in XY[mb]])
    if args.bind_yaw:
        H = np.stack([enc.ctx_head(float(a)).values for a in PSI[mb]])
        M = (Z[mb] * P * H).mean(0)
    else:
        M = (Z[mb] * P).mean(0)
    M /= max(np.abs(M).max(), 1e-12)
    R = M[None, :] / Z[qi]
    if args.bind_yaw:
        R = R / np.stack([enc.ctx_head(float(a)).values for a in PSI[qi]])
    S = (R @ np.conj(G).T).real
    pk = np.stack([gx[S.argmax(1) % args.grid], gy[S.argmax(1) // args.grid]], 1)
    err_vsa = np.hypot(*(pk - XY[qi]).T)
    err_t3 = topk_err(S, XY[qi], gx, gy, args.grid)

    U = Et / (np.linalg.norm(Et, axis=1, keepdims=True) + 1e-12)
    nn = mb[(U[qi] @ U[mb].T).argmax(1)]
    err_knn = np.hypot(*(XY[nn] - XY[qi]).T)
    orc = np.array([np.hypot(*(XY[mb] - XY[qq]).T).min() for qq in qi])

    tag = f"{args.encoder} · {args.treatment}" + (" · bind-yaw" if args.bind_yaw else "")
    print(f"{seq.name}: {len(qi)} queries (blocked seed {args.split_seed}), "
          f"oracle median {np.median(orc):.2f} m — {tag}\n")
    print(fmt("kNN ceiling (same desc.)", err_knn))
    print(fmt("VSA memory (argmax)", err_vsa))
    print(fmt("VSA memory (top-3 modes)", err_t3))

    out = os.path.join(out_dir, f"recall_cdf_{os.path.splitext(args.encoder)[0]}"
                                f"_{args.treatment}{'_yaw' if args.bind_yaw else ''}.json")
    with open(out, "w") as f:
        json.dump(dict(scene=seq.name, encoder=args.encoder,
                       treatment=args.treatment, bind_yaw=args.bind_yaw,
                       n=len(qi), oracle_median=float(np.median(orc)),
                       knn=cdf_row(err_knn), vsa=cdf_row(err_vsa),
                       vsa_top3=cdf_row(err_t3)), f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
