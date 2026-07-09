"""Convert the Telluride Spot run1 (HuggingFace) to the common transition format.

    uv run python scripts/run1_to_transitions.py --out data/spot_run1 [--rate 3.0]

Uses `rgb_d455` frames and the `odometry_lio_sam` pose stream of
lorinachey/spot-telluride-workshop-dataset. Rows of that dataset are NOT
stored in time order (see notebooks/spot_dataset_exploration.ipynb), so both
streams are sorted by timestamp_ns first. Frames are greedily subsampled to
--rate Hz, the LIO-SAM pose is interpolated at each kept frame time, and
actions are labeled from executed motion with the same rule as the SCAND
converter (actions.label_transitions). Requires the `data` extra (datasets).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset

from vsa_cognitive_mapping.actions import keep_at_rate, label_transitions, onehot, yaw_from_quaternion
from vsa_cognitive_mapping.data import ACTIONS, ONEHOT_COLS

REPO = "lorinachey/spot-telluride-workshop-dataset"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--rate", type=float, default=3.0, help="output frame rate (Hz)")
    args = parser.parse_args()
    out = Path(args.out)

    odo = load_dataset(REPO, "odometry_lio_sam", split="train")
    o = np.argsort(np.asarray(odo["timestamp_ns"], dtype=np.int64), kind="stable")
    t_o = np.asarray(odo["timestamp_ns"], dtype=np.int64)[o] / 1e9
    ox, oy, oz = (np.asarray(odo[k])[o] for k in ("x", "y", "z"))
    oyaw = np.unwrap(yaw_from_quaternion(*(np.asarray(odo[k])[o] for k in ("qx", "qy", "qz", "qw"))))

    rgb = load_dataset(REPO, "rgb_d455", split="train")
    r = np.argsort(np.asarray(rgb["timestamp_ns"], dtype=np.int64), kind="stable")
    t_all = np.asarray(rgb["timestamp_ns"], dtype=np.int64)[r] / 1e9
    inside = (t_all >= t_o[0]) & (t_all <= t_o[-1])
    keep = keep_at_rate(t_all[inside], args.rate)
    rows = r[inside][keep]
    t_f = t_all[inside][keep]

    (out / "images").mkdir(parents=True, exist_ok=True)
    relpaths = []
    for i, row in enumerate(rows):
        rel = f"images/frame_{i:05d}.jpg"
        rgb[int(row)]["image"].save(out / rel, quality=92)
        relpaths.append(rel)

    fx, fy, fz, fyaw = (np.interp(t_f, t_o, v) for v in (ox, oy, oz, oyaw))
    labels, speed, yawrate = label_transitions(fx, fy, fyaw, t_f)
    ok = np.diff(t_f) <= 2.0 / args.rate
    oh = onehot(labels)

    gt_rows, pose_rows = [], []
    for i in np.flatnonzero(ok):
        gt_rows.append({"frame_t": i, "frame_tp1": i + 1, "image_t": relpaths[i],
                        "action": ACTIONS[labels[i]], **dict(zip(ONEHOT_COLS, oh[i])),
                        "image_tp1": relpaths[i + 1]})
        dyaw = fyaw[i + 1] - fyaw[i]
        pose_rows.append({"frame_t": i, "frame_tp1": i + 1,
                          "x_t": fx[i], "y_t": fy[i], "z_t": fz[i], "yaw_t_rad": fyaw[i],
                          "x_tp1": fx[i + 1], "y_tp1": fy[i + 1], "z_tp1": fz[i + 1],
                          "yaw_tp1_rad": fyaw[i + 1],
                          "dx_world": fx[i + 1] - fx[i], "dy_world": fy[i + 1] - fy[i],
                          "dz_world": fz[i + 1] - fz[i],
                          "dist_ground": float(np.hypot(fx[i + 1] - fx[i], fy[i + 1] - fy[i])),
                          "dyaw_rad": dyaw, "dyaw_deg": float(np.degrees(dyaw))})

    pd.DataFrame(gt_rows).to_csv(out / "transitions_gt.csv", index=False)
    pd.DataFrame(pose_rows).to_csv(out / "transitions.csv", index=False)
    counts = {a: int((labels[ok] == i).sum()) for i, a in enumerate(ACTIONS)}
    print(f"wrote {len(gt_rows)} transitions ({t_f[-1] - t_f[0]:.0f} s) to {out}  {counts}  "
          f"speed p50/p95 {np.percentile(speed, 50):.2f}/{np.percentile(speed, 95):.2f} m/s  "
          f"|yawrate| p95 {np.degrees(np.percentile(np.abs(yawrate), 95)):.1f} deg/s")


if __name__ == "__main__":
    main()
