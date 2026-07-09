"""Convert SCAND Spot rosbags into the common transition-dataset format.

    uv run python scripts/scand_to_transitions.py \
        --bags data/scand_bags/*.bag --out data/scand_indoor [--rate 3.0]

Per bag: extract Azure Kinect RGB frames (`/image_raw/compressed`; the JPEG
bytes are written to disk unchanged), greedily subsampled to --rate Hz;
interpolate the `/odom` pose at each kept frame time; label one-hot actions
from the executed motion (actions.label_transitions). All bags append to one
merged `transitions_gt.csv` — transitions never span bags, and frame ids are
offset per bag so they stay unique. Also writes `transitions.csv` with poses
per transition (same columns the bezier simulator writes, for probing/eval).

As a label sanity check, the derived action of each transition is compared
with the label implied by the DS4 teleop sticks over the same window
(`/joystick`; `/navigation/cmd_vel` is all zeros in these bags — the nav
stack was idle during teleop). Empirically, axis 4 = -forward command and
axis 0 = -turn command (correlation with odometry ~0.8-0.9), and commands
lead the executed motion by ~0.2 s; expect ~85-95% per-bag agreement.
Requires the `data` extra (rosbags).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rosbags.highlevel import AnyReader

from vsa_cognitive_mapping.actions import (keep_at_rate, label_transitions,
                                           onehot, yaw_from_quaternion)
from vsa_cognitive_mapping.data import ACTIONS, ONEHOT_COLS

IMAGE_TOPIC = "/image_raw/compressed"
ODOM_TOPIC = "/odom"
JOY_TOPIC = "/joystick"

# DS4 stick -> command mapping (found by correlating axes with odometry):
# forward command = -axes[4], turn command = -axes[0], both in [-1, 1] stick
# units; commands lead the executed motion by roughly CMD_LAG seconds.
V_STICK_TH = 0.10
W_STICK_TH = 0.15
CMD_LAG = 0.2

# One SCAND frame id block per bag; no bag comes close to 100k kept frames.
FRAME_ID_STRIDE = 100_000


def read_bag(bag: Path):
    """(odom dict, cmd dict, [(t, jpeg bytes), ...]) — all sorted by time."""
    odom = {"t": [], "x": [], "y": [], "z": [], "yaw": []}
    cmd = {"t": [], "v": [], "w": []}
    frames = []
    with AnyReader([bag]) as reader:
        conns = [c for c in reader.connections if c.topic in (IMAGE_TOPIC, ODOM_TOPIC, JOY_TOPIC)]
        for conn, ts, raw in reader.messages(connections=conns):
            if conn.topic == ODOM_TOPIC:
                m = reader.deserialize(raw, conn.msgtype)
                p, q = m.pose.pose.position, m.pose.pose.orientation
                odom["t"].append(ts / 1e9)
                odom["x"].append(p.x)
                odom["y"].append(p.y)
                odom["z"].append(p.z)
                odom["yaw"].append(yaw_from_quaternion(q.x, q.y, q.z, q.w))
            elif conn.topic == JOY_TOPIC:
                m = reader.deserialize(raw, conn.msgtype)
                cmd["t"].append(ts / 1e9)
                cmd["v"].append(-m.axes[4])
                cmd["w"].append(-m.axes[0])
            else:
                m = reader.deserialize(raw, conn.msgtype)
                frames.append((ts / 1e9, m.data.tobytes()))
    odom = {k: np.asarray(v) for k, v in odom.items()}
    cmd = {k: np.asarray(v) for k, v in cmd.items()}
    frames.sort(key=lambda f: f[0])
    assert (np.diff(odom["t"]) >= 0).all(), f"{bag.name}: odom not time-sorted"
    return odom, cmd, frames


def cmd_labels(cmd: dict, t_frames: np.ndarray) -> np.ndarray | None:
    """Action labels from the mean commanded (v, w) stick deflection over
    each transition window (shifted by CMD_LAG: commands lead the motion)."""
    if len(cmd["t"]) == 0:
        return None
    labels = np.full(len(t_frames) - 1, ACTIONS.index("stop"), dtype=np.int64)
    for i, (t0, t1) in enumerate(zip(t_frames[:-1], t_frames[1:])):
        m = (cmd["t"] >= t0 - CMD_LAG) & (cmd["t"] < t1 - CMD_LAG)
        if not m.any():  # no command in window -> nearest sample
            m = np.argmin(np.abs(cmd["t"] - (0.5 * (t0 + t1) - CMD_LAG)))
        v, w = np.mean(cmd["v"][m]), np.mean(cmd["w"][m])
        if w > W_STICK_TH:
            labels[i] = ACTIONS.index("left")
        elif w < -W_STICK_TH:
            labels[i] = ACTIONS.index("right")
        elif v > V_STICK_TH:
            labels[i] = ACTIONS.index("forward")
    return labels


def convert_bag(bag: Path, out: Path, bag_idx: int, rate_hz: float):
    """Extract one bag; returns (transitions_gt rows, transitions rows)."""
    odom, cmd, frames = read_bag(bag)
    t_all = np.asarray([f[0] for f in frames])

    # frames must lie inside the odometry's time span for interpolation
    inside = (t_all >= odom["t"][0]) & (t_all <= odom["t"][-1])
    frames = [f for f, ok in zip(frames, inside) if ok]
    t_all = t_all[inside]
    keep = keep_at_rate(t_all, rate_hz)
    t_f = t_all[keep]

    img_dir = out / "images" / bag.stem
    img_dir.mkdir(parents=True, exist_ok=True)
    relpaths = []
    for local, i in enumerate(keep):
        rel = f"images/{bag.stem}/frame_{local:05d}.jpg"
        (out / rel).write_bytes(frames[int(i)][1])
        relpaths.append(rel)

    yaw_u = np.unwrap(odom["yaw"])
    fx, fy, fz, fyaw = (np.interp(t_f, odom["t"], v) for v in (odom["x"], odom["y"], odom["z"], yaw_u))
    labels, speed, yawrate = label_transitions(fx, fy, fyaw, t_f)

    # drop transitions spanning a recording gap (would smear the action label)
    ok = np.diff(t_f) <= 2.0 / rate_hz
    oh = onehot(labels)
    ids = FRAME_ID_STRIDE * bag_idx + np.arange(len(t_f))

    gt_rows, pose_rows = [], []
    for i in np.flatnonzero(ok):
        gt_rows.append({"frame_t": ids[i], "frame_tp1": ids[i + 1], "image_t": relpaths[i],
                        "action": ACTIONS[labels[i]], **dict(zip(ONEHOT_COLS, oh[i])),
                        "image_tp1": relpaths[i + 1]})
        dyaw = fyaw[i + 1] - fyaw[i]
        pose_rows.append({"frame_t": ids[i], "frame_tp1": ids[i + 1],
                          "x_t": fx[i], "y_t": fy[i], "z_t": fz[i], "yaw_t_rad": fyaw[i],
                          "x_tp1": fx[i + 1], "y_tp1": fy[i + 1], "z_tp1": fz[i + 1],
                          "yaw_tp1_rad": fyaw[i + 1],
                          "dx_world": fx[i + 1] - fx[i], "dy_world": fy[i + 1] - fy[i],
                          "dz_world": fz[i + 1] - fz[i],
                          "dist_ground": float(np.hypot(fx[i + 1] - fx[i], fy[i + 1] - fy[i])),
                          "dyaw_rad": dyaw, "dyaw_deg": float(np.degrees(dyaw))})

    # report
    counts = {a: int((labels[ok] == i).sum()) for i, a in enumerate(ACTIONS)}
    cl = cmd_labels(cmd, t_f)
    agree = float((cl[ok] == labels[ok]).mean()) if cl is not None else float("nan")
    dur = t_f[-1] - t_f[0]
    print(f"{bag.stem}: {dur:5.0f} s  {len(gt_rows):4d} transitions  {counts}  "
          f"speed p50/p95 {np.percentile(speed, 50):.2f}/{np.percentile(speed, 95):.2f} m/s  "
          f"|yawrate| p95 {np.degrees(np.percentile(np.abs(yawrate), 95)):.1f} deg/s  "
          f"joystick agreement {agree:.1%}")
    return gt_rows, pose_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bags", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--rate", type=float, default=3.0, help="output frame rate (Hz)")
    args = parser.parse_args()

    out = Path(args.out)
    gt_rows, pose_rows = [], []
    for bag_idx, bag in enumerate(sorted(Path(b) for b in args.bags)):
        gt, pose = convert_bag(bag, out, bag_idx, args.rate)
        gt_rows += gt
        pose_rows += pose

    pd.DataFrame(gt_rows).to_csv(out / "transitions_gt.csv", index=False)
    pd.DataFrame(pose_rows).to_csv(out / "transitions.csv", index=False)
    total = {a: sum(1 for r in gt_rows if r["action"] == a) for a in ACTIONS}
    print(f"\nwrote {len(gt_rows)} transitions from {len(set(r['image_t'].split('/')[1] for r in gt_rows))} "
          f"bags to {out}  {total}")


if __name__ == "__main__":
    main()
