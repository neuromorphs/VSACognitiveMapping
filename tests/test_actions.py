"""Tests for odometry-derived action labeling and the real-data converters."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vsa_cognitive_mapping.actions import (V_TH, W_TH, keep_at_rate, label_transitions,
                                           onehot, yaw_from_quaternion)
from vsa_cognitive_mapping.data import ACTIONS, TransitionDataset


def make_track(segments, dt=0.3):
    """Poses from (action, n_steps) segments: forward 0.5 m/s, turns 30 deg/s."""
    x, y, yaw = [0.0], [0.0], [0.0]
    for action, n in segments:
        for _ in range(n):
            v = 0.5 if action == "forward" else 0.0
            w = {"left": np.radians(30), "right": -np.radians(30)}.get(action, 0.0)
            yaw.append(yaw[-1] + w * dt)
            x.append(x[-1] + v * np.cos(yaw[-1]) * dt)
            y.append(y[-1] + v * np.sin(yaw[-1]) * dt)
    t = np.arange(len(x)) * dt
    return np.array(x), np.array(y), np.array(yaw), t


def test_label_transitions_recovers_segments():
    segments = [("stop", 3), ("forward", 5), ("left", 4), ("forward", 3), ("right", 4), ("stop", 2)]
    x, y, yaw, t = make_track(segments)
    labels, speed, yawrate = label_transitions(x, y, yaw, t)
    expected = [ACTIONS.index(a) for a, n in segments for _ in range(n)]
    assert labels.tolist() == expected


def test_label_transitions_turn_wins_over_forward():
    # arc: translating fast AND turning fast -> the turn label wins
    x, y, yaw, t = make_track([("left", 5)])
    x = np.linspace(0, 2.0, len(x))  # add 1.3 m/s forward motion on top
    labels, _, _ = label_transitions(x, y, yaw, t)
    assert all(l == ACTIONS.index("left") for l in labels)


def test_label_transitions_wrapped_yaw():
    # crossing the +/-pi boundary must not produce a fake fast turn
    t = np.arange(5) * 0.3
    yaw = np.array([np.pi - 0.02, np.pi - 0.01, np.pi, -np.pi + 0.01, -np.pi + 0.02])
    labels, _, yawrate = label_transitions(np.zeros(5), np.zeros(5), yaw, t)
    assert np.abs(yawrate).max() < W_TH
    assert all(l == ACTIONS.index("stop") for l in labels)


def test_label_transitions_rejects_unsorted_time():
    with pytest.raises(ValueError):
        label_transitions(np.zeros(3), np.zeros(3), np.zeros(3), np.array([0.0, 1.0, 0.5]))


def test_threshold_boundaries():
    # exactly at threshold -> not a turn / not forward (strict inequality)
    t = np.array([0.0, 1.0])
    labels, _, _ = label_transitions(np.array([0.0, V_TH]), np.zeros(2), np.zeros(2), t)
    assert labels[0] == ACTIONS.index("stop")
    labels, _, _ = label_transitions(np.zeros(2), np.zeros(2), np.array([0.0, W_TH]), t)
    assert labels[0] == ACTIONS.index("stop")


def test_yaw_from_quaternion_pure_z():
    for angle in (-2.0, -0.5, 0.0, 1.0, 3.0):
        q = (0.0, 0.0, np.sin(angle / 2), np.cos(angle / 2))
        assert yaw_from_quaternion(*q) == pytest.approx(angle, abs=1e-9)


def test_onehot_matches_csv_columns():
    oh = onehot(np.array([0, 1, 2, 3]))
    assert oh.shape == (4, 4)
    assert (oh == np.eye(4)).all()


def test_keep_at_rate_spacing():
    t = np.arange(0, 10, 1 / 30)  # 30 Hz stream
    kept = keep_at_rate(t, 3.0)
    assert (np.diff(t[kept]) >= 1 / 3.0 - 1e-9).all()
    assert 25 <= len(kept) <= 31  # ~3 Hz over 10 s
    with pytest.raises(ValueError):
        keep_at_rate(np.array([0.0, 1.0, 0.5]), 3.0)


@pytest.fixture(scope="module")
def synthetic_bag(tmp_path_factory):
    """A tiny ROS1 bag with a known trajectory: stop, forward, left arc."""
    rosbag1 = pytest.importorskip("rosbags.rosbag1")
    from rosbags.typesys import Stores, get_typestore
    from PIL import Image
    import io

    store = get_typestore(Stores.ROS1_NOETIC)
    Odometry = store.types["nav_msgs/msg/Odometry"]
    CompressedImage = store.types["sensor_msgs/msg/CompressedImage"]
    Twist = store.types["geometry_msgs/msg/Twist"]
    Joy = store.types["sensor_msgs/msg/Joy"]
    Header = store.types["std_msgs/msg/Header"]
    Time = store.types["builtin_interfaces/msg/Time"]
    Pose = store.types["geometry_msgs/msg/Pose"]
    PoseWithCovariance = store.types["geometry_msgs/msg/PoseWithCovariance"]
    TwistWithCovariance = store.types["geometry_msgs/msg/TwistWithCovariance"]
    Point = store.types["geometry_msgs/msg/Point"]
    Quaternion = store.types["geometry_msgs/msg/Quaternion"]
    Vector3 = store.types["geometry_msgs/msg/Vector3"]

    segments = [("stop", 4), ("forward", 6), ("left", 5)]
    x, y, yaw, t = make_track(segments, dt=0.1)  # 10 Hz odometry
    t = t + 1000.0  # nonzero epoch

    jpeg = io.BytesIO()
    Image.new("RGB", (32, 24), (200, 120, 40)).save(jpeg, format="JPEG")

    def header(ts):
        return Header(seq=0, stamp=Time(sec=int(ts), nanosec=int((ts % 1) * 1e9)), frame_id="odom")

    def twist(v=0.0, w=0.0):
        return TwistWithCovariance(twist=Twist(linear=Vector3(x=v, y=0.0, z=0.0),
                                               angular=Vector3(x=0.0, y=0.0, z=w)),
                                   covariance=np.zeros(36))

    path = tmp_path_factory.mktemp("bag") / "synthetic.bag"
    from rosbags.rosbag1 import Writer
    with Writer(path) as writer:
        c_odom = writer.add_connection("/odom", Odometry.__msgtype__, typestore=store)
        c_img = writer.add_connection("/image_raw/compressed", CompressedImage.__msgtype__, typestore=store)
        c_joy = writer.add_connection("/joystick", Joy.__msgtype__, typestore=store)
        for i, ts in enumerate(t):
            q = (0.0, 0.0, float(np.sin(yaw[i] / 2)), float(np.cos(yaw[i] / 2)))
            msg = Odometry(header=header(ts), child_frame_id="base",
                           pose=PoseWithCovariance(pose=Pose(position=Point(x=x[i], y=y[i], z=0.0),
                                                             orientation=Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])),
                                                   covariance=np.zeros(36)),
                           twist=twist())
            writer.write(c_odom, int(ts * 1e9), store.serialize_ros1(msg, Odometry.__msgtype__))
            img = CompressedImage(header=header(ts), format="bgr8; jpeg compressed bgr8",
                                  data=np.frombuffer(jpeg.getvalue(), dtype=np.uint8))
            writer.write(c_img, int(ts * 1e9), store.serialize_ros1(img, CompressedImage.__msgtype__))
            # DS4 sticks mirror the executed segment (axis4 = -forward, axis0 = -turn)
            seg = "stop" if i < 4 else ("forward" if i < 10 else "left")
            axes = np.zeros(8, dtype=np.float32)
            axes[4] = -0.5 if seg == "forward" else 0.0
            axes[0] = -0.6 if seg == "left" else 0.0
            joy = Joy(header=header(ts), axes=axes, buttons=np.zeros(4, dtype=np.int32))
            writer.write(c_joy, int(ts * 1e9), store.serialize_ros1(joy, Joy.__msgtype__))
    return path


@pytest.mark.slow
def test_scand_converter_roundtrip(synthetic_bag, tmp_path):
    """Full converter run on a synthetic bag: output loads through
    TransitionDataset and the labels match the generated motion."""
    out = tmp_path / "converted"
    repo = Path(__file__).resolve().parents[1]
    res = subprocess.run(
        [sys.executable, str(repo / "scripts/scand_to_transitions.py"),
         "--bags", str(synthetic_bag), "--out", str(out), "--rate", "5.0"],
        capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert "joystick agreement" in res.stdout

    df = pd.read_csv(out / "transitions_gt.csv")
    assert len(df) > 5
    # every referenced image exists and ids are consecutive within the bag
    for col in ("image_t", "image_tp1"):
        assert all((out / p).exists() for p in df[col])
    assert (df["frame_tp1"] - df["frame_t"] == 1).all()
    # motion was stop -> forward -> left; labels must appear in that order
    seq = df["action"].tolist()
    assert seq[0] == "stop" and seq[-1] == "left" and "forward" in seq
    assert "right" not in seq
    # one-hot columns are consistent with the action column
    for _, row in df.iterrows():
        assert row[f"onehot_{row['action']}"] == 1.0

    # loads through the standard dataset (images 32x24 jpg -> tensor)
    ds = TransitionDataset(out, split="train", val_fraction=0.0)
    img_t, img_tp1, action = ds[0]
    assert img_t.shape == (3, 224, 224) and action.shape == (4,)

    # pose sidecar aligns with the gt CSV
    pose = pd.read_csv(out / "transitions.csv")
    assert (pose["frame_t"].to_numpy() == df["frame_t"].to_numpy()).all()
