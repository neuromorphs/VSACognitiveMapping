"""Derive one-hot actions (forward, stop, left, right) from odometry.

Real-robot recordings (SCAND, the Telluride run1) log executed motion, not
our discrete action alphabet. Both converters use this one rule so train and
test action semantics match exactly: a transition is a turn if the yaw rate
exceeds W_TH (sign gives left/right), else `forward` if translating faster
than V_TH, else `stop`.

Thresholds were picked on run1 LIO-SAM odometry (Spot walks ~0.5-1 m/s,
turn bursts ~25-30 deg/s; see notebooks/spot_dataset_exploration.ipynb) and
are validated per-dataset by the converters' printed speed/yaw-rate stats.
"""

import numpy as np

from .data import ACTIONS

V_TH = 0.15  # m/s — below this the robot is not translating
W_TH = np.radians(12.0)  # rad/s — above this turning dominates


def yaw_from_quaternion(qx, qy, qz, qw) -> np.ndarray:
    """Z-axis (heading) Euler angle of unit quaternions, ROS convention."""
    return np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (np.asarray(qy) ** 2 + np.asarray(qz) ** 2))


def label_transitions(x: np.ndarray, y: np.ndarray, yaw: np.ndarray, t: np.ndarray,
                      v_th: float = V_TH, w_th: float = W_TH):
    """Label the N-1 transitions between N consecutive poses.

    `yaw` may be wrapped; it is unwrapped here. Returns (labels, speed,
    yawrate): labels are indices into data.ACTIONS, speed in m/s, signed
    yawrate in rad/s (positive = left/counter-clockwise, ROS z-up).
    """
    x, y, t = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64), np.asarray(t, dtype=np.float64)
    dt = np.diff(t)
    if (dt <= 0).any():
        raise ValueError("timestamps must be strictly increasing — sort poses by time first")
    speed = np.hypot(np.diff(x), np.diff(y)) / dt
    yawrate = np.diff(np.unwrap(np.asarray(yaw, dtype=np.float64))) / dt

    labels = np.full(len(dt), ACTIONS.index("stop"), dtype=np.int64)
    labels[speed > v_th] = ACTIONS.index("forward")
    labels[yawrate > w_th] = ACTIONS.index("left")
    labels[yawrate < -w_th] = ACTIONS.index("right")
    return labels, speed, yawrate


def onehot(labels: np.ndarray) -> np.ndarray:
    """(N,) action indices -> (N, 4) one-hot float array, ACTIONS order."""
    eye = np.eye(len(ACTIONS), dtype=np.float32)
    return eye[labels]


def keep_at_rate(t: np.ndarray, rate_hz: float) -> np.ndarray:
    """Indices of a greedy subsample of sorted timestamps `t` (seconds) such
    that consecutive kept stamps are >= 1/rate_hz apart. Both real-data
    converters use this to bring ~30 Hz camera streams down to the training
    frame rate (consecutive full-rate frames are near-duplicates)."""
    t = np.asarray(t, dtype=np.float64)
    if (np.diff(t) < 0).any():
        raise ValueError("timestamps must be sorted")
    min_dt = 1.0 / rate_hz
    kept = [0]
    for i in range(1, len(t)):
        if t[i] - t[kept[-1]] >= min_dt:
            kept.append(i)
    return np.asarray(kept, dtype=np.int64)
