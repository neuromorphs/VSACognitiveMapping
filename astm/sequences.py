"""Dataset abstraction: point the pipeline at a new sequence via a config file.

Every stage used to hardcode ``lorinachey/spot-telluride-workshop-dataset`` and
its config names, so trying a new dataset meant editing code in four places and
usually discovering the fifth at runtime. A sequence is now described by a small
JSON file and loaded through one interface.

    python -m vsa_cognitive_mapping.sequences validate --dataset configs/my_room.json

**Poses are optional, and that matters.** The strongest results in this project
(isotropy, effective rank, the Timkey decomposition, encoder comparison, crop
retrieval) need only images. Only crosstalk, place recall and the memory build
need poses. A sequence with no pose file is still fully usable for the first
group, and `validate` says which analyses are available.

**Poses are checked, not trusted.** `school_run1`'s LIO-SAM solution reported
23 km of travel in 455 s with a peak implied speed of 1,595 m/s, and every
pose-dependent number computed on it was silently wrong for a day. `validate`
runs that check up front and `PoseQuality.usable` gates the rest.

Config formats
--------------

HuggingFace (an image config plus an optional odometry config)::

    {
      "name": "classroom",
      "kind": "huggingface",
      "repo": "lorinachey/spot-telluride-workshop-dataset",
      "rgb_config": "rgb_d455",
      "odom_config": "odometry_lio_sam",
      "out_dir": "outputs/classroom"
    }

A local image folder, poses optional::

    {
      "name": "my_room",
      "kind": "folder",
      "images": "data/my_room/rgb/*.png",
      "fps": 30,
      "poses": {"kind": "tum", "path": "data/my_room/groundtruth.txt"},
      "out_dir": "outputs/my_room"
    }

``poses.kind`` is one of:

``tum``
    The TUM RGB-D / ICL-NUIM text format, ``timestamp tx ty tz qx qy qz qw``,
    ``#`` comments ignored. This covers TUM RGB-D, ICL-NUIM (``*.gt.freiburg``)
    and most SLAM benchmarks. ``axes`` picks which two of xyz form the floor
    plane — ICL-NUIM's living room is ``"xz"``, most robot data is ``"xy"``.
``csv``
    Any CSV with columns for a time key and position. Column names are given
    explicitly, e.g.
    ``{"kind": "csv", "path": "...", "time": "timestamp_ns", "x": "x", "y": "y", "yaw": "yaw_rad"}``.
    ``yaw`` may be omitted, or replaced by ``qx/qy/qz/qw`` for quaternions.
"""
from __future__ import annotations

import glob as _glob
import json
import os
import re
from dataclasses import dataclass, field

import numpy as np

DEFAULT_REPO = "lorinachey/spot-telluride-workshop-dataset"


# --------------------------------------------------------------------------
# pose quality
# --------------------------------------------------------------------------

@dataclass
class PoseQuality:
    """Is this pose stream physically plausible?"""
    n: int
    duration_s: float
    path_len_m: float
    extent_m: tuple
    max_step_m: float
    jumps_over_5m: int
    max_speed_ms: float
    median_step_m: float
    usable: bool
    reasons: list = field(default_factory=list)

    def report(self) -> str:
        flag = "OK" if self.usable else "UNUSABLE"
        s = (f"  poses {self.n}, {self.duration_s:.1f} s, path {self.path_len_m:.1f} m, "
             f"extent {self.extent_m[0]:.1f} x {self.extent_m[1]:.1f} m\n"
             f"  step median {self.median_step_m*100:.1f} cm, max {self.max_step_m:.2f} m, "
             f"jumps>5 m {self.jumps_over_5m}, peak speed {self.max_speed_ms:.1f} m/s\n"
             f"  verdict: {flag}")
        for r in self.reasons:
            s += f"\n    - {r}"
        return s


def assess_poses(t_s: np.ndarray, x: np.ndarray, y: np.ndarray,
                 max_speed_ms: float = 5.0) -> PoseQuality:
    """Flag a diverged SLAM solution before it silently poisons every result.

    `max_speed_ms` defaults to 5 m/s: comfortably above a walking robot (a Spot
    tops out near 1.6) and above handheld camera motion, while still orders of
    magnitude below the 1,595 m/s a diverged solution produced.
    """
    step = np.hypot(np.diff(x), np.diff(y))
    dt = np.maximum(np.diff(t_s), 1e-9)
    speed = step / dt
    vmax = float(np.nanmax(speed)) if len(speed) else 0.0
    jumps = int((step > 5.0).sum())
    reasons = []
    if jumps:
        reasons.append(f"{jumps} consecutive jumps over 5 m")
    if vmax > max_speed_ms:
        reasons.append(f"peak implied speed {vmax:.1f} m/s exceeds {max_speed_ms} m/s")
    return PoseQuality(
        n=len(x), duration_s=float(t_s[-1] - t_s[0]) if len(t_s) > 1 else 0.0,
        path_len_m=float(step.sum()),
        extent_m=(float(np.ptp(x)), float(np.ptp(y))),
        max_step_m=float(step.max()) if len(step) else 0.0,
        jumps_over_5m=jumps, max_speed_ms=vmax,
        median_step_m=float(np.median(step)) if len(step) else 0.0,
        usable=not reasons, reasons=reasons)


# --------------------------------------------------------------------------
# pose readers
# --------------------------------------------------------------------------

def _yaw_from_quat(qx, qy, qz, qw):
    return np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def read_tum_poses(path, axes="xy"):
    """TUM RGB-D / ICL-NUIM text poses -> (t_seconds, x, y, yaw)."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            if len(p) >= 8:
                rows.append([float(v) for v in p[:8]])
    if not rows:
        raise ValueError(f"no pose rows parsed from {path}")
    a = np.asarray(rows)
    t, tx, ty, tz = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    qx, qy, qz, qw = a[:, 4], a[:, 5], a[:, 6], a[:, 7]
    pick = {"xy": (tx, ty), "xz": (tx, tz), "yz": (ty, tz)}
    if axes not in pick:
        raise ValueError(f"poses.axes must be one of {sorted(pick)}, got {axes!r}")
    px, py = pick[axes]
    return t, px, py, _yaw_from_quat(qx, qy, qz, qw)


def read_csv_poses(spec):
    """CSV poses with caller-named columns -> (t_seconds, x, y, yaw)."""
    import csv as _csv
    path = spec["path"]
    tcol, xcol, ycol = spec.get("time"), spec.get("x", "x"), spec.get("y", "y")
    ycol_yaw = spec.get("yaw")
    quat = [spec.get(k) for k in ("qx", "qy", "qz", "qw")]
    t, x, y, yaw, q = [], [], [], [], []
    with open(path, newline="") as f:
        for r in _csv.DictReader(f):
            if tcol:
                t.append(float(r[tcol]))
            x.append(float(r[xcol])); y.append(float(r[ycol]))
            if ycol_yaw:
                yaw.append(float(r[ycol_yaw]))
            elif all(quat):
                q.append([float(r[c]) for c in quat])
    x, y = np.asarray(x), np.asarray(y)
    t = np.asarray(t, dtype=np.float64) if t else np.arange(len(x), dtype=np.float64)
    if tcol and "ns" in tcol:
        t = t / 1e9
    if yaw:
        psi = np.asarray(yaw)
    elif q:
        q = np.asarray(q)
        psi = _yaw_from_quat(q[:, 0], q[:, 1], q[:, 2], q[:, 3])
    else:
        psi = np.zeros(len(x))
    return t, x, y, psi


def interp_to(t_target, t_src, x, y, yaw):
    """Interpolate poses onto frame times. Yaw is unwrapped first so the +-pi
    wrap does not spin the interpolation through a full turn."""
    o = np.argsort(t_src)
    t_src, x, y, yaw = t_src[o], x[o], y[o], yaw[o]
    psi = np.interp(t_target, t_src, np.unwrap(yaw))
    return (np.interp(t_target, t_src, x), np.interp(t_target, t_src, y),
            np.arctan2(np.sin(psi), np.cos(psi)))


# --------------------------------------------------------------------------
# sequences
# --------------------------------------------------------------------------

class Sequence:
    """Ordered RGB frames, with optional poses aligned to them."""

    name = "sequence"
    out_dir = "outputs/sequence"

    def __len__(self):
        raise NotImplementedError

    def frame_ids(self) -> np.ndarray:
        raise NotImplementedError

    def timestamps_s(self) -> np.ndarray:
        raise NotImplementedError

    def image(self, i):
        """PIL RGB image for the i-th frame in time order."""
        raise NotImplementedError

    def poses(self):
        """(x, y, yaw) aligned to frame order, or None if the sequence has none."""
        return None

    def pose_quality(self):
        p = self.poses()
        if p is None:
            return None
        return assess_poses(self.timestamps_s(), p[0], p[1])

    def describe(self):
        n = len(self)
        t = self.timestamps_s()
        dur = float(t[-1] - t[0]) if n > 1 else 0.0
        lines = [f"{self.name}: {n} frames, {dur:.1f} s "
                 f"({n / dur:.1f} fps)" if dur > 0 else f"{self.name}: {n} frames"]
        q = self.pose_quality()
        lines.append(q.report() if q else "  poses: NONE (image-only analyses available)")
        return "\n".join(lines)


class HuggingFaceSequence(Sequence):
    def __init__(self, cfg):
        from datasets import load_dataset
        self.name = cfg.get("name", cfg.get("rgb_config", "hf"))
        self.out_dir = cfg.get("out_dir", os.path.join("outputs", self.name))
        repo = cfg.get("repo", DEFAULT_REPO)
        self._ds = load_dataset(repo, cfg["rgb_config"], split="train")
        ts = np.asarray(self._ds["timestamp_ns"], dtype=np.int64)
        self._order = np.argsort(ts)
        self._ts = ts[self._order] / 1e9
        self._fids = np.asarray(self._ds["frame_idx"], dtype=np.int64)[self._order]
        self._cfg, self._repo = cfg, repo
        self._poses = None

    def __len__(self):
        return len(self._order)

    def frame_ids(self):
        return self._fids

    def timestamps_s(self):
        return self._ts

    def image(self, i):
        return self._ds[int(self._order[i])]["image"]

    def poses(self):
        oc = self._cfg.get("odom_config")
        if not oc:
            return None
        if self._poses is None:
            from datasets import load_dataset
            o = load_dataset(self._repo, oc, split="train")
            t = np.asarray(o["timestamp_ns"], dtype=np.float64) / 1e9
            yaw = _yaw_from_quat(*(np.asarray(o[k], dtype=np.float64)
                                   for k in ("qx", "qy", "qz", "qw")))
            self._poses = interp_to(self._ts, t,
                                    np.asarray(o["x"], dtype=np.float64),
                                    np.asarray(o["y"], dtype=np.float64), yaw)
        return self._poses


class FolderSequence(Sequence):
    def __init__(self, cfg):
        self.name = cfg.get("name", "folder")
        self.out_dir = cfg.get("out_dir", os.path.join("outputs", self.name))
        pat = cfg["images"]
        self._paths = sorted(_glob.glob(pat), key=_natural_key)
        if not self._paths:
            raise FileNotFoundError(f"no images matched {pat!r}")
        fps = float(cfg.get("fps", 30.0))
        self._ts = np.arange(len(self._paths), dtype=np.float64) / fps
        self._fids = np.array([_first_int(os.path.basename(p), i)
                               for i, p in enumerate(self._paths)], dtype=np.int64)
        self._cfg = cfg
        self._poses = None

    def __len__(self):
        return len(self._paths)

    def frame_ids(self):
        return self._fids

    def timestamps_s(self):
        return self._ts

    def image(self, i):
        from PIL import Image
        return Image.open(self._paths[int(i)]).convert("RGB")

    def poses(self):
        spec = self._cfg.get("poses")
        if not spec:
            return None
        if self._poses is None:
            kind = spec.get("kind", "tum")
            if kind == "tum":
                t, x, y, yaw = read_tum_poses(spec["path"], spec.get("axes", "xy"))
            elif kind == "csv":
                t, x, y, yaw = read_csv_poses(spec)
            else:
                raise ValueError(f"unknown poses.kind {kind!r}")
            # A rendered sequence's pose clock and its synthesised frame clock
            # need not share an origin; align both to start at zero.
            t = t - t[0]
            ft = self._ts - self._ts[0]
            if spec.get("rescale_time", True) and t[-1] > 0:
                ft = ft * (t[-1] / max(ft[-1], 1e-9))
            self._poses = interp_to(ft, t, x, y, yaw)
        return self._poses


def _natural_key(p):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", p)]


def _first_int(s, default):
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else default


# --------------------------------------------------------------------------

def load_sequence(path_or_cfg) -> Sequence:
    """Build a Sequence from a config dict or a path to a JSON config."""
    cfg = path_or_cfg
    if isinstance(cfg, (str, bytes, os.PathLike)):
        with open(cfg) as f:
            cfg = json.load(f)
    kind = cfg.get("kind")
    if kind is None:
        kind = "huggingface" if ("rgb_config" in cfg or "repo" in cfg) else "folder"
    if kind == "huggingface":
        return HuggingFaceSequence(cfg)
    if kind == "folder":
        return FolderSequence(cfg)
    raise ValueError(f"unknown dataset kind {kind!r} (expected 'huggingface' or 'folder')")


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pv = sub.add_parser("validate", help="load a dataset config and report what it supports")
    pv.add_argument("--dataset", required=True)
    pv.add_argument("--peek", type=int, default=3, help="decode this many frames as a check")
    args = ap.parse_args()

    seq = load_sequence(args.dataset)
    print(seq.describe())
    print(f"  out_dir: {seq.out_dir}")

    n = min(args.peek, len(seq))
    sizes = set()
    for i in np.linspace(0, len(seq) - 1, n).astype(int):
        sizes.add(seq.image(int(i)).size)
    print(f"  decoded {n} frames OK; sizes {sorted(sizes)}")

    q = seq.pose_quality()
    print("\nanalyses available:")
    print("  image-only  (crop features, isotropy ladder, Timkey, encoder comparison,")
    print("               crop retrieval)                                    YES")
    ok = q is not None and q.usable
    print(f"  pose-based  (crosstalk, place recall, memory build, video recall) "
          f"{'YES' if ok else 'NO'}")
    if q is not None and not q.usable:
        print("    poses present but rejected — see verdict above")
    elif q is None:
        print("    no pose file configured")


if __name__ == "__main__":
    main()
