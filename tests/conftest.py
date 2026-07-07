from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
ACTIONS = ("forward", "stop", "left", "right")
CSV_HEADER = ("frame_t,frame_tp1,image_t,action,"
              "onehot_forward,onehot_stop,onehot_left,onehot_right,image_tp1")


@pytest.fixture
def bezier_root() -> Path:
    return REPO_ROOT / "data" / "dataset_vjepa_bezier"


@pytest.fixture
def config_path() -> Path:
    return REPO_ROOT / "configs" / "jepa_sim.yaml"


@pytest.fixture
def tiny_dataset(tmp_path: Path) -> Path:
    """A miniature dataset in the common format: 9 frames, 8 transitions,
    plus a fake DINOv2 embedding cache so nothing needs network access."""
    root = tmp_path / "tiny"
    (root / "images").mkdir(parents=True)
    rng = np.random.default_rng(0)

    n_frames = 9
    paths = []
    for i in range(1, n_frames + 1):
        p = f"images/frame_{i:06d}.png"
        Image.fromarray(rng.integers(0, 255, (16, 16, 4), dtype=np.uint8), "RGBA").save(root / p)
        paths.append(p)

    rows = [CSV_HEADER]
    for t in range(1, n_frames):
        action = ACTIONS[(t - 1) % len(ACTIONS)]
        onehot = ",".join(str(int(a == action)) for a in ACTIONS)
        rows.append(f"{t},{t + 1},{paths[t - 1]},{action},{onehot},{paths[t]}")
    (root / "transitions_gt.csv").write_text("\n".join(rows) + "\n")

    torch.manual_seed(0)
    cache = {p: torch.randn(384) for p in paths}
    torch.save(cache, root / "embeddings_dinov2_vits14.pt")
    return root
