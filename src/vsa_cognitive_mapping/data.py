"""Transition datasets for the common on-disk format.

A dataset directory contains `images/` and `transitions_gt.csv` with columns
`frame_t, frame_tp1, image_t, action, onehot_forward, onehot_stop,
onehot_left, onehot_right, image_tp1` (image paths relative to the dataset
directory). Both the simulator and the Spot recorder write this format.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

ACTIONS = ("forward", "stop", "left", "right")
ONEHOT_COLS = [f"onehot_{a}" for a in ACTIONS]

# Normalization the DINOv2 backbone was trained with (ImageNet statistics).
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_image(path: Path, img_size: int = 224) -> torch.Tensor:
    """PNG -> float32 (3, img_size, img_size), ImageNet-normalized."""
    img = Image.open(path).convert("RGB").resize((img_size, img_size), Image.Resampling.BILINEAR)
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x).permute(2, 0, 1)


def load_transitions(root: str | Path, split: str, val_fraction: float = 0.2,
                     frame_skip: int = 1) -> pd.DataFrame:
    """Read transitions_gt.csv and return the rows of one split.

    The split is by contiguous chunks (train = head, val = tail): temporally
    adjacent frames are near-duplicates, so a random split would leak.

    frame_skip > 1 chains k consecutive transitions into one longer-gap
    transition (frame_t of row i -> frame_tp1 of row i+k-1), only where all k
    actions are identical, so the composed action is well-defined.
    """
    if split not in ("train", "val"):
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")
    df = pd.read_csv(Path(root) / "transitions_gt.csv")

    if frame_skip > 1:
        rows = []
        for i in range(0, len(df) - frame_skip + 1):
            chunk = df.iloc[i:i + frame_skip]
            if chunk["action"].nunique() == 1:
                row = chunk.iloc[0].copy()
                row[["frame_tp1", "image_tp1"]] = chunk.iloc[-1][["frame_tp1", "image_tp1"]]
                rows.append(row)
        df = pd.DataFrame(rows).reset_index(drop=True)

    n_val = int(round(len(df) * val_fraction))
    return (df.iloc[:len(df) - n_val] if split == "train" else df.iloc[len(df) - n_val:]).reset_index(drop=True)


class TransitionDataset(Dataset):
    """One item = (img_t, img_tp1, action) for a single transition."""

    def __init__(self, root: str | Path, split: str = "train", val_fraction: float = 0.2,
                 img_size: int = 224, frame_skip: int = 1):
        self.root = Path(root)
        self.img_size = img_size
        self.df = load_transitions(root, split, val_fraction, frame_skip)
        self.actions = torch.from_numpy(self.df[ONEHOT_COLS].to_numpy(dtype=np.float32))

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        img_t = load_image(self.root / row["image_t"], self.img_size)
        img_tp1 = load_image(self.root / row["image_tp1"], self.img_size)
        return img_t, img_tp1, self.actions[i]


class CachedTransitionDataset(Dataset):
    """Like TransitionDataset, but items are precomputed backbone embeddings.

    `cache` maps the CSV's relative image path to its embedding (see
    encoder.build_embedding_cache). Training only touches head + predictor,
    so epochs never re-run the frozen backbone.
    """

    def __init__(self, root: str | Path, cache: dict[str, torch.Tensor], split: str = "train",
                 val_fraction: float = 0.2, frame_skip: int = 1):
        self.df = load_transitions(root, split, val_fraction, frame_skip)
        self.emb_t = torch.stack([cache[p] for p in self.df["image_t"]])
        self.emb_tp1 = torch.stack([cache[p] for p in self.df["image_tp1"]])
        self.actions = torch.from_numpy(self.df[ONEHOT_COLS].to_numpy(dtype=np.float32))

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        return self.emb_t[i], self.emb_tp1[i], self.actions[i]
