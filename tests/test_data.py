import pandas as pd
import torch
from torch.utils.data import DataLoader

from vsa_cognitive_mapping.data import (ONEHOT_COLS, CachedTransitionDataset,
                                        TransitionDataset, load_image, load_transitions)


def test_item_matches_csv_row(bezier_root):
    ds = TransitionDataset(bezier_root, split="train", val_fraction=0.0, img_size=56)
    csv = pd.read_csv(bezier_root / "transitions_gt.csv")
    for i in (0, 42, len(ds) - 1):
        img_t, img_tp1, action = ds[i]
        row = csv.iloc[i]
        assert torch.equal(action, torch.tensor(row[ONEHOT_COLS].to_numpy(dtype="float32")))
        assert torch.equal(img_t, load_image(bezier_root / row["image_t"], 56))
        assert torch.equal(img_tp1, load_image(bezier_root / row["image_tp1"], 56))


def test_split_is_contiguous_and_disjoint(bezier_root):
    train = load_transitions(bezier_root, "train", val_fraction=0.2)
    val = load_transitions(bezier_root, "val", val_fraction=0.2)
    full = pd.read_csv(bezier_root / "transitions_gt.csv")
    assert len(train) + len(val) == len(full)
    assert set(train["frame_t"]).isdisjoint(set(val["frame_t"]))
    assert train["frame_t"].max() < val["frame_t"].min()  # val is the tail


def test_loader_yields_len_batches(tiny_dataset):
    # 8 transitions, batch 4 -> exactly 2 batches (the divisible case that
    # topjepa's REV04 off-by-one silently dropped).
    cache = torch.load(tiny_dataset / "embeddings_dinov2_vits14.pt")
    ds = CachedTransitionDataset(tiny_dataset, cache, split="train", val_fraction=0.0)
    assert len(ds) == 8
    loader = DataLoader(ds, batch_size=4)
    assert sum(1 for _ in loader) == len(loader) == 2


def test_frame_skip_chains_same_action_runs(bezier_root):
    df1 = load_transitions(bezier_root, "train", val_fraction=0.0, frame_skip=1)
    df2 = load_transitions(bezier_root, "train", val_fraction=0.0, frame_skip=2)
    expected = sum(df1["action"].iloc[i] == df1["action"].iloc[i + 1] for i in range(len(df1) - 1))
    assert len(df2) == expected
    first = df2.iloc[0]
    i = df1.index[df1["frame_t"] == first["frame_t"]][0]
    assert first["image_t"] == df1.iloc[i]["image_t"]
    assert first["image_tp1"] == df1.iloc[i + 1]["image_tp1"]  # skips one frame


def test_seeded_shuffle_is_deterministic(tiny_dataset):
    cache = torch.load(tiny_dataset / "embeddings_dinov2_vits14.pt")
    ds = CachedTransitionDataset(tiny_dataset, cache, split="train", val_fraction=0.0)

    def order(seed):
        loader = DataLoader(ds, batch_size=2, shuffle=True,
                            generator=torch.Generator().manual_seed(seed))
        return torch.cat([a.argmax(-1) for _, _, a in loader])

    assert torch.equal(order(0), order(0))
