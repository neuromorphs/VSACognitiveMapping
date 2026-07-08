"""Evaluate a trained JEPA world model on the validation split.

    python scripts/eval.py --config configs/jepa_sim.yaml --checkpoint checkpoints/jepa_sim/final.pt

Reports (and writes JSON next to the checkpoint):
- teacher-forced latent MSE,
- rel_mse_vs_persistence: MSE(predictor) / MSE(ẑ_{t+1} := z_t) — the headline
  number; < 1.0 means learned action-conditioned dynamics (raw MSE alone is
  uninterpretable: a collapsed model has excellent raw MSE),
- rel_mse_vs_mean: same against the constant mean-latent baseline,
- per-action rel-MSE (the sim data is ~65% `forward`; this exposes a model
  that ignores turn actions),
- collapse diagnostics on the val latents.

With --save-embeddings PATH, also dumps the per-frame z_t/z_tp1 latents,
actions, timesteps (frame ids), and ground-truth position (x, y, z, yaw, if
transitions.csv is present) for the train and val splits, for downstream use
(e.g. binding into an HD associative memory — see scripts/associative_memory.py).
"""

import argparse
import json
from pathlib import Path

import torch
import yaml

from vsa_cognitive_mapping.data import ACTIONS, CachedTransitionDataset, load_pose_by_frame
from vsa_cognitive_mapping.diagnostics import collapse_metrics
from vsa_cognitive_mapping.encoder import load_or_build_cache
from vsa_cognitive_mapping.model import JEPAWorldModel


@torch.no_grad()
def _encode_split(model: JEPAWorldModel, cache: dict, data_cfg: dict, split: str, device: str) -> dict:
    """One split's latents + actions + timesteps + (if available) pose."""
    dataset = CachedTransitionDataset(data_cfg["root"], cache, split=split,
                                      val_fraction=data_cfg.get("val_fraction", 0.2),
                                      frame_skip=data_cfg.get("frame_skip", 1))
    emb_t, emb_tp1, action = dataset.emb_t.to(device), dataset.emb_tp1.to(device), dataset.actions.to(device)
    z_t, z_tp1 = model.encode(emb_t), model.encode(emb_tp1)
    z_pred = model.predictor(z_t, action)

    pose = load_pose_by_frame(data_cfg["root"])
    pos_t = pos_tp1 = None
    if pose is not None:
        pos_t = torch.tensor([pose[int(f)] for f in dataset.df["frame_t"]], dtype=torch.float32)
        pos_tp1 = torch.tensor([pose[int(f)] for f in dataset.df["frame_tp1"]], dtype=torch.float32)

    return {
        "z_t": z_t.cpu(), "z_tp1": z_tp1.cpu(), "z_pred": z_pred.cpu(), "action": dataset.actions,
        "frame_t": torch.tensor(dataset.df["frame_t"].to_numpy()),
        "frame_tp1": torch.tensor(dataset.df["frame_tp1"].to_numpy()),
        "image_t": dataset.df["image_t"].tolist(), "image_tp1": dataset.df["image_tp1"].tolist(),
        "pos_t": pos_t, "pos_tp1": pos_tp1,
    }


@torch.no_grad()
def evaluate(cfg: dict, checkpoint: str, save_embeddings: str | None = None) -> dict:
    data_cfg, model_cfg = cfg["data"], cfg["model"]
    device = cfg.get("device", "cpu")

    cache = load_or_build_cache(data_cfg["root"], model_cfg.get("backbone", "dinov2_vits14"),
                                img_size=data_cfg.get("img_size", 224), device=device)

    model = JEPAWorldModel.from_config(model_cfg).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()  # BatchNorm uses running stats: single-observation-safe

    val_split = _encode_split(model, cache, data_cfg, "val", device)
    if save_embeddings is not None:
        train_split = _encode_split(model, cache, data_cfg, "train", device)
        if val_split["pos_t"] is None:
            print(f"note: no transitions.csv under {data_cfg['root']} — saving without position data")
        torch.save({"checkpoint": str(checkpoint), "backbone": model_cfg.get("backbone", "dinov2_vits14"),
                   "train": train_split, "val": val_split}, save_embeddings)
        print(f"embeddings written to {save_embeddings}")

    n_val = len(val_split["z_t"])
    action = val_split["action"].to(device)
    z_t, z_tp1, z_pred = (val_split["z_t"].to(device), val_split["z_tp1"].to(device),
                          val_split["z_pred"].to(device))

    per_sample = lambda a, b: (a - b).square().mean(dim=-1)
    mse = per_sample(z_pred, z_tp1)
    persistence = per_sample(z_t, z_tp1)
    mean_baseline = per_sample(z_tp1.mean(dim=0, keepdim=True).expand_as(z_tp1), z_tp1)

    per_action = {}
    labels = action.argmax(dim=-1)
    for i, name in enumerate(ACTIONS):
        mask = labels == i
        if mask.any():
            per_action[name] = {"n": int(mask.sum()),
                                "rel_mse_vs_persistence": (mse[mask].mean() / persistence[mask].mean()).item()}

    results = {
        "checkpoint": str(checkpoint),
        "n_val_transitions": n_val,
        "latent_mse": mse.mean().item(),
        "rel_mse_vs_persistence": (mse.mean() / persistence.mean()).item(),
        "rel_mse_vs_mean": (mse.mean() / mean_baseline.mean()).item(),
        "per_action": per_action,
        "collapse": collapse_metrics(torch.cat([z_t, z_tp1])),
    }
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--save-embeddings", default=None,
                        help="path to dump per-frame z_t/z_tp1/action/frame_id/position for train+val")
    args = parser.parse_args()
    with open(args.config) as f:
        results = evaluate(yaml.safe_load(f), args.checkpoint, save_embeddings=args.save_embeddings)
    print(json.dumps(results, indent=2))
    out = Path(args.checkpoint).parent / "eval.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"written to {out}")
