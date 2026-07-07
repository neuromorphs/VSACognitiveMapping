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
"""

import argparse
import json
from pathlib import Path

import torch
import yaml

from vsa_cognitive_mapping.data import ACTIONS, CachedTransitionDataset
from vsa_cognitive_mapping.diagnostics import collapse_metrics
from vsa_cognitive_mapping.encoder import load_or_build_cache
from vsa_cognitive_mapping.model import JEPAWorldModel


@torch.no_grad()
def evaluate(cfg: dict, checkpoint: str) -> dict:
    data_cfg, model_cfg = cfg["data"], cfg["model"]
    device = cfg.get("device", "cpu")

    cache = load_or_build_cache(data_cfg["root"], model_cfg.get("backbone", "dinov2_vits14"),
                                img_size=data_cfg.get("img_size", 224), device=device)
    val_set = CachedTransitionDataset(data_cfg["root"], cache, split="val",
                                      val_fraction=data_cfg.get("val_fraction", 0.2),
                                      frame_skip=data_cfg.get("frame_skip", 1))

    model = JEPAWorldModel.from_config(model_cfg).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()  # BatchNorm uses running stats: single-observation-safe

    emb_t, emb_tp1, action = val_set.emb_t.to(device), val_set.emb_tp1.to(device), val_set.actions.to(device)
    z_t = model.encode(emb_t)
    z_tp1 = model.encode(emb_tp1)
    z_pred = model.predictor(z_t, action)

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
        "n_val_transitions": len(val_set),
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
    args = parser.parse_args()
    with open(args.config) as f:
        results = evaluate(yaml.safe_load(f), args.checkpoint)
    print(json.dumps(results, indent=2))
    out = Path(args.checkpoint).parent / "eval.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"written to {out}")
