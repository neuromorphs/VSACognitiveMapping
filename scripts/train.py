"""Train the JEPA world model (head + predictor) on cached backbone embeddings.

    python scripts/train.py --config configs/jepa_sim.yaml [--resume PATH]

Logs to stdout and appends to <out_dir>/metrics.csv. Checkpoints contain
model + optimizer + scheduler + step so --resume continues exactly;
<out_dir>/final.pt is weights-only for inference/eval.
"""

import argparse
import csv
import math
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from vsa_cognitive_mapping.data import CachedTransitionDataset
from vsa_cognitive_mapping.encoder import load_or_build_cache
from vsa_cognitive_mapping.model import JEPAWorldModel

CSV_FIELDS = ["step", "epoch", "loss", "mse", "sigreg",
              "per_dim_std", "effective_rank", "mean_pairwise_cosine", "lr"]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def param_groups(model: torch.nn.Module, weight_decay: float) -> list[dict]:
    """AdamW decays weight matrices only — never biases or BatchNorm scales."""
    decay = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


def warmup_cosine(warmup_steps: int, total_steps: int):
    """LR multiplier: linear 0->1 over warmup, cosine 1->0 over the rest."""
    def fn(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return fn


def train(cfg: dict, resume: str | None = None, init: str | None = None) -> Path:
    seed_everything(cfg["seed"])
    device = cfg.get("device", "cpu")
    data_cfg, model_cfg, train_cfg = cfg["data"], cfg["model"], cfg["train"]
    out_dir = Path(train_cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    cache = load_or_build_cache(data_cfg["root"], model_cfg.get("backbone", "dinov2_vits14"),
                                img_size=data_cfg.get("img_size", 224), device=device)
    train_set = CachedTransitionDataset(data_cfg["root"], cache, split="train",
                                        val_fraction=data_cfg.get("val_fraction", 0.2),
                                        frame_skip=data_cfg.get("frame_skip", 1))
    loader = DataLoader(train_set, batch_size=train_cfg["batch_size"], shuffle=True,
                        drop_last=len(train_set) > train_cfg["batch_size"],
                        generator=torch.Generator().manual_seed(cfg["seed"]))

    model = JEPAWorldModel.from_config(model_cfg).to(device)
    if init is not None:  # warm start (e.g. finetuning): weights only, fresh optimizer
        state = torch.load(init, map_location=device)
        model.load_state_dict(state["model"] if "model" in state else state)
        print(f"initialized weights from {init}")
    optimizer = torch.optim.AdamW(param_groups(model, train_cfg.get("weight_decay", 0.01)),
                                  lr=train_cfg["lr"])
    total_steps = len(loader) * train_cfg["epochs"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, warmup_cosine(train_cfg.get("warmup_steps", 0), total_steps))

    step, start_epoch = 0, 0
    if resume is not None:
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        step, start_epoch = ckpt["step"], ckpt["epoch"]
        print(f"resumed from {resume} at epoch {start_epoch}, step {step}")

    metrics_file = out_dir / "metrics.csv"
    if not metrics_file.exists():
        with open(metrics_file, "w", newline="") as f:
            csv.DictWriter(f, CSV_FIELDS).writeheader()

    grad_clip = train_cfg.get("grad_clip", 1.0)
    log_every = train_cfg.get("log_every", 10)
    ckpt_every = train_cfg.get("checkpoint_every_epochs", 10)

    model.train()
    for epoch in range(start_epoch, train_cfg["epochs"]):
        for emb_t, emb_tp1, action in loader:
            loss, metrics = model.loss(emb_t.to(device), emb_tp1.to(device), action.to(device))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            step += 1

            if step % log_every == 0 or step == 1:
                row = {"step": step, "epoch": epoch, "lr": scheduler.get_last_lr()[0], **metrics}
                print("  ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                                for k, v in row.items()))
                with open(metrics_file, "a", newline="") as f:
                    csv.DictWriter(f, CSV_FIELDS).writerow(row)

        if (epoch + 1) % ckpt_every == 0 or epoch + 1 == train_cfg["epochs"]:
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "step": step, "epoch": epoch + 1},
                       out_dir / "checkpoint.pt")

    torch.save(model.state_dict(), out_dir / "final.pt")
    print(f"done — {out_dir / 'final.pt'}")
    return out_dir / "final.pt"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--init", default=None,
                        help="checkpoint to initialize weights from (fresh optimizer; for finetuning)")
    args = parser.parse_args()
    with open(args.config) as f:
        train(yaml.safe_load(f), resume=args.resume, init=args.init)
