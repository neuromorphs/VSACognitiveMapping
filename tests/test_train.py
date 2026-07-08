"""End-to-end: real training/eval runs (subprocess) on the tiny dataset with a
fake embedding cache — no network, no backbone download."""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def make_config(tmp_path, tiny_dataset, epochs):
    cfg = yaml.safe_load((REPO_ROOT / "configs" / "jepa_sim.yaml").read_text())
    cfg["data"]["root"] = str(tiny_dataset)
    cfg["train"].update(epochs=epochs, batch_size=4, warmup_steps=2, log_every=1,
                        checkpoint_every_epochs=2, out_dir=str(tmp_path / "ckpt"))
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path, cfg


def run(script, *args):
    return subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / script), *args],
                          capture_output=True, text=True, cwd=REPO_ROOT, check=True)


@pytest.mark.slow
def test_train_resume_eval(tmp_path, tiny_dataset):
    config, _ = make_config(tmp_path, tiny_dataset, epochs=4)
    out_dir = tmp_path / "ckpt"

    run("train.py", "--config", str(config))
    assert (out_dir / "final.pt").exists()
    metrics = pd.read_csv(out_dir / "metrics.csv")
    assert bool(metrics["loss"].notna().all())
    assert bool((metrics["sigreg"] >= 0).all())
    steps_after_first_run = metrics["step"].max()

    # resume continues (more epochs in the config, same checkpoint)
    config2, _ = make_config(tmp_path, tiny_dataset, epochs=8)
    result = run("train.py", "--config", str(config2), "--resume", str(out_dir / "checkpoint.pt"))
    assert "resumed" in result.stdout
    metrics = pd.read_csv(out_dir / "metrics.csv")
    assert metrics["step"].max() > steps_after_first_run

    run("eval.py", "--config", str(config2), "--checkpoint", str(out_dir / "final.pt"))
    results = json.loads((out_dir / "eval.json").read_text())
    assert {"latent_mse", "rel_mse_vs_persistence", "rel_mse_vs_mean",
            "per_action", "collapse"} <= set(results)
