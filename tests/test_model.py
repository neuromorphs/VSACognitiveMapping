import copy

import torch
import yaml

from vsa_cognitive_mapping.model import JEPAWorldModel


def make_batch(n=32, backbone_dim=384, seed=0):
    g = torch.Generator().manual_seed(seed)
    emb_t = torch.randn(n, backbone_dim, generator=g)
    emb_tp1 = emb_t + 0.1 * torch.randn(n, backbone_dim, generator=g)
    action = torch.eye(4)[torch.randint(0, 4, (n,), generator=g)]
    return emb_t, emb_tp1, action


def test_loss_and_metrics(config_path):
    torch.manual_seed(0)
    model = JEPAWorldModel.from_config(yaml.safe_load(config_path.read_text())["model"])
    loss, metrics = model.loss(*make_batch())
    assert loss.isfinite()
    assert set(metrics) == {"loss", "mse", "sigreg", "per_dim_std",
                            "effective_rank", "mean_pairwise_cosine"}
    assert metrics["sigreg"] >= 0.0


def test_training_reduces_mse_and_updates_only_trained_params(config_path):
    torch.manual_seed(0)
    model = JEPAWorldModel.from_config(yaml.safe_load(config_path.read_text())["model"])
    before = copy.deepcopy(model.state_dict())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    batch = make_batch()
    mses = []
    for _ in range(100):
        loss, metrics = model.loss(*batch)
        mses.append(metrics["mse"])
        opt.zero_grad()
        loss.backward()
        opt.step()

    assert sum(mses[-5:]) < sum(mses[:5])  # overfits the fixed batch
    after = model.state_dict()
    assert not torch.equal(before["head.proj.weight"], after["head.proj.weight"])
    assert not torch.equal(before["predictor.action_embed.weight"],
                           after["predictor.action_embed.weight"])
    # SIGReg buffers are constants and must never move (topjepa BUG04)
    assert torch.equal(before["sigreg.weights"], after["sigreg.weights"])
