import pytest
import torch
import yaml

from vsa_cognitive_mapping.predictor import MLPPredictor, make_predictor


def test_output_shape():
    torch.manual_seed(0)
    pred = MLPPredictor(latent_dim=128)
    z, a = torch.randn(8, 128), torch.eye(4)[torch.randint(0, 4, (8,))]
    assert pred(z, a).shape == (8, 128)


def test_construction_at_config_defaults(config_path):
    # topjepa lesson (TEST01): bugs hid because no test used the shipped config.
    cfg = yaml.safe_load(config_path.read_text())["model"]
    pred = make_predictor(cfg["predictor"], cfg["latent_dim"])
    z, a = torch.randn(4, cfg["latent_dim"]), torch.eye(4)
    assert pred(z, a).shape == (4, cfg["latent_dim"])


def test_unknown_kind_raises():
    with pytest.raises(ValueError, match="unknown predictor kind"):
        make_predictor({"kind": "nope"}, 128)
