"""Version-agnostic access to CLIP's projected features.

Older transformers return a plain tensor from CLIPModel.get_text_features /
get_image_features; newer versions (Colab) return a ModelOutput wrapper.
Worse, the wrapper's fields are ambiguous: its pooler_output may already BE
the projected embedding (observed 512-d on the vision tower, whose raw
width is 768), and on the text tower raw and projected widths coincide
(512 == 512) so a wrong guess would run silently. So: take a plain tensor
or an explicit *_embeds field at face value; for anything else, bypass the
convenience method and recompute from the base towers + projection heads,
which is exact and stable across versions.
"""
from __future__ import annotations

import torch


def text_features(model, tok):
    out = model.get_text_features(**tok)
    if torch.is_tensor(out):
        return out
    v = getattr(out, "text_embeds", None)
    if v is not None and torch.is_tensor(v):
        return v
    enc = model.text_model(input_ids=tok["input_ids"],
                           attention_mask=tok.get("attention_mask"))
    return model.text_projection(enc.pooler_output)


def image_features(model, pix):
    out = model.get_image_features(**pix)
    if torch.is_tensor(out):
        return out
    v = getattr(out, "image_embeds", None)
    if v is not None and torch.is_tensor(v):
        return v
    enc = model.vision_model(pixel_values=pix["pixel_values"])
    return model.visual_projection(enc.pooler_output)
