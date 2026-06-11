"""Helpers to load CLIP and run the shared text+image forward pass."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from transformers import CLIPModel, CLIPProcessor

from .config import CLIP_MODEL_NAME


def load_clip(
    checkpoint: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> Tuple[CLIPModel, CLIPProcessor, torch.device]:
    """Load a CLIP model + processor from a checkpoint or the default HF id."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source = checkpoint or CLIP_MODEL_NAME
    processor = CLIPProcessor.from_pretrained(source)
    model = CLIPModel.from_pretrained(source).to(device)
    print(f"CLIP model loaded from {source} on device: {device}")
    return model, processor, device


def _to_features(outputs):
    return outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs


def clip_features(
    model: CLIPModel,
    processor: CLIPProcessor,
    images: torch.Tensor,
    texts,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run CLIP image + text encoders and return raw feature tensors."""
    clean_texts = [str(t) for t in texts]
    text_inputs = processor(
        text=clean_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77,
    ).to(device)
    pixel_values = images.to(device)

    image_outputs = model.get_image_features(pixel_values=pixel_values)
    text_outputs = model.get_text_features(**text_inputs)

    return _to_features(image_outputs), _to_features(text_outputs)
