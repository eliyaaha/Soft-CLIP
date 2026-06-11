"""Shared paths, defaults, and the ExperimentConfig dataclass."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from preprocess import (
    BASE_DATA_DIR as _PP_BASE_DATA_DIR,
    OUTPUT_TRAIN_CSV_PATH,
    OUTPUT_VAL_CSV_PATH,
)


BASE_DATA_DIR = _PP_BASE_DATA_DIR
IMAGE_DIR = os.path.join(BASE_DATA_DIR, "official_data_iccv_final")
TRAIN_CSV_PATH = OUTPUT_TRAIN_CSV_PATH
VAL_CSV_PATH = OUTPUT_VAL_CSV_PATH

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_ROOT = os.path.join(PROJECT_ROOT, "checkpoints")

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# Allowed text fields for CLIP training / BERT embedding.
ALLOWED_TEXT_FIELDS = ("text", "findings_clean", "impression_clean")


def embeddings_path(split: str, tag: str) -> str:
    """Return the canonical path for a precomputed BERT embedding file.

    Path format: ``{BASE_DATA_DIR}/{split}_{tag}_embeddings.pt`` where
    ``tag`` is ``{model_slug}_{field}`` (e.g. ``biomedvlp_text``).
    """
    return os.path.join(BASE_DATA_DIR, f"{split}_{tag}_embeddings.pt")


@dataclass
class ExperimentConfig:
    """Single config object passed to the shared training/eval loops."""

    loss_type: str = "hard"
    text_field: str = "impression_clean"

    batch_size: int = 256
    num_workers: int = 4
    learning_rate: float = 5e-6
    weight_decay: float = 0.2
    epochs: int = 10
    patience: int = 2

    # Soft-CLIP only
    alpha: float = 0.5
    soft_temp: float = 0.1
    soft_top_k: Optional[int] = None
    embeddings_tag: Optional[str] = None
    train_embeddings_path: Optional[str] = None
    val_embeddings_path: Optional[str] = None

    run_name: Optional[str] = None
    checkpoint_dir: Optional[str] = None

    extra: dict = field(default_factory=dict)

    def resolve_embeddings_paths(self) -> None:
        """Fill in train/val embeddings paths from ``embeddings_tag`` if needed."""
        if self.loss_type != "soft":
            return
        if self.train_embeddings_path is None:
            if self.embeddings_tag is None:
                raise ValueError(
                    "Soft-CLIP requires either --embeddings-tag or "
                    "--train-embeddings / --val-embeddings paths."
                )
            self.train_embeddings_path = embeddings_path("train", self.embeddings_tag)
        if self.val_embeddings_path is None:
            if self.embeddings_tag is None:
                raise ValueError(
                    "Soft-CLIP requires either --embeddings-tag or "
                    "--train-embeddings / --val-embeddings paths."
                )
            self.val_embeddings_path = embeddings_path("val", self.embeddings_tag)

    def resolve_run_name(self) -> None:
        """Auto-generate a run name if none was provided."""
        if self.run_name:
            return
        parts = [self.loss_type, self.text_field]
        if self.loss_type == "soft":
            tag = self.embeddings_tag or "custom_embeddings"
            parts.append(tag)
            parts.append(f"a{self.alpha}")
            parts.append(f"t{self.soft_temp}")
            if self.soft_top_k is not None:
                parts.append(f"k{self.soft_top_k}")
        self.run_name = "_".join(str(p) for p in parts)

    def resolve_checkpoint_dir(self) -> None:
        if self.checkpoint_dir is None:
            self.checkpoint_dir = os.path.join(
                CHECKPOINT_ROOT, self.loss_type, self.run_name or "default"
            )

    def finalize(self) -> "ExperimentConfig":
        """Resolve all derived fields. Call once after argparse."""
        if self.text_field not in ALLOWED_TEXT_FIELDS:
            raise ValueError(
                f"text_field={self.text_field!r} not in {ALLOWED_TEXT_FIELDS}"
            )
        self.resolve_embeddings_paths()
        self.resolve_run_name()
        self.resolve_checkpoint_dir()
        return self

    def summary(self) -> str:
        lines = [
            f"  loss_type       : {self.loss_type}",
            f"  text_field      : {self.text_field}",
            f"  batch_size      : {self.batch_size}",
            f"  learning_rate   : {self.learning_rate}",
            f"  weight_decay    : {self.weight_decay}",
            f"  epochs          : {self.epochs}",
            f"  patience        : {self.patience}",
        ]
        if self.loss_type == "soft":
            lines.extend([
                f"  alpha           : {self.alpha}",
                f"  soft_temp       : {self.soft_temp}",
                f"  soft_top_k      : {self.soft_top_k}",
                f"  embeddings_tag  : {self.embeddings_tag}",
                f"  train embeds    : {self.train_embeddings_path}",
                f"  val embeds      : {self.val_embeddings_path}",
            ])
        lines.append(f"  run_name        : {self.run_name}")
        lines.append(f"  checkpoint_dir  : {self.checkpoint_dir}")
        return "\n".join(lines)
