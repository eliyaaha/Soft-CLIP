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

# ---------------------------------------------------------------------------
# Shared optimisation defaults.
#
# These MUST be identical for the hard and soft arms, otherwise the two are not
# comparable: batch size sets the number of in-batch negatives (and the support
# of the soft-target distribution), and the learning rate interacts with alpha,
# since only (1 - alpha) of the gradient reaches the hard term.
#
# Both train_baseline.py and train_soft_clip.py import these, so there is a
# single source of truth and the two arms cannot silently drift apart.
# ---------------------------------------------------------------------------
DEFAULT_BATCH_SIZE = 256
DEFAULT_LR = 5e-6
DEFAULT_WEIGHT_DECAY = 0.2
DEFAULT_EPOCHS = 10
DEFAULT_PATIENCE = 2
DEFAULT_NUM_WORKERS = 4
DEFAULT_SEED = 42

# OpenAI CLIP caps the learned temperature at 100. Without this cap, the KL
# term can satisfy a near-uniform soft target simply by shrinking logit_scale,
# which flattens every similarity in the space -- including at evaluation time.
MAX_LOGIT_SCALE = 100.0


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
    # NOTE: "text" (the full report) is what the argparse layer has always
    # passed, so this default now matches what the experiments actually ran.
    text_field: str = "text"

    batch_size: int = DEFAULT_BATCH_SIZE
    num_workers: int = DEFAULT_NUM_WORKERS
    learning_rate: float = DEFAULT_LR
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    epochs: int = DEFAULT_EPOCHS
    patience: int = DEFAULT_PATIENCE

    seed: int = DEFAULT_SEED

    # Which quantity early stopping monitors. "hard" is the default because it
    # is defined identically for both arms, so checkpoints selected across
    # different alpha values (and against the baseline) are comparable. "total"
    # restores the previous behaviour of monitoring the blended loss.
    early_stop_metric: str = "hard"

    # Soft-CLIP only
    alpha: float = 0.5
    soft_temp: float = 0.1
    soft_top_k: Optional[int] = None
    soft_threshold: Optional[float] = None
    text_similarity_weight: float = 1.0
    calibrate_temp: Optional[float] = None
    shuffle_embeddings: bool = False
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
            if self.soft_threshold is not None:
                parts.append(f"thr{self.soft_threshold}")
                parts.append(f"tw{self.text_similarity_weight}")
            if self.calibrate_temp is not None:
                parts.append(f"cal{self.calibrate_temp}")
            # Must be in the name, otherwise the control run overwrites the
            # real run's checkpoint directory.
            if self.shuffle_embeddings:
                parts.append("shuffled")
        if self.seed != DEFAULT_SEED:
            parts.append(f"s{self.seed}")
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

        if self.soft_top_k is not None and self.soft_threshold is not None:
            raise ValueError(
                "soft_top_k and soft_threshold cannot be used together."
            )

        if self.soft_threshold is not None:
            if not -1.0 <= self.soft_threshold <= 1.0:
                raise ValueError(
                    "soft_threshold must be between -1 and 1."
                )

        if not 0.0 <= self.text_similarity_weight <= 1.0:
            raise ValueError(
                "text_similarity_weight must be between 0 and 1."
            )

        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1]. Received {self.alpha}.")

        if self.early_stop_metric not in ("hard", "total"):
            raise ValueError(
                f"early_stop_metric={self.early_stop_metric!r} not in ('hard', 'total')"
            )

        if self.calibrate_temp is not None and not 0.0 < self.calibrate_temp < 1.0:
            raise ValueError(
                "calibrate_temp is a target diagonal mass and must be in (0, 1). "
                f"Received {self.calibrate_temp}."
            )

        if self.calibrate_temp is not None and self.soft_threshold is not None:
            raise ValueError(
                "calibrate_temp only applies to softmax soft targets, not to "
                "threshold mode."
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
            f"  seed            : {self.seed}",
            f"  early_stop_on   : {self.early_stop_metric}",
        ]
        if self.loss_type == "soft":
            lines.extend([
                f"  alpha           : {self.alpha}",
                f"  soft_temp       : {self.soft_temp}",
                f"  soft_top_k      : {self.soft_top_k}",
                f"  soft_threshold         : {self.soft_threshold}",
                f"  text_similarity_weight : {self.text_similarity_weight}",
                f"  calibrate_temp  : {self.calibrate_temp}",
                f"  shuffle_embeds  : {self.shuffle_embeddings}",
                f"  embeddings_tag  : {self.embeddings_tag}",
                f"  train embeds    : {self.train_embeddings_path}",
                f"  val embeds      : {self.val_embeddings_path}",
            ])
            if self.alpha == 0.0:
                lines.append(
                    "  >> alpha=0: this run MUST reproduce the hard baseline "
                    "exactly (same seed/batch/lr). Use it as the control."
                )
            if self.shuffle_embeddings:
                lines.append(
                    "  >> SHUFFLE CONTROL: semantic embeddings are permuted, "
                    "so soft targets carry no image-text correspondence."
                )
        lines.append(f"  run_name        : {self.run_name}")
        lines.append(f"  checkpoint_dir  : {self.checkpoint_dir}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialisable view, persisted next to each checkpoint."""
        import dataclasses

        return {k: v for k, v in dataclasses.asdict(self).items()}
