"""Shared utilities for MIMIC-CXR CLIP ablation experiments."""

from .config import (
    ExperimentConfig,
    BASE_DATA_DIR,
    IMAGE_DIR,
    TRAIN_CSV_PATH,
    VAL_CSV_PATH,
    CHECKPOINT_ROOT,
    CLIP_MODEL_NAME,
    embeddings_path,
)
from .data import (
    MimicCLIPDataset,
    build_dataloaders,
    load_validation_df,
    load_semantic_embeddings,
    get_study_id
)
from .clip_utils import load_clip, clip_features
from .losses import study_level_contrastive_loss, soft_clip_hybrid_loss
from .metrics import calculate_study_level_metrics, run_retrieval_eval
from .trainer import fit

__all__ = [
    "ExperimentConfig",
    "BASE_DATA_DIR",
    "IMAGE_DIR",
    "TRAIN_CSV_PATH",
    "VAL_CSV_PATH",
    "CHECKPOINT_ROOT",
    "CLIP_MODEL_NAME",
    "embeddings_path",
    "MimicCLIPDataset",
    "build_dataloaders",
    "load_validation_df",
    "load_semantic_embeddings",
    "load_clip",
    "clip_features",
    "study_level_contrastive_loss",
    "soft_clip_hybrid_loss",
    "calculate_study_level_metrics",
    "run_retrieval_eval",
    "fit",
]
