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
from .config import (
    ALLOWED_TEXT_FIELDS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LR,
    DEFAULT_NUM_WORKERS,
    DEFAULT_PATIENCE,
    DEFAULT_SEED,
    DEFAULT_WEIGHT_DECAY,
    MAX_LOGIT_SCALE,
)
from .data import (
    MimicCLIPDataset,
    build_dataloaders,
    drop_unreadable_rows,
    load_validation_df,
    load_semantic_embeddings,
    get_study_id,
)
from .clip_utils import load_clip, clip_features
from .losses import (
    calibrate_soft_temp,
    study_level_contrastive_loss,
    soft_clip_hybrid_loss,
)
from .metrics import (
    build_text_group_ids,
    calculate_study_level_metrics,
    extract_val_features,
    normalize_report,
    run_retrieval_eval,
)
from .trainer import fit, save_run_config, set_seed

__all__ = [
    "ExperimentConfig",
    "BASE_DATA_DIR",
    "IMAGE_DIR",
    "TRAIN_CSV_PATH",
    "VAL_CSV_PATH",
    "CHECKPOINT_ROOT",
    "CLIP_MODEL_NAME",
    "embeddings_path",
    "ALLOWED_TEXT_FIELDS",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_EPOCHS",
    "DEFAULT_LR",
    "DEFAULT_NUM_WORKERS",
    "DEFAULT_PATIENCE",
    "DEFAULT_SEED",
    "DEFAULT_WEIGHT_DECAY",
    "MAX_LOGIT_SCALE",
    "MimicCLIPDataset",
    "build_dataloaders",
    "drop_unreadable_rows",
    "load_validation_df",
    "load_semantic_embeddings",
    "get_study_id",
    "load_clip",
    "clip_features",
    "calibrate_soft_temp",
    "study_level_contrastive_loss",
    "soft_clip_hybrid_loss",
    "build_text_group_ids",
    "calculate_study_level_metrics",
    "extract_val_features",
    "normalize_report",
    "run_retrieval_eval",
    "fit",
    "save_run_config",
    "set_seed",
]
