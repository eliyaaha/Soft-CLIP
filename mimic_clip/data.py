"""Unified dataset + dataloader builders for hard- and soft-CLIP."""

from __future__ import annotations

import os
import re
from typing import Optional, Tuple

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .config import (
    IMAGE_DIR,
    TRAIN_CSV_PATH,
    VAL_CSV_PATH,
    ALLOWED_TEXT_FIELDS,
    ExperimentConfig,
)


_CLIP_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    ),
])


def get_study_id(image_path: str) -> str:
    match = re.search(r"/s(\d+)/", image_path)
    return match.group(1) if match else "unknown"


class MimicCLIPDataset(Dataset):
    """Single dataset class used for both hard- and soft-CLIP training.

    Always returns ``(image, text, study_id, index)``. The hard-loss path
    ignores ``index``; the soft-loss path uses it to look up the matching
    precomputed BERT embedding.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        base_image_dir: str = IMAGE_DIR,
        text_field: str = "impression_clean",
    ) -> None:
        if text_field not in ALLOWED_TEXT_FIELDS:
            raise ValueError(
                f"text_field={text_field!r} not in {ALLOWED_TEXT_FIELDS}"
            )
        if text_field not in dataframe.columns:
            raise ValueError(
                f"DataFrame is missing column {text_field!r}; "
                f"available: {list(dataframe.columns)}"
            )
        self.df = dataframe.reset_index(drop=True)
        self.base_dir = base_image_dir
        self.text_field = text_field
        self.transform = _CLIP_TRANSFORM

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        current_idx = idx
        while True:
            row = self.df.iloc[current_idx]
            img_path_rel = row["image"]
            img_path_full = os.path.join(self.base_dir, img_path_rel)
            text_signal = str(row[self.text_field])

            if os.path.exists(img_path_full):
                try:
                    image = Image.open(img_path_full).convert("RGB")
                    image = self.transform(image)
                    study_id = get_study_id(img_path_rel)
                    return image, text_signal, study_id, current_idx
                except Exception:
                    pass
            current_idx = (current_idx - 1) % len(self.df)


def load_validation_df() -> pd.DataFrame:
    if not os.path.exists(VAL_CSV_PATH):
        raise FileNotFoundError(
            f"Processed validation CSV not found at {VAL_CSV_PATH}. "
            f"Run `python preprocess.py` first."
        )
    return pd.read_csv(VAL_CSV_PATH).fillna("")


def _load_training_df() -> pd.DataFrame:
    if not os.path.exists(TRAIN_CSV_PATH):
        raise FileNotFoundError(
            f"Processed training CSV not found at {TRAIN_CSV_PATH}. "
            f"Run `python preprocess.py` first."
        )
    return pd.read_csv(TRAIN_CSV_PATH).fillna("")


def build_dataloaders(
    config: ExperimentConfig,
    mode: str,
) -> Tuple[Optional[DataLoader], DataLoader, pd.DataFrame]:
    """Build train (optional) and validation dataloaders.

    Returns ``(train_loader_or_None, val_loader, df_val)``.
    """
    df_val = load_validation_df()
    val_dataset = MimicCLIPDataset(df_val, IMAGE_DIR, text_field=config.text_field)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    train_loader: Optional[DataLoader] = None
    if mode in ("train", "both"):
        df_train = _load_training_df()
        train_dataset = MimicCLIPDataset(
            df_train, IMAGE_DIR, text_field=config.text_field
        )
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
        )

    return train_loader, val_loader, df_val


def load_semantic_embeddings(path: str) -> torch.Tensor:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Precomputed embeddings file not found at {path}. "
            f"Generate it with `python create_embeddings.py` "
            f"(see --model / --field flags)."
        )
    return torch.load(path, map_location="cpu")
