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


ORIG_INDEX_COL = "_orig_row"


def get_study_id(image_path: str) -> str:
    match = re.search(r"/s(\d+)/", image_path)
    return match.group(1) if match else "unknown"


class MimicCLIPDataset(Dataset):
    """Single dataset class used for both hard- and soft-CLIP training.

    Always returns ``(image, text, study_id, orig_index)``. The hard-loss path
    ignores ``orig_index``; the soft-loss path uses it to look up the matching
    precomputed BERT embedding.

    ``orig_index`` is the row's position in the *unfiltered* processed CSV, which
    is the order ``create_embeddings.py`` wrote the ``.pt`` tensors in. Keeping
    the original index means rows can be dropped (e.g. missing image files)
    without silently misaligning every semantic embedding lookup.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        base_image_dir: str = IMAGE_DIR,
        text_field: str = "text",
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
        if ORIG_INDEX_COL not in dataframe.columns:
            raise ValueError(
                f"DataFrame is missing {ORIG_INDEX_COL!r}. Load it through "
                f"load_validation_df() / build_dataloaders() so the original "
                f"CSV row index is preserved for embedding lookup."
            )
        self.df = dataframe.reset_index(drop=True)
        self.base_dir = base_image_dir
        self.text_field = text_field
        self.transform = _CLIP_TRANSFORM

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        # NOTE: this used to walk backwards to a neighbouring row whenever an
        # image was missing or unreadable, and return THAT row instead. Because
        # run_retrieval_eval takes features in dataset order but study ids
        # positionally from df_val, any such substitution silently desynchronised
        # features from their labels -- and it looped forever if every image was
        # missing. Unreadable rows are now dropped up front in build_dataloaders,
        # and anything that still fails here raises loudly.
        row = self.df.iloc[idx]
        img_path_rel = row["image"]
        img_path_full = os.path.join(self.base_dir, img_path_rel)

        try:
            image = Image.open(img_path_full).convert("RGB")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to read image at row {idx}: {img_path_full}. "
                f"Rows with missing files are dropped in build_dataloaders(); "
                f"this one became unreadable afterwards."
            ) from exc

        image = self.transform(image)
        text_signal = str(row[self.text_field])
        study_id = get_study_id(img_path_rel)
        return image, text_signal, study_id, int(row[ORIG_INDEX_COL])


def _read_processed_csv(path: str, what: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Processed {what} CSV not found at {path}. "
            f"Run `python preprocess.py` first."
        )
    df = pd.read_csv(path).fillna("")
    # Record the pre-filter row index; this is the row order the precomputed
    # embedding tensors were written in.
    df[ORIG_INDEX_COL] = range(len(df))
    return df


def load_validation_df() -> pd.DataFrame:
    return _read_processed_csv(VAL_CSV_PATH, "validation")


def _load_training_df() -> pd.DataFrame:
    return _read_processed_csv(TRAIN_CSV_PATH, "training")


def drop_unreadable_rows(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Drop rows whose image file is missing, loudly.

    Previously handled by a silent fallback inside ``__getitem__`` that returned
    a *different* row, which desynchronised evaluation features from their labels.
    Filtering here is safe because ``ORIG_INDEX_COL`` preserves the mapping back
    into the precomputed embedding tensors.
    """
    exists = df["image"].apply(
        lambda p: os.path.exists(os.path.join(IMAGE_DIR, str(p)))
    )
    n_missing = int((~exists).sum())
    if n_missing:
        print(
            f"[{name}] WARNING: dropping {n_missing:,} of {len(df):,} rows with "
            f"missing image files. Report the post-filter size as the candidate "
            f"pool, not the raw CSV size."
        )
    df = df[exists].reset_index(drop=True)

    study_ids = df["image"].apply(get_study_id)
    n_unknown = int((study_ids == "unknown").sum())
    if n_unknown:
        raise ValueError(
            f"[{name}] {n_unknown:,} image paths do not match the /s<digits>/ "
            f"pattern, so get_study_id() returns 'unknown' for them. All such "
            f"rows would count as mutual matches and inflate recall."
        )
    return df


def build_dataloaders(
    config: ExperimentConfig,
    mode: str,
) -> Tuple[Optional[DataLoader], DataLoader, pd.DataFrame]:
    """Build train (optional) and validation dataloaders.

    Returns ``(train_loader_or_None, val_loader, df_val)``. ``df_val`` is the
    *filtered* frame, positionally aligned with the order the val loader yields
    samples in -- which is what run_retrieval_eval relies on.
    """
    generator = torch.Generator()
    generator.manual_seed(config.seed)

    df_val = drop_unreadable_rows(load_validation_df(), "val")
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
        df_train = drop_unreadable_rows(_load_training_df(), "train")
        train_dataset = MimicCLIPDataset(
            df_train, IMAGE_DIR, text_field=config.text_field
        )
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
            generator=generator,
        )

    return train_loader, val_loader, df_val


def load_semantic_embeddings(path: str) -> torch.Tensor:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Precomputed embeddings file not found at {path}. "
            f"Generate it with `python create_embeddings.py` "
            f"(see --model / --field flags)."
        )
    return torch.load(path, map_location="cpu", weights_only=True)
