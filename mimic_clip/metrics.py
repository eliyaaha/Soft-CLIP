"""Single shared retrieval evaluation used by every training script."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import tqdm
from transformers import CLIPModel, CLIPProcessor

from .clip_utils import clip_features
from .data import get_study_id


def calculate_study_level_metrics(
    sim_matrix: torch.Tensor,
    study_ids: np.ndarray,
) -> Tuple[float, float, float, float, float]:
    """Return (Recall@1, Recall@5, Recall@10, MedianRank, MRR)."""
    sim_matrix_np = sim_matrix.cpu().numpy()
    r1 = r5 = r10 = mrr = 0.0
    ranks = []
    num_queries = sim_matrix_np.shape[0]

    for i in range(num_queries):
        query_study = study_ids[i]
        sorted_indices = np.argsort(-sim_matrix_np[i])
        retrieved_studies = study_ids[sorted_indices]
        matches = retrieved_studies == query_study

        first_match_rank = np.where(matches)[0][0] + 1
        ranks.append(first_match_rank)

        if first_match_rank <= 1:
            r1 += 1
        if first_match_rank <= 5:
            r5 += 1
        if first_match_rank <= 10:
            r10 += 1
        mrr += 1.0 / first_match_rank

    return (
        (r1 / num_queries) * 100,
        (r5 / num_queries) * 100,
        (r10 / num_queries) * 100,
        float(np.median(ranks)),
        mrr / num_queries,
    )


def _format_row(label: str, r1, r5, r10, medr, mrr) -> str:
    return (
        f"[{label}]\n"
        f"Recall@1 : {r1:.2f}% | Recall@5 : {r5:.2f}% | "
        f"Recall@10: {r10:.2f}% | Median R : {medr:.1f} | MRR: {mrr:.4f}"
    )


def run_retrieval_eval(
    model: CLIPModel,
    processor: CLIPProcessor,
    val_loader,
    df_val: pd.DataFrame,
    device: torch.device,
    desc: str = "Validation Batches",
) -> Dict[str, Dict[str, float]]:
    """Compute study-level retrieval metrics in both directions.

    Prints results in the original format and returns a nested dict so
    callers can log or persist the numbers.
    """
    model.eval()

    all_image_features = []
    all_text_features = []

    print("Extracting features for the Validation Set for Evaluation...")
    with torch.no_grad():
        for batch in tqdm.tqdm(val_loader, desc=desc):
            images, texts = batch[0], batch[1]
            img_feats, txt_feats = clip_features(
                model, processor, images, texts, device
            )
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)
            all_image_features.append(img_feats.cpu())
            all_text_features.append(txt_feats.cpu())

    all_image_features = torch.cat(all_image_features, dim=0).to(device)
    all_text_features = torch.cat(all_text_features, dim=0).to(device)

    print("\n--- Calculating Retrieval Metrics (Study-Level) ---")
    sim_matrix = torch.matmul(all_image_features, all_text_features.t())

    df_val = df_val.copy()
    df_val["study_id"] = df_val["image"].apply(get_study_id)
    study_ids = df_val["study_id"].values

    i2t = calculate_study_level_metrics(sim_matrix, study_ids)
    t2i = calculate_study_level_metrics(sim_matrix.t(), study_ids)

    print()
    print(_format_row("Image-to-Text Retrieval (Study-Level)", *i2t))
    print()
    print(_format_row("Text-to-Image Retrieval (Study-Level)", *t2i))
    print("==========================================\n")

    keys = ("recall_at_1", "recall_at_5", "recall_at_10", "median_rank", "mrr")
    return {
        "image_to_text": dict(zip(keys, i2t)),
        "text_to_image": dict(zip(keys, t2i)),
    }
