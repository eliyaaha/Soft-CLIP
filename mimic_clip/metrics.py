"""Single shared retrieval evaluation used by every training script.

Three complementary views of the same similarity matrix:

1. ``exact``      -- match on ``study_id``. The original metric. It asks the
                     model to rank the one true report above 2.8k others.
2. ``clinical``   -- match on near-duplicate report text. Two studies whose
                     reports normalise to the same string are clinically
                     equivalent *by definition*, so retrieving either one is a
                     correct answer. Requires no labels and no model.
3. ``subject``    -- match on ``subject_id``. A nuisance probe: how much
                     patient-identity information survives in the encoder.

The motivation for (2) and (3): MIMIC-CXR contains thousands of studies whose
reports are verbatim identical ("No acute cardiopulmonary abnormality"). Under
``exact``, retrieving one of those for another is scored as an error even though
it is clinically correct. The only way to win at ``exact`` on such pairs is to
encode nuisance variation -- anatomy, positioning, exposure -- which is exactly
what soft supervision is designed to suppress. So ``exact`` alone cannot
distinguish "the soft targets are noise" from "the soft targets worked".

The expected signature of a method that works: ``exact`` drops, ``clinical``
rises, ``subject`` drops.
"""

from __future__ import annotations

import re
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import tqdm
from transformers import CLIPModel, CLIPProcessor

from .clip_utils import clip_features
from .data import get_study_id

_NON_ALPHA = re.compile(r"[^a-z ]+")
_WS = re.compile(r"\s+")


def normalize_report(text: str) -> str:
    """Aggressively normalise report text for duplicate detection.

    Lowercase, strip everything but letters and spaces, collapse whitespace.
    Deliberately crude: the goal is to collapse formatting and punctuation
    variation between otherwise identical templated reports.
    """
    lowered = str(text).lower()
    lowered = _NON_ALPHA.sub(" ", lowered)
    return _WS.sub(" ", lowered).strip()


def build_text_group_ids(texts: pd.Series) -> np.ndarray:
    """Map each report to a clinical-equivalence group id via exact text match.

    This is lexical ground truth, not a model output, so using it to score the
    models under comparison is not circular.
    """
    normalized = texts.apply(normalize_report)
    codes, _ = pd.factorize(normalized)
    return codes.astype(np.int64)


def calculate_study_level_metrics(
    sim_matrix: torch.Tensor,
    study_ids: np.ndarray,
    candidate_mask: Optional[np.ndarray] = None,
) -> Tuple[float, float, float, float, float]:
    """Return (Recall@1, Recall@5, Recall@10, MedianRank, MRR).

    ``study_ids`` is any grouping -- study id, text-equivalence group, or
    subject id. A retrieval counts as correct when the retrieved item shares the
    query's group.

    ``candidate_mask`` optionally restricts which columns may be retrieved, used
    for the de-duplicated candidate pool. Queries are unaffected; only the pool
    shrinks.
    """
    sim_matrix_np = sim_matrix.detach().cpu().numpy()
    study_ids = np.asarray(study_ids)

    if candidate_mask is None:
        col_idx = np.arange(sim_matrix_np.shape[1])
    else:
        col_idx = np.flatnonzero(np.asarray(candidate_mask))

    r1 = r5 = r10 = mrr = 0.0
    ranks = []
    num_queries = sim_matrix_np.shape[0]
    scored = 0

    for i in range(num_queries):
        # Always allow the query's own column, even if the pool excludes it.
        cols = col_idx if candidate_mask is None or candidate_mask[i] else \
            np.concatenate([col_idx, [i]])

        row = sim_matrix_np[i, cols]
        order = cols[np.argsort(-row)]
        matches = study_ids[order] == study_ids[i]

        found = np.flatnonzero(matches)
        if found.size == 0:
            continue

        first_match_rank = int(found[0]) + 1
        ranks.append(first_match_rank)
        scored += 1

        if first_match_rank <= 1:
            r1 += 1
        if first_match_rank <= 5:
            r5 += 1
        if first_match_rank <= 10:
            r10 += 1
        mrr += 1.0 / first_match_rank

    if scored == 0:
        return (0.0, 0.0, 0.0, float("nan"), 0.0)

    return (
        (r1 / scored) * 100,
        (r5 / scored) * 100,
        (r10 / scored) * 100,
        float(np.median(ranks)),
        mrr / scored,
    )


def _format_row(label: str, r1, r5, r10, medr, mrr) -> str:
    return (
        f"[{label}]\n"
        f"Recall@1 : {r1:.2f}% | Recall@5 : {r5:.2f}% | "
        f"Recall@10: {r10:.2f}% | Median R : {medr:.1f} | MRR: {mrr:.4f}"
    )


_KEYS = ("recall_at_1", "recall_at_5", "recall_at_10", "median_rank", "mrr")


def extract_val_features(
    model: CLIPModel,
    processor: CLIPProcessor,
    val_loader,
    device: torch.device,
    desc: str = "Validation Batches",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """L2-normalised image and text features for the whole validation loader.

    Returned so downstream analyses (linear probes, clustering) can reuse them
    without a second forward pass.
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

    return (
        torch.cat(all_image_features, dim=0),
        torch.cat(all_text_features, dim=0),
    )


def run_retrieval_eval(
    model: CLIPModel,
    processor: CLIPProcessor,
    val_loader,
    df_val: pd.DataFrame,
    device: torch.device,
    desc: str = "Validation Batches",
    return_features: bool = False,
) -> Dict[str, Dict[str, float]]:
    """Compute retrieval metrics in both directions under three groupings."""
    image_features, text_features = extract_val_features(
        model, processor, val_loader, device, desc
    )
    image_features = image_features.to(device)
    text_features = text_features.to(device)

    sim_matrix = torch.matmul(image_features, text_features.t())

    df_val = df_val.copy().reset_index(drop=True)
    if len(df_val) != sim_matrix.size(0):
        raise ValueError(
            f"df_val has {len(df_val)} rows but the loader produced "
            f"{sim_matrix.size(0)} samples. These must correspond positionally; "
            f"pass the filtered frame returned by build_dataloaders()."
        )

    df_val["study_id"] = df_val["image"].apply(get_study_id)
    study_ids = df_val["study_id"].values

    # --- grouping 2: clinical equivalence via near-duplicate report text -----
    group_ids = build_text_group_ids(df_val["text"])
    n_groups = len(np.unique(group_ids))
    dup_rows = len(group_ids) - n_groups

    # --- de-duplicated candidate pool ---------------------------------------
    # Keep one representative per text group, so no two candidates are
    # clinically identical and `exact` retrieval stops contradicting itself.
    first_of_group = ~pd.Series(group_ids).duplicated().values

    results: Dict[str, Dict[str, float]] = {}

    print("\n--- Retrieval Metrics ---")
    print(
        f"Candidates: {len(df_val):,} images | "
        f"{len(np.unique(study_ids)):,} studies | "
        f"{n_groups:,} distinct report texts ({dup_rows:,} duplicate rows)"
    )

    groupings = [
        ("exact", study_ids, None, "study_id"),
        ("clinical", group_ids, None, "near-duplicate report text"),
        ("exact_dedup", study_ids, first_of_group, "study_id, deduplicated pool"),
    ]
    if "subject_id" in df_val.columns:
        groupings.append(
            ("subject", df_val["subject_id"].values, None, "subject_id (nuisance probe)")
        )

    for name, ids, mask, description in groupings:
        i2t = calculate_study_level_metrics(sim_matrix, ids, mask)
        t2i = calculate_study_level_metrics(sim_matrix.t(), ids, mask)

        print(f"\n### {name}  (match on {description})")
        print(_format_row("Image-to-Text", *i2t))
        print(_format_row("Text-to-Image", *t2i))

        results[f"{name}_image_to_text"] = dict(zip(_KEYS, i2t))
        results[f"{name}_text_to_image"] = dict(zip(_KEYS, t2i))

    print("==========================================\n")

    # Backwards-compatible aliases for existing callers/notebooks.
    results["image_to_text"] = results["exact_image_to_text"]
    results["text_to_image"] = results["exact_text_to_image"]

    if return_features:
        results["_features"] = {
            "image": image_features.cpu(),
            "text": text_features.cpu(),
        }
    return results
