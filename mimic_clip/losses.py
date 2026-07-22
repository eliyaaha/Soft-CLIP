"""Loss functions for hard- and soft-CLIP training."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F


def study_level_contrastive_loss(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    logit_scale: torch.Tensor,
    study_ids: Sequence[str],
) -> torch.Tensor:
    """Supervised contrastive loss treating same study_id as positive."""
    device = image_features.device

    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    scale = logit_scale.exp()
    logits_per_image = scale * torch.matmul(image_features, text_features.t())
    logits_per_text = logits_per_image.t()

    study_ids_np = np.array(study_ids)
    mask_np = study_ids_np[:, None] == study_ids_np[None, :]
    mask = torch.tensor(mask_np, dtype=torch.float32, device=device)

    log_probs_img = F.log_softmax(logits_per_image, dim=1)
    log_probs_txt = F.log_softmax(logits_per_text, dim=1)

    positives_per_row = mask.sum(dim=1)
    loss_img = -(log_probs_img * mask).sum(dim=1) / positives_per_row
    loss_txt = -(log_probs_txt * mask).sum(dim=1) / positives_per_row

    return (loss_img.mean() + loss_txt.mean()) / 2


def _top_k_row_mask(matrix: torch.Tensor, k: int) -> torch.Tensor:
    """Zero out everything except the top-k entries per row.

    ``k`` is clamped to the number of columns. We do not strip the diagonal —
    the row's own similarity is naturally the largest entry and is meant to
    be preserved as the strongest soft target.
    """
    n_cols = matrix.size(1)
    k = max(1, min(k, n_cols))
    topk_vals, topk_idx = torch.topk(matrix, k=k, dim=1)
    mask = torch.zeros_like(matrix)
    mask.scatter_(1, topk_idx, 1.0)
    return matrix.masked_fill(mask == 0, float("-inf"))

def _threshold_soft_targets(
    similarity_matrix: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """Create threshold-based soft targets as defined in the paper."""
    if not -1.0 <= threshold < 1.0:
        raise ValueError(
            "soft_threshold must be in [-1, 1). "
            f"Received {threshold}."
        )

    keep_mask = similarity_matrix > threshold

    # Always preserve the paired sample.
    diagonal_mask = torch.eye(
        similarity_matrix.size(0),
        similarity_matrix.size(1),
        dtype=torch.bool,
        device=similarity_matrix.device,
    )
    keep_mask = keep_mask | diagonal_mask

    targets = torch.where(
        keep_mask,
        (similarity_matrix - threshold) / (1.0 - threshold),
        torch.zeros_like(similarity_matrix),
    )

    targets = targets.clamp_min(0.0)

    row_sums = targets.sum(
        dim=1,
        keepdim=True,
    ).clamp_min(1e-12)

    return targets / row_sums


def _combined_text_image_similarity(
    image_features: torch.Tensor,
    batch_semantic_embeddings: torch.Tensor,
    text_similarity_weight: float,
) -> torch.Tensor:
    """Compute combined cosine similarity for threshold mode only."""
    if not 0.0 <= text_similarity_weight <= 1.0:
        raise ValueError(
            "text_similarity_weight must be between 0 and 1. "
            f"Received {text_similarity_weight}."
        )

    # This normalization is used only by the new threshold mode.
    normalized_semantic_embeddings = F.normalize(
        batch_semantic_embeddings,
        dim=-1,
    )

    text_similarity = torch.matmul(
        normalized_semantic_embeddings,
        normalized_semantic_embeddings.t(),
    )

    # The image features were already normalized earlier in the loss.
    # detach() prevents gradients through target construction.
    detached_image_features = image_features.detach()

    image_similarity = torch.matmul(
        detached_image_features,
        detached_image_features.t(),
    )

    combined_similarity = (
        text_similarity_weight * text_similarity
        + (1.0 - text_similarity_weight) * image_similarity
    )

    return combined_similarity.detach()

def soft_clip_hybrid_loss(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    logit_scale: torch.Tensor,
    batch_semantic_embeddings: torch.Tensor,
    study_ids: Sequence[str],
    alpha: float = 0.5,
    soft_temp: float = 0.1,
    soft_top_k: Optional[int] = None,
    soft_threshold: Optional[float] = None,
    text_similarity_weight: float = 0.5,
) -> torch.Tensor:
    """Hybrid hard (study-level contrastive) + soft (KL to semantic similarities) loss."""
    # 1. Directly reuse the study-level baseline function for the hard loss component
    hard_loss = study_level_contrastive_loss(
        image_features=image_features,
        text_features=text_features,
        logit_scale=logit_scale,
        study_ids=study_ids,
    )

    # 2. Re-compute standard CLIP probabilities for the soft (KL) loss component
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    scale = logit_scale.exp()
    logits_per_image = scale * torch.matmul(image_features, text_features.t())
    logits_per_text = logits_per_image.t()

    log_preds_img = F.log_softmax(logits_per_image, dim=1)
    log_preds_txt = F.log_softmax(logits_per_text, dim=1)

    # # 3. Soft target calculation
    # semantic_sim = torch.matmul(
    #     batch_semantic_embeddings, batch_semantic_embeddings.t()
    # )
    # if soft_top_k is not None:
    #     semantic_sim = _top_k_row_mask(semantic_sim, soft_top_k)

    # soft_targets_dist = F.softmax(semantic_sim / soft_temp, dim=1)

    # soft_loss = (
    #     F.kl_div(log_preds_img, soft_targets_dist, reduction="batchmean")
    #     + F.kl_div(log_preds_txt, soft_targets_dist.t(), reduction="batchmean")
    # ) / 2

    # 3. Soft target calculation

    if soft_threshold is not None:
        combined_similarity = _combined_text_image_similarity(
            image_features=image_features,
            batch_semantic_embeddings=batch_semantic_embeddings,
            text_similarity_weight=text_similarity_weight,
        )

        # threshold -> rescale -> row-normalize.
        soft_targets_dist = _threshold_soft_targets(
            similarity_matrix=combined_similarity,
            threshold=soft_threshold,
        )

    else:
        # Original behavior remains unchanged.
        semantic_sim = torch.matmul(
            batch_semantic_embeddings,
            batch_semantic_embeddings.t(),
        )

        if soft_top_k is not None:
            semantic_sim = _top_k_row_mask(
                semantic_sim,
                soft_top_k,
            )

        soft_targets_dist = F.softmax(
            semantic_sim / soft_temp,
            dim=1,
        )

    soft_targets_dist = F.softmax(
        semantic_sim / soft_temp,
        dim=1,
    )

    soft_loss = (
        F.kl_div(log_preds_img, soft_targets_dist, reduction="batchmean")
        + F.kl_div(log_preds_txt, soft_targets_dist.t(), reduction="batchmean")
    ) / 2

    # 4. Blend
    return (1 - alpha) * hard_loss + alpha * soft_loss