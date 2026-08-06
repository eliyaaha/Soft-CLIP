"""Loss functions for hard- and soft-CLIP training."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _assert_row_stochastic(targets: torch.Tensor, name: str) -> None:
    """Guard against the transposed-target bug: KL targets must be row-normalised.

    ``F.kl_div`` does not validate its target, so a matrix whose rows do not sum
    to 1 silently becomes a mis-normalised cross-entropy that reweights anchors
    by an arbitrary per-row factor.
    """
    row_sums = targets.sum(dim=1)
    if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4):
        raise AssertionError(
            f"{name} is not row-stochastic: row sums range "
            f"[{row_sums.min().item():.4f}, {row_sums.max().item():.4f}]"
        )


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


def calibrate_soft_temp(
    similarity_matrix: torch.Tensor,
    target_diagonal_mass: float = 0.5,
    lo: float = 1e-3,
    hi: float = 2.0,
    iters: int = 40,
) -> float:
    """Bisect on T so that ``softmax(sim / T)`` puts a chosen mass on the diagonal.

    Motivation
    ----------
    ``soft_temp`` is far more consequential than ``alpha`` and is easy to set
    badly. With L2-normalised embeddings the diagonal is exactly 1.0, and for a
    similarity distribution with mean ~0.48 / std ~0.19 at batch size 128:

        T = 0.03 -> ~0.66 of the target mass lands on the true pair
        T = 0.05 -> ~0.50
        T = 0.10 -> ~0.21     <- the value used in the original experiments
        T = 0.20 -> ~0.06

    At T = 0.10 the true pair is barely preferred over an arbitrary other
    report, so sweeping ``alpha`` at that temperature really just sweeps "how
    much label corruption do I apply". Calibrating T fixes the amount of
    smoothing to something interpretable, so ``alpha`` means the same thing
    across embedding sources and batch sizes.

    Returns the temperature; monotonicity in T makes plain bisection safe.
    """
    if not 0.0 < target_diagonal_mass < 1.0:
        raise ValueError(
            f"target_diagonal_mass must be in (0, 1). Got {target_diagonal_mass}."
        )

    sim = similarity_matrix.detach().float()
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        mass = F.softmax(sim / mid, dim=1).diagonal().mean().item()
        # Lower T -> peakier -> more diagonal mass.
        if mass < target_diagonal_mass:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2.0


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
    text_similarity_weight: float = 1.0,
    calibrate_temp: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Hybrid hard (study-level contrastive) + soft (KL to semantic similarities) loss.

    Returns
    -------
    (total_loss, components) where ``components`` carries the detached ``hard``
    and ``soft`` terms plus ``diag_mass`` -- the mean probability the soft target
    assigns to the true pair. ``diag_mass`` is the single most useful diagnostic
    here: it says how much smoothing this configuration actually applied, and it
    makes rows of the results table comparable across embedding sources,
    temperatures and batch sizes.

    At alpha=0 this reduces exactly to ``study_level_contrastive_loss``, so an
    alpha=0 run is a strict control that must reproduce the hard baseline.
    """
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

    # 3. Soft target calculation.
    #
    # Each retrieval direction gets its own ROW-NORMALISED target. The previous
    # implementation passed `soft_targets_dist.t()` for the text direction:
    # softmax(dim=1) makes rows sum to 1, but the transpose's rows sum to
    # sum_i exp(S_ij / T) / Z_i, which varies per row. F.kl_div does not check,
    # so that silently became a mis-normalised cross-entropy.
    #
    # The similarity matrix is symmetric, so soft_img == soft_txt here -- but
    # that is the *correct* equality, and building both explicitly keeps the
    # code right if an asymmetric similarity is ever introduced.
    if soft_threshold is not None:
        combined_similarity = _combined_text_image_similarity(
            image_features=image_features,
            batch_semantic_embeddings=batch_semantic_embeddings,
            text_similarity_weight=text_similarity_weight,
        )

        # threshold -> rescale -> row-normalize, independently per direction.
        soft_targets_img = _threshold_soft_targets(
            similarity_matrix=combined_similarity,
            threshold=soft_threshold,
        )
        soft_targets_txt = _threshold_soft_targets(
            similarity_matrix=combined_similarity.t(),
            threshold=soft_threshold,
        )

    else:
        semantic_sim = torch.matmul(
            batch_semantic_embeddings,
            batch_semantic_embeddings.t(),
        )

        # Pick T so the target puts a chosen mass on the true pair. Done before
        # top-k masking so the calibration target refers to the full row.
        effective_temp = soft_temp
        if calibrate_temp is not None:
            effective_temp = calibrate_soft_temp(
                semantic_sim, target_diagonal_mass=calibrate_temp
            )

        sim_img = semantic_sim
        sim_txt = semantic_sim.t()
        if soft_top_k is not None:
            sim_img = _top_k_row_mask(sim_img, soft_top_k)
            sim_txt = _top_k_row_mask(sim_txt, soft_top_k)

        soft_targets_img = F.softmax(sim_img / effective_temp, dim=1)
        soft_targets_txt = F.softmax(sim_txt / effective_temp, dim=1)

    _assert_row_stochastic(soft_targets_img, "soft_targets_img")
    _assert_row_stochastic(soft_targets_txt, "soft_targets_txt")

    soft_loss = (
        F.kl_div(log_preds_img, soft_targets_img, reduction="batchmean")
        + F.kl_div(log_preds_txt, soft_targets_txt, reduction="batchmean")
    ) / 2

    # 4. Blend
    total = (1 - alpha) * hard_loss + alpha * soft_loss

    components = {
        "hard": hard_loss.detach(),
        "soft": soft_loss.detach(),
        # Mean probability the soft target places on the true pair.
        "diag_mass": soft_targets_img.detach().diagonal().mean(),
    }
    return total, components