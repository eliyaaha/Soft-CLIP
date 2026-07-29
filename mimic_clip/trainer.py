"""Shared training loop with early stopping and checkpointing."""

from __future__ import annotations

import json
import math
import os
import random
import time
from typing import Callable, Dict, Tuple

import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

from .config import ExperimentConfig, MAX_LOGIT_SCALE


# loss_fn(model, processor, batch, device) -> (loss, {"hard": ..., ...})
BatchLossFn = Callable[
    [CLIPModel, CLIPProcessor, "tuple", torch.device],
    Tuple[torch.Tensor, Dict[str, torch.Tensor]],
]


def set_seed(seed: int) -> None:
    """Seed every RNG that affects training.

    Without this there is no way to tell whether a gap like 5.70 vs 5.10 R@1 is
    a real effect or run-to-run noise. Run the baseline at >= 3 seeds and report
    mean +/- std before reading anything into small differences.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_run_config(config: ExperimentConfig) -> None:
    """Persist the resolved config next to the weights.

    Evaluation should read the training parameters from here rather than having
    a human restate them -- restating ``text_field`` by hand is exactly how
    checkpoints trained on ``findings_clean`` ended up being evaluated on
    ``text``.
    """
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    path = os.path.join(config.checkpoint_dir, "run_config.json")
    with open(path, "w") as fh:
        json.dump(config.to_dict(), fh, indent=2, default=str)


def _run_epoch(
    model, processor, loader, loss_fn, device, optimizer=None
) -> Dict[str, float]:
    """One pass over ``loader``. Trains when ``optimizer`` is given."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    totals: Dict[str, float] = {}
    n_batches = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch_idx, batch in enumerate(loader):
            if is_train:
                optimizer.zero_grad()

            loss, components = loss_fn(model, processor, batch, device)

            if is_train:
                loss.backward()
                optimizer.step()
                # Cap the learned temperature exactly as OpenAI CLIP does.
                # Without this, the KL term can satisfy a near-uniform soft
                # target simply by shrinking logit_scale, which flattens every
                # similarity in the space -- including at evaluation time.
                with torch.no_grad():
                    model.logit_scale.clamp_(0.0, math.log(MAX_LOGIT_SCALE))

            totals["total"] = totals.get("total", 0.0) + loss.item()
            for key, value in components.items():
                totals[key] = totals.get(key, 0.0) + float(value)
            n_batches += 1

            if is_train and batch_idx % 50 == 0:
                extra = "  ".join(
                    f"{k}: {float(v):.4f}" for k, v in components.items()
                )
                print(
                    f"  Step [{batch_idx}/{len(loader)}] | "
                    f"Loss: {loss.item():.4f} | {extra}"
                )

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


def fit(
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
    train_loader,
    val_loader,
    loss_fn: BatchLossFn,
    config: ExperimentConfig,
) -> None:
    """Standard train / val loop with early stopping.

    ``loss_fn`` returns ``(loss, components)``; ``components`` must contain a
    ``"hard"`` entry, which is defined identically for both arms and is what
    early stopping monitors by default. Monitoring the blended loss instead
    (``early_stop_metric="total"``) selects checkpoints against a different
    objective for every alpha, so those checkpoints are not comparable to each
    other or to the baseline.
    """
    set_seed(config.seed)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    print("Beginning execution of fine-tuning loop...")
    print(config.summary())

    monitor = "hard" if config.early_stop_metric == "hard" else "total"
    best_val = float("inf")
    epochs_without_improvement = 0

    for epoch in range(config.epochs):
        start_time = time.time()
        print(f"\nEpoch [{epoch + 1}/{config.epochs}]")

        train_stats = _run_epoch(
            model, processor, train_loader, loss_fn, device, optimizer
        )
        val_stats = _run_epoch(model, processor, val_loader, loss_fn, device)

        elapsed = time.time() - start_time
        current = val_stats[monitor]

        print("\n=======================================================")
        print(f"Epoch {epoch + 1} Metrics:")
        for key in sorted(train_stats):
            print(
                f"-> {key:<10} train: {train_stats[key]:.4f} | "
                f"val: {val_stats.get(key, float('nan')):.4f}"
            )
        print(f"-> logit_scale       : {model.logit_scale.exp().item():.2f}")
        print(f"-> early stopping on : val/{monitor}")
        print(f"-> Time Taken        : {elapsed:.2f}s")
        print("=======================================================\n")

        if current < best_val:
            best_val = current
            epochs_without_improvement = 0

            os.makedirs(config.checkpoint_dir, exist_ok=True)
            model.save_pretrained(config.checkpoint_dir)
            processor.save_pretrained(config.checkpoint_dir)
            save_run_config(config)
            print(
                f"New best val/{monitor} = {current:.4f}. "
                f"Checkpoint stored at {config.checkpoint_dir}\n"
            )
        else:
            epochs_without_improvement += 1
            print(
                f"val/{monitor} did not improve. "
                f"Early stopping counter: {epochs_without_improvement}/{config.patience}\n"
            )
            if epochs_without_improvement >= config.patience:
                print(
                    f"Early stopping condition triggered. "
                    f"Terminating training at Epoch {epoch + 1}."
                )
                break
