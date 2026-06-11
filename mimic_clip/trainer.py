"""Shared training loop with early stopping and checkpointing."""

from __future__ import annotations

import os
import time
from typing import Callable

import torch
from transformers import CLIPModel, CLIPProcessor

from .config import ExperimentConfig


BatchLossFn = Callable[
    [CLIPModel, CLIPProcessor, "tuple", torch.device], torch.Tensor
]


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

    The caller passes a ``loss_fn(model, processor, batch, device)`` closure
    that knows how to compute either the hard or the soft loss from a batch.
    """
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    print("Beginning execution of fine-tuning loop...")
    print(config.summary())

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        start_time = time.time()

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            loss = loss_fn(model, processor, batch, device)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            if batch_idx % 50 == 0:
                print(
                    f"Epoch [{epoch + 1}/{config.epochs}] | "
                    f"Train Step [{batch_idx}/{len(train_loader)}] | "
                    f"Loss: {loss.item():.4f}"
                )

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                loss = loss_fn(model, processor, batch, device)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        elapsed = time.time() - start_time

        print("\n=======================================================")
        print(f"Epoch {epoch + 1} Metrics:")
        print(f"-> Average Train Loss: {avg_train_loss:.4f}")
        print(f"-> Average Val Loss  : {avg_val_loss:.4f}")
        print(f"-> Time Taken        : {elapsed:.2f}s")
        print("=======================================================\n")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0

            os.makedirs(config.checkpoint_dir, exist_ok=True)
            model.save_pretrained(config.checkpoint_dir)
            processor.save_pretrained(config.checkpoint_dir)
            print(
                f"New best validation loss achieved. "
                f"Model checkpoint successfully stored at {config.checkpoint_dir}\n"
            )
        else:
            epochs_without_improvement += 1
            print(
                f"Validation loss did not improve. "
                f"Early stopping counter: {epochs_without_improvement}/{config.patience}\n"
            )
            if epochs_without_improvement >= config.patience:
                print(
                    f"Early stopping condition triggered. "
                    f"Terminating training at Epoch {epoch + 1}."
                )
                break
