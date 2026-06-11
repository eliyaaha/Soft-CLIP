"""Hard-CLIP study-level contrastive fine-tuning on MIMIC-CXR.

Examples
--------
    python train_baseline.py --mode both
    python train_baseline.py --mode train --text-field findings_clean --run-name hard_find
    python train_baseline.py --mode eval --checkpoint checkpoints/hard/hard_imp
"""

import argparse
import os

import torch

from mimic_clip import (
    ExperimentConfig,
    build_dataloaders,
    clip_features,
    fit,
    load_clip,
    run_retrieval_eval,
    study_level_contrastive_loss,
)
from mimic_clip.config import ALLOWED_TEXT_FIELDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune and/or evaluate a hard CLIP baseline on MIMIC-CXR.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["train", "eval", "both"],
        default="both",
        help="Run only training, only evaluation, or both.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint dir to load. Required for --mode eval; "
             "in --mode both we eval the best checkpoint produced by training.",
    )
    parser.add_argument(
        "--text-field",
        choices=ALLOWED_TEXT_FIELDS,
        default="text",
        help="Which text column from the processed CSV CLIP should see.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--run-name", default=None, help="Override auto-generated run name.")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    config = ExperimentConfig(
        loss_type="hard",
        text_field=args.text_field,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        num_workers=args.num_workers,
        run_name=args.run_name,
    )
    return config.finalize()


def make_hard_loss_fn():
    def loss_fn(model, processor, batch, device):
        images, texts, study_ids, _ = batch
        image_features, text_features = clip_features(
            model, processor, images, texts, device
        )
        return study_level_contrastive_loss(
            image_features, text_features, model.logit_scale, study_ids
        )
    return loss_fn


def main() -> None:
    args = parse_args()
    config = build_config(args)
    print(f"Running in mode: {args.mode}")

    train_loader, val_loader, df_val = build_dataloaders(config, args.mode)

    if args.mode == "eval":
        if not args.checkpoint or not os.path.exists(args.checkpoint):
            raise FileNotFoundError(
                "Provide --checkpoint <path> when running --mode eval."
            )
        model, processor, device = load_clip(args.checkpoint)
    else:
        model, processor, device = load_clip()

    if args.mode in ("train", "both"):
        fit(
            model=model,
            processor=processor,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=make_hard_loss_fn(),
            config=config,
        )

    if args.mode in ("eval", "both"):
        if args.mode == "both":
            model, processor, device = load_clip(config.checkpoint_dir)
        run_retrieval_eval(model, processor, val_loader, df_val, device)


if __name__ == "__main__":
    main()
