"""Soft-CLIP training (hybrid hard + KL soft) on MIMIC-CXR.

You pick which precomputed embeddings to use independently from the CLIP
training hyperparameters, so you can compare different embedding models or
text fields without re-embedding.

Examples
--------
    # Default: BiomedVLP embeddings of the `text` column
    python train_soft_clip.py --mode both \\
        --embeddings-tag biomedvlp_text --run-name soft_biomedvlp_text

    # K-neighbor ablation
    python train_soft_clip.py --mode train --embeddings-tag biomedvlp_text \\
        --alpha 0.3 --soft-top-k 5 --run-name soft_a03_k5

    # Compare embedding sources, same loss params
    python train_soft_clip.py --mode train --embeddings-tag biomedvlp_findings_clean
    python train_soft_clip.py --mode train --embeddings-tag bioclinicalbert_impression_clean

    # Explicit paths (override the tag-based lookup)
    python train_soft_clip.py --mode train \\
        --train-embeddings /path/to/train_xxx_embeddings.pt \\
        --val-embeddings   /path/to/val_xxx_embeddings.pt
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
    load_semantic_embeddings,
    run_retrieval_eval,
    soft_clip_hybrid_loss,
)
from mimic_clip.config import ALLOWED_TEXT_FIELDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and/or evaluate a Soft-CLIP model on MIMIC-CXR.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", choices=["train", "eval", "both"], default="both")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint to load. Required for --mode eval.",
    )

    parser.add_argument(
        "--text-field",
        choices=ALLOWED_TEXT_FIELDS,
        default="text",
        help="Text column CLIP itself trains on (independent of embeddings tag).",
    )

    parser.add_argument(
        "--embeddings-tag",
        default="biomedvlp_text",
        help="Selects {BASE_DATA_DIR}/train_{tag}_embeddings.pt and val_{tag}_embeddings.pt. "
             "Match the {model}_{field} convention from create_embeddings.py.",
    )
    parser.add_argument(
        "--train-embeddings",
        default=None,
        help="Override --embeddings-tag with an explicit path.",
    )
    parser.add_argument(
        "--val-embeddings",
        default=None,
        help="Override --embeddings-tag with an explicit path.",
    )

    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Weight on the soft (KL) loss term; (1-alpha) on the hard term.")
    parser.add_argument("--soft-temp", type=float, default=0.1,
                        help="Temperature for the softmax over semantic similarities.")
    parser.add_argument("--soft-top-k", type=int, default=None,
                        help="Keep only the top-K neighbors per row in the soft targets "
                             "(None = full row).")

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    config = ExperimentConfig(
        loss_type="soft",
        text_field=args.text_field,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        num_workers=args.num_workers,
        alpha=args.alpha,
        soft_temp=args.soft_temp,
        soft_top_k=args.soft_top_k,
        embeddings_tag=args.embeddings_tag,
        train_embeddings_path=args.train_embeddings,
        val_embeddings_path=args.val_embeddings,
        run_name=args.run_name,
    )
    return config.finalize()


def make_soft_loss_fn(
    train_embeddings: torch.Tensor,
    val_embeddings: torch.Tensor,
    config: ExperimentConfig,
):
    """Build a closure that picks the right embedding tensor by model.training."""

    def loss_fn(model, processor, batch, device):
        images, texts, study_ids, indices = batch
        semantic_source = train_embeddings if model.training else val_embeddings
        batch_semantic = semantic_source[indices].to(device)

        image_features, text_features = clip_features(
            model, processor, images, texts, device
        )
        return soft_clip_hybrid_loss(
            image_features=image_features,
            text_features=text_features,
            logit_scale=model.logit_scale,
            batch_semantic_embeddings=batch_semantic,
            study_ids=study_ids,
            alpha=config.alpha,
            soft_temp=config.soft_temp,
            soft_top_k=config.soft_top_k,
        )
    return loss_fn


def main() -> None:
    args = parse_args()
    config = build_config(args)
    print(f"Running in mode: {args.mode}")
    print(config.summary())

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
        train_embeddings = load_semantic_embeddings(config.train_embeddings_path)
        val_embeddings = load_semantic_embeddings(config.val_embeddings_path)
        print(f"Loaded train embeddings: {tuple(train_embeddings.shape)}")
        print(f"Loaded val embeddings  : {tuple(val_embeddings.shape)}")

        fit(
            model=model,
            processor=processor,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=make_soft_loss_fn(train_embeddings, val_embeddings, config),
            config=config,
        )

    if args.mode in ("eval", "both"):
        if args.mode == "both":
            model, processor, device = load_clip(config.checkpoint_dir)
        run_retrieval_eval(model, processor, val_loader, df_val, device)


if __name__ == "__main__":
    main()
