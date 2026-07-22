"""Analyze the cosine-similarity distribution of BiomedVLP (full-text) report
embeddings, to help choose a sensible `soft_threshold` value for the
threshold-based Soft-CLIP variant.

Usage
-----
python analyze_similarity_threshold.py --split train
python analyze_similarity_threshold.py --split val --sample-size 2000

Assumes the standard naming convention from create_embeddings.py:
    {split}_biomedvlp_text_embeddings.pt
located under BASE_DATA_DIR (imported from preprocess.py, same as the rest
of the pipeline).
"""

import argparse
import os

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from preprocess import BASE_DATA_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        choices=("train", "val"),
        default="train",
        help="Which split's embeddings to analyze.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5000,
        help=(
            "Randomly subsample this many reports before computing the full "
            "pairwise similarity matrix (avoids O(N^2) memory blowup on the "
            "full training set)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for subsampling.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="similarity_histogram.png",
        help="Path to save the histogram figure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    embeddings_path = os.path.join(
        BASE_DATA_DIR, f"{args.split}_biomedvlp_text_embeddings.pt"
    )
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(
            f"Could not find {embeddings_path}. "
            f"Run create_embeddings.py --model biomedvlp --field text first."
        )

    embeddings = torch.load(embeddings_path)
    print(f"Loaded embeddings: {embeddings.shape} from {embeddings_path}")

    n = embeddings.size(0)
    if n > args.sample_size:
        idx = torch.randperm(n)[: args.sample_size]
        embeddings = embeddings[idx]
        print(f"Subsampled to {embeddings.size(0)} reports for similarity computation.")

    # Embeddings from create_embeddings.py are already L2-normalized for the
    # BERT-family branch, but we normalize again defensively.
    embeddings = F.normalize(embeddings, dim=-1)

    similarity_matrix = torch.matmul(embeddings, embeddings.t())

    # Exclude the diagonal (self-similarity, always 1.0) — we care about the
    # distribution of similarity to *other* reports, since that's what
    # soft_threshold filters over.
    n_rows = similarity_matrix.size(0)
    off_diag_mask = ~torch.eye(n_rows, dtype=torch.bool)
    off_diag_values = similarity_matrix[off_diag_mask]

    print(f"\n--- Off-diagonal cosine similarity stats ({args.split}, BiomedVLP, full text) ---")
    print(f"count : {off_diag_values.numel():,}")
    print(f"mean  : {off_diag_values.mean().item():.4f}")
    print(f"std   : {off_diag_values.std().item():.4f}")
    print(f"min   : {off_diag_values.min().item():.4f}")
    print(f"max   : {off_diag_values.max().item():.4f}")

    percentiles = [50, 75, 90, 95, 97, 99, 99.5]
    print("\nPercentiles (candidate threshold values):")
    for p in percentiles:
        val = torch.quantile(off_diag_values, p / 100.0).item()
        frac_kept = (off_diag_values > val).float().mean().item()
        print(
            f"  p{p:<5}: similarity = {val:.4f}  "
            f"-> keeps ~{frac_kept*100:.2f}% of off-diagonal pairs per row"
        )

    # Plot histogram
    plt.figure(figsize=(8, 5))
    plt.hist(off_diag_values.numpy(), bins=100, color="#4C72B0", alpha=0.85)
    for p in [90, 95, 99]:
        val = torch.quantile(off_diag_values, p / 100.0).item()
        plt.axvline(val, linestyle="--", linewidth=1, label=f"p{p} = {val:.2f}")
    plt.xlabel("Cosine similarity (off-diagonal report pairs)")
    plt.ylabel("Count")
    plt.title(
        f"BiomedVLP full-text report similarity distribution ({args.split} split)"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"\nSaved histogram to {args.output}")


if __name__ == "__main__":
    main()
