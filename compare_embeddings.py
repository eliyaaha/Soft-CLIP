"""Intrinsically score embedding variants against lexical clinical equivalence.

Why this exists
---------------
Choosing an embedding source by training a Soft-CLIP model per variant is slow
and confounded: each training run adds optimisation noise, and the retrieval
metric it produces is several causal steps away from the thing being compared.

The soft targets only ever use ONE property of an embedding: does it assign high
similarity to reports that mean the same thing? That property can be measured
directly, with no training at all, using a ground truth that already exists in
the data -- reports whose text normalises to the same string are clinically
equivalent by definition.

This gives a controlled comparison in the strict sense: identical reports,
identical ground truth, identical metric, no optimisation in the loop. The only
thing that varies is the embedding file.

Metrics
-------
pair_auroc   Probability a same-meaning pair scores above a different-meaning
             pair. The headline number. 0.5 = no signal.
p_at_1       Fraction of anchors whose nearest neighbour is same-meaning.
separation   mean(same) - mean(different), in cosine units.
bimodality   Higher = the two populations are more distinctly separated
             (a standardised effect size, Cohen's d).

Usage
-----
    # Controlled: same pooling for every model, only the model varies
    python compare_embeddings.py --tags \
        biomedvlp_text_mean bioclinicalbert_text_mean gemma_embed_text_native

    # Native: each model's intended representation (answers a different question)
    python compare_embeddings.py --tags \
        biomedvlp_text_projection bioclinicalbert_text_mean gemma_embed_text_native

    # Full grid, written to CSV
    python compare_embeddings.py --tags $(ls $BASE_DATA_DIR | grep '^val_' | \
        sed 's/^val_//;s/_embeddings.pt$//') --out embedding_comparison.csv
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch

from mimic_clip.config import BASE_DATA_DIR, VAL_CSV_PATH, TRAIN_CSV_PATH
from mimic_clip.metrics import build_text_group_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tags", nargs="+", required=True,
                        help="Embedding tags to compare, i.e. {split}_{tag}_embeddings.pt")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--sample-size", type=int, default=5000,
                        help="Rows to subsample. The SAME rows are used for every "
                             "tag, so the comparison is paired.")
    parser.add_argument("--group-field", default="text",
                        help="Column whose near-duplicates define clinical equivalence. "
                             "Held fixed across all tags -- do not vary this per tag.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None, help="Optional CSV path for the results table.")
    return parser.parse_args()


def score_embedding(emb: torch.Tensor, groups: np.ndarray) -> dict:
    """Score one embedding matrix against clinical-equivalence groups."""
    emb = torch.nn.functional.normalize(emb.float(), dim=-1)
    sim = (emb @ emb.t()).numpy()
    n = sim.shape[0]

    off = ~np.eye(n, dtype=bool)
    same = (groups[:, None] == groups[None, :]) & off
    diff = (groups[:, None] != groups[None, :]) & off

    same_vals = sim[same]
    diff_vals = sim[diff]
    if same_vals.size == 0:
        raise ValueError(
            "No same-meaning pairs in the sample -- increase --sample-size. "
            "Duplicate reports need to co-occur in the subsample to be usable."
        )

    # AUROC via rank statistic (Mann-Whitney U), exact and memory-light.
    allv = np.concatenate([same_vals, diff_vals])
    ranks = allv.argsort().argsort().astype(np.float64) + 1
    r_same = ranks[: same_vals.size].sum()
    n1, n2 = same_vals.size, diff_vals.size
    auroc = (r_same - n1 * (n1 + 1) / 2) / (n1 * n2)

    # Precision@1 over anchors that actually have a same-meaning partner.
    has_partner = same.any(axis=1)
    sim_masked = sim.copy()
    np.fill_diagonal(sim_masked, -np.inf)
    nn = sim_masked.argmax(axis=1)
    p_at_1 = same[np.arange(n), nn][has_partner].mean()

    pooled_sd = np.sqrt((same_vals.var() + diff_vals.var()) / 2)
    return {
        "pair_auroc": float(auroc),
        "p_at_1": float(p_at_1),
        "separation": float(same_vals.mean() - diff_vals.mean()),
        "bimodality": float((same_vals.mean() - diff_vals.mean()) / max(pooled_sd, 1e-9)),
        "mean_same": float(same_vals.mean()),
        "mean_diff": float(diff_vals.mean()),
        "std_diff": float(diff_vals.std()),
        "dim": int(emb.shape[1]),
    }


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    csv_path = VAL_CSV_PATH if args.split == "val" else TRAIN_CSV_PATH
    df = pd.read_csv(csv_path).fillna("")
    print(f"Loaded {len(df):,} rows from {csv_path}")

    # One fixed subsample, shared by every tag -> paired comparison.
    if len(df) > args.sample_size:
        idx = np.sort(rng.choice(len(df), args.sample_size, replace=False))
    else:
        idx = np.arange(len(df))

    groups = build_text_group_ids(df[args.group_field])[idx]
    sizes = pd.Series(groups).value_counts()
    n_pairs = int((sizes * (sizes - 1) / 2).sum())
    print(
        f"Ground truth from {args.group_field!r}: {len(np.unique(groups)):,} groups "
        f"over {len(idx):,} rows | {int((sizes > 1).sum()):,} groups have >1 member "
        f"| {n_pairs:,} same-meaning pairs"
    )
    if n_pairs < 100:
        print("WARNING: very few same-meaning pairs; results will be noisy.")

    rows = []
    for tag in args.tags:
        path = os.path.join(BASE_DATA_DIR, f"{args.split}_{tag}_embeddings.pt")
        if not os.path.exists(path):
            print(f"  [skip] {tag}: not found at {path}")
            continue
        emb = torch.load(path, map_location="cpu", weights_only=True)
        if emb.shape[0] != len(df):
            print(
                f"  [skip] {tag}: has {emb.shape[0]:,} rows but the CSV has "
                f"{len(df):,}. Regenerate it against the current processed CSV."
            )
            continue
        rows.append({"tag": tag, **score_embedding(emb[idx], groups)})
        print(f"  [ok]   {tag}")

    if not rows:
        raise SystemExit("No embeddings could be scored.")

    out = pd.DataFrame(rows).sort_values("pair_auroc", ascending=False)
    pd.set_option("display.width", 200)
    print("\n" + "=" * 90)
    print("Embedding quality vs. lexical clinical equivalence (higher is better)")
    print("=" * 90)
    print(out[["tag", "pair_auroc", "p_at_1", "separation", "bimodality",
               "mean_same", "mean_diff"]].to_string(index=False, float_format="%.4f"))

    best = out.iloc[0]
    print(
        f"\nBest: {best['tag']}  (AUROC {best['pair_auroc']:.4f})\n"
        f"AUROC near 0.5 means the embedding cannot tell same-meaning reports from "
        f"different ones,\nin which case the soft targets built from it are noise "
        f"and no alpha will help."
    )
    print(
        "\nReminder: only compare rows that differ in ONE factor. Tags encode "
        "{model}_{field}_{pooling}[_sent],\nso hold pooling fixed when comparing "
        "models, and hold the model fixed when comparing pooling."
    )

    if args.out:
        out.to_csv(args.out, index=False)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
