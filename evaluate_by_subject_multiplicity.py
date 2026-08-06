"""Stratify retrieval evaluation by patient imaging history: subjects who
appear with a single study in the validation set vs. subjects who appear
with more than one.

Motivation
----------
Multi-study subjects are patients who returned for follow-up imaging, which
correlates with abnormal findings; single-study subjects skew toward
normal/templated reports (the population the `clinical` grouping exists to
handle -- see mimic_clip/metrics.py). So raw recall is expected to differ
somewhat mechanically between the two subgroups. The informative read is
whether that gap is large enough that a topline number doesn't generalize to
one subgroup, not just that the gap is nonzero.

Method
------
Loads one checkpoint, extracts image/text features once, builds the full
similarity matrix (same as run_retrieval_eval), then computes a per-query
first-hit rank under each match definition. Subgroup metrics are obtained by
slicing that rank vector by subject -- the candidate pool is always the FULL
validation set for every query, only which queries get scored changes. This
keeps the comparison apples-to-apples: same pool, different query subgroup.

A subject-level (cluster) bootstrap gives a 95% CI on the Recall@1 gap
between subgroups, resampling subjects (not rows) with replacement so that a
patient's multiple studies aren't treated as independent evidence.

Usage
-----
python evaluate_by_subject_multiplicity.py
python evaluate_by_subject_multiplicity.py --checkpoint checkpoints/soft/final_beta0.3_p97_text
python evaluate_by_subject_multiplicity.py --checkpoint <dir> --grouping clinical --n-boot 5000
python evaluate_by_subject_multiplicity.py --checkpoint <dir> --output subject_multiplicity.csv
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch.nn.functional as F

from mimic_clip.clip_utils import load_clip
from mimic_clip.config import ALLOWED_TEXT_FIELDS, ExperimentConfig
from mimic_clip.data import build_dataloaders, get_study_id
from mimic_clip.metrics import build_text_group_ids, extract_val_features

GROUPINGS = ("exact", "clinical", "exact_dedup", "subject")
DIRECTIONS = ("image_to_text", "text_to_image")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/soft/final_beta0.3_p97_text",
        help="Checkpoint directory to evaluate (default: the final combined config).",
    )
    parser.add_argument(
        "--text-field",
        type=str,
        default=None,
        choices=ALLOWED_TEXT_FIELDS,
        help="Override the text_field recorded in the checkpoint's run_config.json.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--grouping",
        type=str,
        default="all",
        choices=GROUPINGS + ("all",),
        help="Which match definition(s) to stratify. 'all' runs all four.",
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=2000,
        help="Subject-level bootstrap resamples for the R@1 gap CI. 0 disables.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional CSV path to save the full stratified summary table.",
    )
    return parser.parse_args()


def resolve_text_field(checkpoint_path: str) -> str:
    """Recover the text_field a checkpoint was trained with.

    Prefers run_config.json (written by trainer.fit); falls back to a
    substring check on the run name. Getting this wrong scores the model on
    a text distribution it was never fine-tuned on.
    """
    cfg_path = os.path.join(checkpoint_path, "run_config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as fh:
            saved = json.load(fh)
        return saved.get("text_field", "text")
    name = os.path.basename(checkpoint_path.rstrip("/"))
    for field in ("findings_clean", "impression_clean"):
        if field in name:
            return field
    return "text"


def per_query_ranks(
    sim_matrix_np: np.ndarray, ids: np.ndarray, candidate_mask: np.ndarray = None
) -> np.ndarray:
    """First-hit rank (1-indexed) of a same-group candidate, per query row.

    NaN where no match was found in the pool. Mirrors the ranking loop in
    mimic_clip.metrics.calculate_study_level_metrics but returns per-query
    ranks instead of an aggregate, so subgroup metrics can be computed by
    slicing this vector afterwards without re-running retrieval per subgroup.
    """
    ids = np.asarray(ids)
    n_queries, n_candidates = sim_matrix_np.shape
    col_idx = np.arange(n_candidates) if candidate_mask is None else np.flatnonzero(candidate_mask)

    ranks = np.full(n_queries, np.nan)
    for i in range(n_queries):
        cols = col_idx if candidate_mask is None or candidate_mask[i] else np.concatenate([col_idx, [i]])
        row = sim_matrix_np[i, cols]
        order = cols[np.argsort(-row)]
        matches = ids[order] == ids[i]
        found = np.flatnonzero(matches)
        if found.size:
            ranks[i] = found[0] + 1
    return ranks


def aggregate(ranks: np.ndarray, mask: np.ndarray = None) -> dict:
    r = ranks if mask is None else ranks[mask]
    r = r[~np.isnan(r)]
    n = r.size
    if n == 0:
        return dict(
            n=0, recall_at_1=float("nan"), recall_at_5=float("nan"),
            recall_at_10=float("nan"), median_rank=float("nan"), mrr=float("nan"),
        )
    return dict(
        n=n,
        recall_at_1=float((r <= 1).mean() * 100),
        recall_at_5=float((r <= 5).mean() * 100),
        recall_at_10=float((r <= 10).mean() * 100),
        median_rank=float(np.median(r)),
        mrr=float(np.mean(1.0 / r)),
    )


def bootstrap_r1_gap(
    ranks: np.ndarray,
    subject_ids: np.ndarray,
    single_mask: np.ndarray,
    multi_mask: np.ndarray,
    n_boot: int,
    seed: int,
) -> tuple:
    """Subject-level (cluster) bootstrap 95% CI on R@1(multi) - R@1(single)."""
    rng = np.random.default_rng(seed)

    def prep(mask):
        idx = np.flatnonzero(mask)
        subs = subject_ids[idx]
        r = ranks[idx]
        uniq = np.unique(subs)
        rows_by_subject = {s: np.flatnonzero(subs == s) for s in uniq}
        return r, uniq, rows_by_subject

    r_single, subs_single, rows_single = prep(single_mask)
    r_multi, subs_multi, rows_multi = prep(multi_mask)

    def one_draw(r, uniq, rows_by_subject):
        sampled = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([rows_by_subject[s] for s in sampled])
        rr = r[rows]
        rr = rr[~np.isnan(rr)]
        return float((rr <= 1).mean() * 100) if rr.size else float("nan")

    diffs = []
    for _ in range(n_boot):
        d = one_draw(r_multi, subs_multi, rows_multi) - one_draw(r_single, subs_single, rows_single)
        if not np.isnan(d):
            diffs.append(d)
    if not diffs:
        return float("nan"), float("nan")
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def main() -> None:
    args = parse_args()
    text_field = args.text_field or resolve_text_field(args.checkpoint)
    print(f"checkpoint : {args.checkpoint}")
    print(f"text_field : {text_field}  (from run_config.json unless overridden)")

    config = ExperimentConfig(
        text_field=text_field, batch_size=args.batch_size, seed=args.seed
    ).finalize()
    model, processor, device = load_clip(args.checkpoint)
    _, val_loader, df_val = build_dataloaders(config, mode="eval")

    image_features, text_features = extract_val_features(model, processor, val_loader, device)
    image_np = F.normalize(image_features, dim=-1).cpu().numpy()
    text_np = F.normalize(text_features, dim=-1).cpu().numpy()

    df_val = df_val.copy().reset_index(drop=True)
    df_val["study_id"] = df_val["image"].apply(get_study_id)
    if "subject_id" not in df_val.columns:
        raise ValueError("df_val has no subject_id column -- can't stratify by patient.")

    study_ids = df_val["study_id"].values
    subject_ids = df_val["subject_id"].values
    group_ids = build_text_group_ids(df_val["text"])
    first_of_group = ~pd.Series(group_ids).duplicated().values

    # --- subgroup definition: how many distinct studies does this subject
    # contribute to THIS validation set? -------------------------------------
    study_count = df_val.groupby("subject_id")["study_id"].transform("nunique").values
    single_mask = study_count == 1
    multi_mask = study_count > 1
    n_subj_single = df_val.loc[single_mask, "subject_id"].nunique()
    n_subj_multi = df_val.loc[multi_mask, "subject_id"].nunique()
    print(f"\nSubjects with 1 study in val : {n_subj_single} ({int(single_mask.sum())} rows)")
    print(f"Subjects with >1 study in val: {n_subj_multi} ({int(multi_mask.sum())} rows)")

    groupings_to_run = GROUPINGS if args.grouping == "all" else (args.grouping,)
    grouping_specs = {
        "exact": (study_ids, None),
        "clinical": (group_ids, None),
        "exact_dedup": (study_ids, first_of_group),
        "subject": (subject_ids, None),
    }

    rows = []
    for direction in DIRECTIONS:
        sim = image_np @ text_np.T if direction == "image_to_text" else text_np @ image_np.T
        for gname in groupings_to_run:
            ids, mask = grouping_specs[gname]
            ranks = per_query_ranks(sim, ids, mask)

            overall = aggregate(ranks)
            single = aggregate(ranks, single_mask)
            multi = aggregate(ranks, multi_mask)

            lo, hi = (float("nan"), float("nan"))
            if args.n_boot > 0:
                lo, hi = bootstrap_r1_gap(
                    ranks, subject_ids, single_mask, multi_mask, args.n_boot, args.seed
                )

            print(f"\n### {direction} / {gname}")
            print(
                f"  overall      n={overall['n']:5d}  R@1={overall['recall_at_1']:.2f}%  "
                f"R@5={overall['recall_at_5']:.2f}%  R@10={overall['recall_at_10']:.2f}%  "
                f"MedR={overall['median_rank']:.1f}  MRR={overall['mrr']:.4f}"
            )
            print(
                f"  single-study n={single['n']:5d}  R@1={single['recall_at_1']:.2f}%  "
                f"R@5={single['recall_at_5']:.2f}%  R@10={single['recall_at_10']:.2f}%  "
                f"MedR={single['median_rank']:.1f}  MRR={single['mrr']:.4f}"
            )
            print(
                f"  multi-study  n={multi['n']:5d}  R@1={multi['recall_at_1']:.2f}%  "
                f"R@5={multi['recall_at_5']:.2f}%  R@10={multi['recall_at_10']:.2f}%  "
                f"MedR={multi['median_rank']:.1f}  MRR={multi['mrr']:.4f}"
            )
            if args.n_boot > 0:
                gap = multi["recall_at_1"] - single["recall_at_1"]
                print(
                    f"  R@1 gap (multi - single): {gap:+.2f}pp  "
                    f"95% CI [{lo:+.2f}, {hi:+.2f}] over {args.n_boot} subject-level bootstraps"
                )

            for label, stats in (("overall", overall), ("single_study", single), ("multi_study", multi)):
                rows.append(dict(direction=direction, grouping=gname, subgroup=label, **stats))

    summary = pd.DataFrame(rows)
    if args.output:
        summary.to_csv(args.output, index=False)
        print(f"\nSaved stratified summary to {args.output}")


if __name__ == "__main__":
    main()
