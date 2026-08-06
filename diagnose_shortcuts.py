"""For validation queries where the top-1 retrieval fails a chosen match
definition but shares the query's subject_id, decide whether that's better
explained by:

  (a) "case_similarity"      -- the two reports are genuinely, unusually
                                 alike in content (independent of images),
                                 so confusing them isn't really about the
                                 patient at all.
  (b) "patient_memorization" -- the two report texts are NOT unusually
                                 similar, yet the two images ARE unusually
                                 similar -- nothing in the report content
                                 explains the match, so it's more likely the
                                 model is keying off something in the image
                                 that persistently identifies this patient.
  (c) "ambiguous"             -- neither score clears its cutoff.

Runs both retrieval directions (image_to_text and text_to_image) in one call.
Only "patient_memorization" examples are exported (report texts + both
images) -- those are the ones that actually support a patient-identity
shortcut rather than a genuinely confusable case.

Method
------
Two similarity scores per candidate pair, each independent of the retrieval
decision that flagged the pair in the first place:

  text_sim  -- cosine similarity between the query's and retrieved item's
               report embeddings, using the precomputed BiomedVLP embeddings
               already used to build pseudo-positives (frozen, independent
               of this checkpoint's own training).
  image_sim -- cosine similarity between the query's and retrieved item's
               image embeddings, from THIS checkpoint's own image encoder
               (there is no independent image encoder in this pipeline).

"Unusually similar" is defined relative to a null distribution built from a
random sample of DIFFERENT-subject pairs in the validation set: the cutoff is
the --percentile-th percentile of that null distribution, computed separately
for text_sim and image_sim.

Usage
-----
python diagnose_shortcuts.py --checkpoint "checkpoints/soft/final_beta0.3_p97_text"
python diagnose_shortcuts.py --checkpoint <dir> --fail-on exact --examples-dir shortcut_diagnosis
"""

import argparse
import os
import shutil

import numpy as np
import pandas as pd
import torch.nn.functional as F

from mimic_clip.clip_utils import load_clip
from mimic_clip.config import ExperimentConfig, IMAGE_DIR, embeddings_path
from mimic_clip.data import (
    ORIG_INDEX_COL,
    build_dataloaders,
    get_study_id,
    load_semantic_embeddings,
)
from mimic_clip.metrics import build_text_group_ids, extract_val_features

DIRECTIONS = ("image_to_text", "text_to_image")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a saved checkpoint directory, e.g. "
             "checkpoints/soft/final_beta0.3_p97_text",
    )
    parser.add_argument(
        "--text-field",
        type=str,
        default="text",
        choices=("text", "findings_clean", "impression_clean"),
        help="Must match the text_field the checkpoint was trained/evaluated with.",
    )
    parser.add_argument(
        "--embeddings-tag",
        type=str,
        default="biomedvlp_text_projection",
        help="Precomputed report-embedding tag to use for text_sim (must "
             "already exist as val_<tag>_embeddings.pt).",
    )
    parser.add_argument(
        "--fail-on",
        type=str,
        default="clinical",
        choices=("exact", "clinical"),
        help="Which match definition top-1 must FAIL (Subject must always succeed).",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=90.0,
        help="Percentile of the different-subject null distribution used as "
             "the 'unusually similar' cutoff, for both text_sim and image_sim.",
    )
    parser.add_argument(
        "--null-sample-size",
        type=int,
        default=3000,
        help="Number of random different-subject pairs used to build each "
             "null distribution.",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=10,
        help="Exactly this many patient_memorization examples are exported "
             "per direction (one per subject_id, the first one found).",
    )
    parser.add_argument(
        "--examples-dir",
        type=str,
        default="shortcut_diagnosis",
        help="Directory (relative to cwd) to write <direction>/example_1, ... into.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path prefix to save ALL candidates (both directions, "
             "all labels) as CSV, e.g. 'diagnosis' -> diagnosis_image_to_text.csv, "
             "diagnosis_text_to_image.csv.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_null_cutoff(
    embed: np.ndarray, subject_ids: np.ndarray, sample_size: int, percentile: float, rng: np.random.Generator
) -> float:
    """Percentile of cosine similarity for a random sample of different-subject pairs."""
    n = embed.shape[0]
    sims = []
    tries = 0
    max_tries = sample_size * 20
    while len(sims) < sample_size and tries < max_tries:
        tries += 1
        i, j = rng.integers(0, n, size=2)
        if i == j or subject_ids[i] == subject_ids[j]:
            continue
        sims.append(float(np.dot(embed[i], embed[j])))
    return float(np.percentile(sims, percentile))


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    config = ExperimentConfig(text_field=args.text_field, seed=args.seed).finalize()
    model, processor, device = load_clip(args.checkpoint)
    _, val_loader, df_val = build_dataloaders(config, mode="eval")

    image_features, text_features = extract_val_features(model, processor, val_loader, device)
    image_np = F.normalize(image_features, dim=-1).cpu().numpy()

    df_val = df_val.copy().reset_index(drop=True)
    df_val["study_id"] = df_val["image"].apply(get_study_id)
    if "subject_id" not in df_val.columns:
        raise ValueError("df_val has no subject_id column.")
    if ORIG_INDEX_COL not in df_val.columns:
        raise ValueError(
            f"df_val has no {ORIG_INDEX_COL!r} column -- can't realign it to "
            f"the precomputed embeddings, which were written in the row order "
            f"of the unfiltered validation CSV."
        )

    # Independent (frozen) report embeddings, same source used for pseudo-positives.
    # These were computed on the UNFILTERED validation CSV (text embeddings don't
    # need the image file to exist), so build_dataloaders()'s missing-image filter
    # leaves df_val smaller and out of positional sync with this tensor. _orig_row
    # (added before filtering, in mimic_clip.data._read_processed_csv) records each
    # surviving row's position in that unfiltered CSV, which is exactly the order
    # the embeddings were written in -- so re-index by it to realign.
    report_embed_path = embeddings_path("val", args.embeddings_tag)
    report_embed_full = load_semantic_embeddings(report_embed_path)
    report_np_full = F.normalize(report_embed_full, dim=-1).cpu().numpy()
    orig_rows = df_val[ORIG_INDEX_COL].values
    if orig_rows.max() >= report_np_full.shape[0]:
        raise ValueError(
            f"{report_embed_path} has {report_np_full.shape[0]} rows, but "
            f"df_val references original row {orig_rows.max()} -- the "
            f"embeddings file doesn't match this validation CSV."
        )
    report_np = report_np_full[orig_rows]

    study_ids = df_val["study_id"].values
    subject_ids = df_val["subject_id"].values
    group_ids = build_text_group_ids(df_val["text"])

    print("Building null distributions from random different-subject pairs...")
    text_cutoff = build_null_cutoff(report_np, subject_ids, args.null_sample_size, args.percentile, rng)
    image_cutoff = build_null_cutoff(image_np, subject_ids, args.null_sample_size, args.percentile, rng)
    print(f"text_sim  p{args.percentile:g} cutoff (different-subject pairs): {text_cutoff:.4f}")
    print(f"image_sim p{args.percentile:g} cutoff (different-subject pairs): {image_cutoff:.4f}\n")

    os.makedirs(args.examples_dir, exist_ok=True)
    image_features_np = image_np  # already normalized
    text_features_np = F.normalize(text_features, dim=-1).cpu().numpy()

    for direction in DIRECTIONS:
        if direction == "image_to_text":
            sim_matrix = image_features_np @ text_features_np.T
        else:
            sim_matrix = text_features_np @ image_features_np.T

        rows = []
        n = sim_matrix.shape[0]
        for i in range(n):
            top1 = int(np.argmax(sim_matrix[i]))

            if args.fail_on == "exact":
                definition_failed = study_ids[top1] != study_ids[i]
            else:
                definition_failed = group_ids[top1] != group_ids[i]
            same_subject = subject_ids[top1] == subject_ids[i]
            if not (definition_failed and same_subject):
                continue

            text_sim = float(np.dot(report_np[i], report_np[top1]))
            image_sim = float(np.dot(image_np[i], image_np[top1]))

            if text_sim >= text_cutoff:
                label = "case_similarity"
            elif image_sim >= image_cutoff:
                label = "patient_memorization"
            else:
                label = "ambiguous"

            rows.append(
                {
                    "query_idx": i,
                    "retrieved_idx": top1,
                    "subject_id": subject_ids[i],
                    "query_study_id": study_ids[i],
                    "retrieved_study_id": study_ids[top1],
                    "text_sim": text_sim,
                    "image_sim": image_sim,
                    "label": label,
                    "query_image": df_val.loc[i, "image"],
                    "query_text": df_val.loc[i, "text"],
                    "retrieved_image": df_val.loc[top1, "image"],
                    "retrieved_text": df_val.loc[top1, "text"],
                }
            )

        result_df = pd.DataFrame(rows)
        counts = result_df["label"].value_counts().to_dict() if len(result_df) else {}
        print(f"=== {direction} ===")
        print(f"{len(result_df)} candidates (fails {args.fail_on}, same subject_id).")
        print(f"  case_similarity:      {counts.get('case_similarity', 0)}")
        print(f"  patient_memorization: {counts.get('patient_memorization', 0)}")
        print(f"  ambiguous:            {counts.get('ambiguous', 0)}")

        if args.output:
            result_df.to_csv(f"{args.output}_{direction}.csv", index=False)
            print(f"Saved all {len(result_df)} candidates to {args.output}_{direction}.csv")

        # Only export patient_memorization examples: one per subject, first found.
        mem_df = result_df[result_df["label"] == "patient_memorization"]
        mem_df = mem_df.drop_duplicates(subset="subject_id", keep="first")
        final_df = mem_df.head(args.num_examples).reset_index(drop=True)

        dir_folder = os.path.join(args.examples_dir, direction)
        os.makedirs(dir_folder, exist_ok=True)
        summary_rows = []
        for idx, row in enumerate(final_df.itertuples(index=False), start=1):
            folder = os.path.join(dir_folder, f"example_{idx}")
            os.makedirs(folder, exist_ok=True)

            true_src = os.path.join(IMAGE_DIR, str(row.query_image))
            retrieved_src = os.path.join(IMAGE_DIR, str(row.retrieved_image))
            true_ext = os.path.splitext(true_src)[1] or ".jpg"
            retrieved_ext = os.path.splitext(retrieved_src)[1] or ".jpg"
            true_dst = os.path.join(folder, f"true_image{true_ext}")
            retrieved_dst = os.path.join(folder, f"retrieved_image{retrieved_ext}")
            shutil.copy2(true_src, true_dst)
            shutil.copy2(retrieved_src, retrieved_dst)

            with open(os.path.join(folder, "info.txt"), "w") as f:
                f.write(f"label: patient_memorization\n")
                f.write(f"direction: {direction}\n")
                f.write(f"subject_id: {row.subject_id}\n")
                f.write(f"text_sim: {row.text_sim:.4f}  (cutoff: {text_cutoff:.4f})\n")
                f.write(f"image_sim: {row.image_sim:.4f}  (cutoff: {image_cutoff:.4f})\n\n")
                f.write(f"query study_id: {row.query_study_id}\n")
                f.write(f"true image: {true_dst}\n")
                f.write(f"true report text:\n{row.query_text}\n\n")
                f.write(f"retrieved study_id: {row.retrieved_study_id}\n")
                f.write(f"retrieved image: {retrieved_dst}\n")
                f.write(f"retrieved report text:\n{row.retrieved_text}\n")

            summary_rows.append(
                {
                    "example": f"example_{idx}",
                    "subject_id": row.subject_id,
                    "text_sim": row.text_sim,
                    "image_sim": row.image_sim,
                    "query_study_id": row.query_study_id,
                    "retrieved_study_id": row.retrieved_study_id,
                }
            )
            print(f"  exported {direction}/example_{idx}: subject_id={row.subject_id} "
                  f"text_sim={row.text_sim:.4f} image_sim={row.image_sim:.4f}")

        if summary_rows:
            pd.DataFrame(summary_rows).to_csv(
                os.path.join(dir_folder, "examples_summary.csv"), index=False
            )
        print()


if __name__ == "__main__":
    main()
