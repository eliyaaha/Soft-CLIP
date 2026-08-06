"""Find validation queries where the retrieved top-1 item FAILS one match
definition but SUCCEEDS another (Subject succeeding is always required; which
definition must fail is controlled by --fail-on), for use as concrete report
examples.

--fail-on exact (default): top-1 is the wrong study_id but the same
    subject_id -- Exact Match fails at R@1, Subject succeeds at R@1. This
    includes cases where the retrieved report is a near-duplicate of the true
    report (i.e. would count as correct under Clinical too), so some of these
    aren't real content mistakes, just a different study_id.
--fail-on clinical: top-1 is not even a near-duplicate report (different
    text-equivalence group, per the same normalize_report/build_text_group_ids
    logic used for the Clinical match definition) but is the same subject_id
    -- Clinical fails at R@1, Subject succeeds at R@1. This excludes the
    near-duplicate case above, leaving only examples where the retrieved
    report is genuinely different content from the true report.

This does not compute aggregate metrics (see mimic_clip/metrics.py for that);
it dumps individual query/retrieved pairs behind the chosen failure mode,
keeps at most one example per subject_id (the first one found), and copies
the true and retrieved images for the top --num-examples of those into
<examples-dir>/example_1, example_2, ... so they're ready to drop into the
report.

Usage
-----
python find_shortcut_examples.py --checkpoint "checkpoints/soft/final_beta0.3_p97_text"
python find_shortcut_examples.py --checkpoint <dir> --direction text_to_image --num-examples 10
python find_shortcut_examples.py --checkpoint <dir> --fail-on clinical --examples-dir examples_clinical
"""

import argparse
import os
import shutil

import numpy as np
import pandas as pd
import torch

from mimic_clip.clip_utils import load_clip
from mimic_clip.config import ExperimentConfig, IMAGE_DIR
from mimic_clip.data import build_dataloaders, get_study_id
from mimic_clip.metrics import build_text_group_ids, extract_val_features


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
        "--direction",
        type=str,
        default="image_to_text",
        choices=("image_to_text", "text_to_image"),
        help="Which retrieval direction to inspect.",
    )
    parser.add_argument(
        "--fail-on",
        type=str,
        default="exact",
        choices=("exact", "clinical"),
        help="Which match definition top-1 must FAIL (Subject must always "
             "succeed). 'exact' = wrong study_id. 'clinical' = not even a "
             "near-duplicate report (stricter, excludes near-duplicate hits).",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=10,
        help="Exactly this many examples are exported (one per subject_id, "
             "the first one found), no more.",
    )
    parser.add_argument(
        "--examples-dir",
        type=str,
        default="examples",
        help="Directory (relative to cwd) to write example_1, example_2, ... into.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to also save ALL qualifying examples (before the "
             "one-per-subject / --num-examples cut) as a flat CSV.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = ExperimentConfig(text_field=args.text_field, seed=args.seed).finalize()

    model, processor, device = load_clip(args.checkpoint)
    _, val_loader, df_val = build_dataloaders(config, mode="eval")

    image_features, text_features = extract_val_features(
        model, processor, val_loader, device
    )
    image_features = image_features.to(device)
    text_features = text_features.to(device)

    df_val = df_val.copy().reset_index(drop=True)
    df_val["study_id"] = df_val["image"].apply(get_study_id)
    if "subject_id" not in df_val.columns:
        raise ValueError(
            "df_val has no subject_id column -- can't identify same-patient "
            "retrievals without it."
        )

    if args.direction == "image_to_text":
        sim_matrix = torch.matmul(image_features, text_features.t())
    else:
        sim_matrix = torch.matmul(text_features, image_features.t())

    sim_np = sim_matrix.detach().cpu().numpy()
    study_ids = df_val["study_id"].values
    subject_ids = df_val["subject_id"].values
    group_ids = build_text_group_ids(df_val["text"])  # same grouping as "Clinical"

    rows = []
    n = sim_np.shape[0]
    for i in range(n):
        order = np.argsort(-sim_np[i])
        top1 = int(order[0])

        if args.fail_on == "exact":
            definition_failed = study_ids[top1] != study_ids[i]
        else:
            definition_failed = group_ids[top1] != group_ids[i]
        same_subject = subject_ids[top1] == subject_ids[i]

        # The failure mode: top-1 fails the chosen definition but is the SAME patient.
        if definition_failed and same_subject:
            # df_val row i already pairs the query with its own true
            # counterpart (each row is one image-report pair), so the
            # "true" item on the other side of the query is just that same
            # row's other column -- e.g. for a text_to_image query, row i's
            # "image" column is the photo that should have been retrieved.
            rows.append(
                {
                    "query_idx": i,
                    "query_study_id": study_ids[i],
                    "subject_id": subject_ids[i],
                    "query_image": df_val.loc[i, "image"],
                    "query_text": df_val.loc[i, "text"],
                    "retrieved_idx": top1,
                    "retrieved_study_id": study_ids[top1],
                    "retrieved_image": df_val.loc[top1, "image"],
                    "retrieved_text": df_val.loc[top1, "text"],
                    "similarity": float(sim_np[i, top1]),
                }
            )

    result_df = pd.DataFrame(rows)
    n_qualifying = len(result_df)

    if args.output:
        result_df.to_csv(args.output, index=False)
        print(f"Saved all {n_qualifying} qualifying examples to {args.output}")

    # One row per subject_id: rows were appended in ascending query_idx order,
    # so keep="first" is exactly "the first example" for that subject.
    deduped_df = result_df.drop_duplicates(subset="subject_id", keep="first")
    final_df = deduped_df.head(args.num_examples).reset_index(drop=True)

    definition_label = "Exact Match" if args.fail_on == "exact" else "Clinical"
    print(
        f"\n{n_qualifying} / {n} queries ({args.direction}) have a top-1 "
        f"retrieval that fails {definition_label} but shares the query's "
        f"subject_id ({definition_label} fails at R@1, Subject succeeds at R@1)."
    )
    print(f"{len(deduped_df)} distinct subjects among those.")
    print(f"Exporting {len(final_df)} example(s) (one per subject) to "
          f"'{args.examples_dir}/'.\n")

    os.makedirs(args.examples_dir, exist_ok=True)
    summary_rows = []
    for idx, row in enumerate(final_df.itertuples(index=False), start=1):
        folder = os.path.join(args.examples_dir, f"example_{idx}")
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
            f.write(f"subject_id: {row.subject_id}\n")
            f.write(f"similarity (top-1 score): {row.similarity:.4f}\n\n")
            f.write(f"query study_id: {row.query_study_id}\n")
            f.write(f"true image (should have been retrieved): {true_dst}\n")
            f.write(f"true report text:\n{row.query_text}\n\n")
            f.write(f"retrieved study_id: {row.retrieved_study_id}\n")
            f.write(f"retrieved image (actually retrieved): {retrieved_dst}\n")
            f.write(f"retrieved report text:\n{row.retrieved_text}\n")

        summary_rows.append(
            {
                "example": f"example_{idx}",
                "subject_id": row.subject_id,
                "similarity": row.similarity,
                "query_study_id": row.query_study_id,
                "retrieved_study_id": row.retrieved_study_id,
            }
        )
        print(f"example_{idx}: subject_id={row.subject_id}  "
              f"query study={row.query_study_id}  retrieved study={row.retrieved_study_id}")

    pd.DataFrame(summary_rows).to_csv(
        os.path.join(args.examples_dir, "examples_summary.csv"), index=False
    )


if __name__ == "__main__":
    main()
