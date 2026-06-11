"""Generate precomputed BERT text embeddings for soft-CLIP training.

Pipeline order: run ``preprocess.py`` first; this script reads its outputs.

Examples
--------
    # Default: BiomedVLP over the raw `text` column
    python create_embeddings.py

    # Different model
    python create_embeddings.py --model bioclinicalbert

    # Different field (impression-only or findings-only ablations)
    python create_embeddings.py --field impression_clean
    python create_embeddings.py --field findings_clean

    # Re-generate even if the output already exists
    python create_embeddings.py --field findings_clean --overwrite

    # Inspect what's already on disk
    python create_embeddings.py --list

Output naming: ``{split}_{model_slug}_{field}_embeddings.pt`` under BASE_DATA_DIR.
Each (model, field) combination writes its own files, so different ablations
accumulate side-by-side and existing files are never silently overwritten.
"""

import argparse
import glob
import os

import pandas as pd
import torch
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from preprocess import (
    BASE_DATA_DIR,
    OUTPUT_TRAIN_CSV_PATH,
    OUTPUT_VAL_CSV_PATH,
)

load_dotenv()

MODELS = {
    "biomedvlp": "microsoft/BiomedVLP-CXR-BERT-specialized",
    "bioclinicalbert": "emilyalsentzer/Bio_ClinicalBERT",
}

SUPPORTED_FIELDS = ("text", "findings_clean", "impression_clean")
DEFAULT_FIELD = "text"
DEFAULT_MODEL = "biomedvlp"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed processed CSV text with a BERT-family model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODELS.keys()),
        default=DEFAULT_MODEL,
        help="Which BERT model to use for text embedding.",
    )
    parser.add_argument(
        "--field",
        choices=SUPPORTED_FIELDS,
        default=DEFAULT_FIELD,
        help="Which text column from the processed CSV to embed.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Tokenizer/inference batch size.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Tokenizer max sequence length.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute and overwrite the output .pt file if it already exists.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List existing *_embeddings.pt files in BASE_DATA_DIR and exit.",
    )
    return parser.parse_args()


def _list_existing_embeddings() -> None:
    pattern = os.path.join(BASE_DATA_DIR, "*_embeddings.pt")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No embedding files found under {BASE_DATA_DIR}")
        return
    print(f"Existing embedding files in {BASE_DATA_DIR}:")
    for path in files:
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  {os.path.basename(path)}  ({size_mb:.1f} MB)")


def _output_path(split: str, model_slug: str, field: str) -> str:
    return os.path.join(BASE_DATA_DIR, f"{split}_{model_slug}_{field}_embeddings.pt")


def _resolve_text_column(df: pd.DataFrame, field: str) -> pd.Series:
    if field not in df.columns:
        available = [c for c in df.columns if c in SUPPORTED_FIELDS]
        raise ValueError(
            f"Field {field!r} not found in processed CSV. "
            f"Available supported fields: {available}"
        )
    return df[field].fillna("").astype(str)


def prepare_and_embed(
    csv_path: str,
    output_pt_path: str,
    *,
    model_hf_id: str,
    field: str,
    batch_size: int,
    max_length: int,
    overwrite: bool,
    device: torch.device,
) -> None:
    print(f"\n--- Processing: {os.path.basename(csv_path)} ---")

    if os.path.exists(output_pt_path) and not overwrite:
        print(
            f"Output already exists, skipping: {output_pt_path}\n"
            f"(pass --overwrite to regenerate)"
        )
        return

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Preprocessed file not found: {csv_path}. "
            f"Run `python preprocess.py` first."
        )

    df = pd.read_csv(csv_path)
    if df.empty:
        print("No data to embed. Exiting.")
        return

    texts = _resolve_text_column(df, field).tolist()
    print(f"Embedding column {field!r}  |  rows: {len(texts):,}")

    tokenizer = AutoTokenizer.from_pretrained(model_hf_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_hf_id, trust_remote_code=True).to(device)
    model.eval()

    all_embeddings = []
    print(f"Running {model_hf_id} inference...")
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i : i + batch_size]
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)

            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            all_embeddings.append(embeddings.cpu())

    final_embeddings = torch.cat(all_embeddings, dim=0)
    os.makedirs(os.path.dirname(output_pt_path), exist_ok=True)
    torch.save(final_embeddings, output_pt_path)
    print(f"Saved {final_embeddings.shape[0]} embeddings to: {output_pt_path}")


def main() -> None:
    args = parse_args()

    if args.list:
        _list_existing_embeddings()
        return

    model_slug = args.model
    model_hf_id = MODELS[model_slug]
    field = args.field
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Model : {model_slug}  ({model_hf_id})")
    print(f"Field : {field}")
    print(f"Tag   : {model_slug}_{field}")
    print(f"Device: {device}")

    for split, csv_path in (("train", OUTPUT_TRAIN_CSV_PATH), ("val", OUTPUT_VAL_CSV_PATH)):
        prepare_and_embed(
            csv_path=csv_path,
            output_pt_path=_output_path(split, model_slug, field),
            model_hf_id=model_hf_id,
            field=field,
            batch_size=args.batch_size,
            max_length=args.max_length,
            overwrite=args.overwrite,
            device=device,
        )

    print(
        f"\nDone. Use --embeddings-tag {model_slug}_{field} "
        f"in train_soft_clip.py to consume these files."
    )


if __name__ == "__main__":
    main()
