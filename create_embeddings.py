"""Generate precomputed BERT text embeddings for soft-CLIP training.

Pipeline order: run ``preprocess.py`` first; this script reads its outputs.

Examples
--------
    # Default: BiomedVLP over the raw `text` column
    python create_embeddings.py

    # Different model
    python create_embeddings.py --model bioclinicalbert
    python create_embeddings.py --model gemma_embed

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
import re

import pandas as pd
import torch
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer

from preprocess import (
    BASE_DATA_DIR,
    OUTPUT_TRAIN_CSV_PATH,
    OUTPUT_VAL_CSV_PATH,
)

load_dotenv()

MODELS = {
    "biomedvlp": "microsoft/BiomedVLP-CXR-BERT-specialized",
    "bioclinicalbert": "emilyalsentzer/Bio_ClinicalBERT",
    "gemma_embed": "google/embeddinggemma-300m",
}

SUPPORTED_FIELDS = ("text", "findings_clean", "impression_clean")
DEFAULT_FIELD = "text"
DEFAULT_MODEL = "biomedvlp"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed processed CSV text with a BERT/Gemma-family model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODELS.keys()),
        default=DEFAULT_MODEL,
        help="Which model to use for text embedding.",
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
        "--pooling",
        choices=("auto", "projection", "mean", "cls"),
        default="auto",
        help=(
            "How to reduce token states to one vector. 'auto' picks "
            "'projection' for biomedvlp and 'mean' otherwise. 'cls' reproduces "
            "the original (incorrect) behaviour for comparison. Ignored for "
            "gemma_embed, which pools internally."
        ),
    )
    parser.add_argument(
        "--sentence-level",
        action="store_true",
        help=(
            "Embed each sentence of the report separately and average. CXR-BERT "
            "is documented for SENTENCE embeddings (its reported inputs average "
            "~58 tokens); a full findings+impression report is out of that "
            "regime and gets truncated. Recommended for --field text."
        ),
    )
    parser.add_argument(
        "--tag-suffix",
        default="",
        help=(
            "Appended to the output filename, e.g. --tag-suffix cls writes "
            "train_biomedvlp_text_cls_embeddings.pt. Use it to keep old and new "
            "pooling variants side by side."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute and overwrite the output .pt file if it already exists.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "val"),
        default=["train", "val"],
        help=(
            "Which splits to embed. Use `--splits val` when screening many "
            "variants with compare_embeddings.py: the intrinsic comparison only "
            "needs val (~3k rows vs ~350k), so the whole grid costs minutes "
            "instead of hours. Generate train embeddings only for the winner."
        ),
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


def build_tag(model_slug: str, field: str, pooling: str,
              sentence_level: bool = False, tag_suffix: str = "") -> str:
    """Embeddings tag. Pooling is ALWAYS encoded, deliberately.

    If the filename did not record how the vectors were extracted, it would be
    possible to compare e.g. `biomedvlp` under `projection` against
    `bioclinicalbert` under `mean` and read the difference as a property of the
    models. Putting pooling in the name makes that confound visible at the point
    where the comparison is made.
    """
    parts = [model_slug, field, pooling]
    if sentence_level:
        parts.append("sent")
    if tag_suffix:
        parts.append(tag_suffix)
    return "_".join(parts)


def _output_path(split: str, tag: str) -> str:
    return os.path.join(BASE_DATA_DIR, f"{split}_{tag}_embeddings.pt")


_SENTENCE_SPLIT = re.compile(r"(?<=[.;])\s+|\n+")
# Bare section headers ("FINDINGS:", "IMPRESSION:") carry no clinical content
# but would otherwise be averaged in as if they were sentences, pulling every
# report toward a shared, meaningless direction.
_HEADER_ONLY = re.compile(r"^\s*(findings?|impressions?|comparison|indication|technique|history)\s*:?\s*$", re.I)


def split_sentences(report: str) -> list:
    """Split a report into content-bearing sentences."""
    parts = []
    for chunk in _SENTENCE_SPLIT.split(str(report)):
        chunk = chunk.strip()
        if not chunk or _HEADER_ONLY.match(chunk):
            continue
        # Drop fragments with no alphabetic content (stray punctuation, "___").
        if not re.search(r"[a-zA-Z]{2,}", chunk):
            continue
        parts.append(chunk)
    return parts


def _embed_batch(model, tokenizer, texts, pooling, max_length, device) -> torch.Tensor:
    """Encode a list of strings into one vector each, L2-normalised.

    Raw ``last_hidden_state[:, 0, :]`` (the "cls" option) is the wrong
    representation for both encoders used here, and it is what produced the
    unimodal similarity distribution reported in the write-up:

    * ``BiomedVLP-CXR-BERT-specialized`` ships a projection head trained
      specifically to make CLS embeddings comparable. The model card's own usage
      example calls ``get_projected_text_embeddings`` and then takes a plain dot
      product -- the projected output is already normalised. The pre-projection
      CLS vector is simply not the space the model learned similarity in.
    * ``Bio_ClinicalBERT`` has no sentence-level objective at all. CLS from a
      masked-LM-only BERT is anisotropic: every pair lands in a narrow
      similarity band regardless of content.

    A healthy clinical similarity matrix on MIMIC-CXR should be clearly BIMODAL
    -- a dense cluster of near-identical normal studies plus a long tail of
    distinct pathology. Check the histogram before tuning anything downstream.
    """
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    ).to(device)

    if pooling == "projection":
        # Documented API from the model card. It runs its own forward pass, so
        # do NOT also call model(**encoded) -- that would double the compute.
        if not hasattr(model, "get_projected_text_embeddings"):
            raise AttributeError(
                "This model has no `get_projected_text_embeddings`. It is "
                "specific to CXR-BERT; use `--pooling mean` for other encoders."
            )
        embeddings = model.get_projected_text_embeddings(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
        )
    else:
        outputs = model(**encoded)
        if pooling == "cls":
            embeddings = outputs.last_hidden_state[:, 0, :]
        elif pooling == "mean":
            mask = encoded["attention_mask"].unsqueeze(-1).to(
                outputs.last_hidden_state.dtype
            )
            summed = (outputs.last_hidden_state * mask).sum(dim=1)
            embeddings = summed / mask.sum(dim=1).clamp_min(1e-9)
        else:
            raise ValueError(f"Unknown pooling strategy: {pooling!r}")

    # get_projected_text_embeddings already normalises; doing it again is a
    # no-op and keeps every branch on the same footing.
    return embeddings / embeddings.norm(dim=-1, keepdim=True).clamp_min(1e-9)


def _embed_report_by_sentence(
    model, tokenizer, report, base_pooling, max_length, device
) -> torch.Tensor:
    """Embed each sentence separately and average.

    CXR-BERT was trained and is documented for *sentence* embeddings ("extract
    radiological sentence embeddings", and its reported token statistics average
    ~58 tokens). A full MIMIC-CXR report -- findings plus impression -- is well
    outside that regime, and truncating it to a fixed window silently discards
    the tail. Averaging sentence embeddings keeps every input inside the length
    the encoder was trained for and covers the whole report.
    """
    sentences = split_sentences(report) or [str(report).strip() or ""]
    per_sentence = _embed_batch(
        model, tokenizer, sentences, base_pooling, max_length, device
    )
    pooled = per_sentence.mean(dim=0, keepdim=True)
    return pooled / pooled.norm(dim=-1, keepdim=True).clamp_min(1e-9)


def _default_pooling(model_slug: str) -> str:
    if model_slug == "gemma_embed":
        # SentenceTransformer.encode_document pools internally; --pooling has no
        # effect. Label it "native" so the tag never claims a pooling that was
        # not applied.
        return "native"
    return "projection" if model_slug == "biomedvlp" else "mean"


def _resolve_pooling(model_slug: str, requested: str) -> str:
    """Resolve the pooling label, honouring what each model can actually do.

    ``gemma_embed`` goes through SentenceTransformer.encode_document, which does
    its own tokenisation and pooling internally -- no pooling option applies. It
    is labelled ``native`` so the tag never claims a strategy that was not used.
    """
    if model_slug == "gemma_embed":
        return "native"
    if requested == "auto":
        return _default_pooling(model_slug)
    if requested == "projection" and model_slug != "biomedvlp":
        raise ValueError(
            f"--pooling projection is only available for biomedvlp "
            f"(CXR-BERT's projection head); {model_slug} has no such head. "
            f"Use --pooling mean for a matched cross-model comparison."
        )
    return requested


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
    model_slug: str,
    model_hf_id: str,
    field: str,
    batch_size: int,
    max_length: int,
    overwrite: bool,
    device: torch.device,
    pooling: str = "auto",
    sentence_level: bool = False,
) -> None:
    print(f"\n--- Processing: {os.path.basename(csv_path)} ---")

    if pooling == "auto":
        pooling = _default_pooling(model_slug)
    print(f"Pooling: {pooling}{'  (per sentence, then averaged)' if sentence_level else ''}")

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

    # Branching logic based on the model framework needed
    if model_slug == "gemma_embed":
        print(f"Loading SentenceTransformer: {model_hf_id}...")
        hf_token = os.getenv("HF_TOKEN")
        model = SentenceTransformer(model_hf_id, device=str(device), token=hf_token)
        
        # Set max_length directly on the model instance
        model.max_seq_length = max_length
        
        print(f"Running {model_hf_id} inference...")
        # encode_document automatically handles tokenization, left-padding,
        # causal pooling, and normalization internally.
        embeddings_np = model.encode_document(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        final_embeddings = torch.from_numpy(embeddings_np)
        
    else:
        # Standard workflow for BERT/Encoder-only Hugging Face models
        tokenizer = AutoTokenizer.from_pretrained(model_hf_id, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_hf_id, trust_remote_code=True).to(device)
        model.eval()

        all_embeddings = []
        print(f"Running {model_hf_id} inference...")
        with torch.no_grad():
            if sentence_level:
                for report in tqdm(texts):
                    all_embeddings.append(
                        _embed_report_by_sentence(
                            model, tokenizer, report, pooling, max_length, device
                        ).float().cpu()
                    )
            else:
                for i in tqdm(range(0, len(texts), batch_size)):
                    all_embeddings.append(
                        _embed_batch(
                            model,
                            tokenizer,
                            texts[i : i + batch_size],
                            pooling,
                            max_length,
                            device,
                        ).float().cpu()
                    )

        final_embeddings = torch.cat(all_embeddings, dim=0)

    os.makedirs(os.path.dirname(output_pt_path), exist_ok=True)
    torch.save(final_embeddings, output_pt_path)
    print(f"Saved {final_embeddings.shape[0]} embeddings to: {output_pt_path}")
    _report_similarity_shape(final_embeddings)


def _report_similarity_shape(embeddings: torch.Tensor, sample: int = 2000) -> None:
    """Print the off-diagonal similarity distribution as a sanity check.

    A wide unimodal blob means the embeddings carry little usable structure and
    the soft targets built from them will be close to noise. Bimodal is what a
    working clinical similarity signal looks like on MIMIC-CXR.
    """
    n = embeddings.size(0)
    if n < 2:
        return
    idx = torch.randperm(n)[: min(sample, n)]
    sub = embeddings[idx].float()
    sim = sub @ sub.t()
    off = sim[~torch.eye(sub.size(0), dtype=torch.bool)]
    print(
        f"  off-diagonal cosine similarity: mean={off.mean():.4f} "
        f"std={off.std():.4f} "
        f"p50={off.median():.4f} "
        f"p90={off.quantile(0.90):.4f} "
        f"p99={off.quantile(0.99):.4f}"
    )


def main() -> None:
    args = parse_args()

    if args.list:
        _list_existing_embeddings()
        return

    model_slug = args.model
    model_hf_id = MODELS[model_slug]
    field = args.field
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pooling = _resolve_pooling(model_slug, args.pooling)
    if args.pooling == "auto" and model_slug != "gemma_embed":
        print(
            f"\n!! --pooling auto resolved to {pooling!r} for {model_slug}.\n"
            f"   'auto' picks each model's NATIVE representation, which differs "
            f"between models.\n"
            f"   That is the right choice for 'which pipeline works best', but it "
            f"CONFOUNDS\n"
            f"   a model-vs-model comparison. For a controlled comparison pass the "
            f"same\n"
            f"   --pooling to every model (e.g. --pooling mean). See "
            f"compare_embeddings.py.\n"
        )

    tag = build_tag(model_slug, field, pooling, args.sentence_level, args.tag_suffix)

    print(f"Model    : {model_slug}  ({model_hf_id})")
    print(f"Field    : {field}")
    print(f"Pooling  : {pooling}{'  (per sentence)' if args.sentence_level else ''}")
    print(f"Tag      : {tag}")
    print(f"Device   : {device}")

    if not args.overwrite:
        print(
            "\nNOTE: any .pt file generated before the pooling fix used raw CLS "
            "and has a different filename now, so it will not be picked up. Pass "
            "--overwrite if you are regenerating an existing tag."
        )

    split_paths = {"train": OUTPUT_TRAIN_CSV_PATH, "val": OUTPUT_VAL_CSV_PATH}
    for split in args.splits:
        csv_path = split_paths[split]
        prepare_and_embed(
            csv_path=csv_path,
            output_pt_path=_output_path(split, tag),
            model_slug=model_slug,
            model_hf_id=model_hf_id,
            field=field,
            batch_size=args.batch_size,
            max_length=args.max_length,
            overwrite=args.overwrite,
            device=device,
            pooling=pooling,
            sentence_level=args.sentence_level,
        )

    print(
        f"\nDone. Use --embeddings-tag {tag} "
        f"in train_soft_clip.py to consume these files."
    )


if __name__ == "__main__":
    main()