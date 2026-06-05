import os
import ast
import re
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Configuration
BASE_DATA_DIR = "/groups/orentsur_group/work/omertole/mimic_data"
MODEL_NAME = "microsoft/BiomedVLP-CXR-BERT-specialized" 
# MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def safe_literal_eval(val):
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return val

# Reuse the saved preprocessed CSV paths from preprocess.py
from preprocess import OUTPUT_TRAIN_CSV_PATH, OUTPUT_VAL_CSV_PATH

def prepare_and_embed(csv_path, output_pt_path):
    print(f"\n--- Processing: {os.path.basename(csv_path)} ---")
    # Read the saved, preprocessed CSV (must be produced by preprocess.py)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Preprocessed file not found: {csv_path}. Run preprocess.py first to generate it."
        )

    df = pd.read_csv(csv_path)
    if df.empty:
        print("No data to embed. Exiting.")
        return

    # Prefer the 'text' column (original text) as requested; fallback to 'impression_clean' if missing
    if 'text' in df.columns:
        texts = df['text'].fillna('').astype(str).tolist()
    elif 'impression_clean' in df.columns:
        texts = df['impression_clean'].fillna('').astype(str).tolist()
    else:
        raise ValueError("Preprocessed CSV does not contain 'text' or 'impression_clean' columns")
    print(f"Total rows to embed: {len(texts):,}")

    # 2. Extract Embeddings
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True).to(device)
    model.eval()
    
    dataset = torch.utils.data.TensorDataset(torch.arange(len(texts))) # Just to batch indices
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    all_embeddings = []
    print("Running BiomedVLP-CXR-BERT inference...")
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), 64)):
            batch_texts = texts[i : i + 64]
            inputs = tokenizer(
                batch_texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=128
            ).to(device)
            
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :] # CLS token
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            all_embeddings.append(embeddings.cpu())
            
    final_embeddings = torch.cat(all_embeddings, dim=0)
    torch.save(final_embeddings, output_pt_path)
    print(f"Saved {final_embeddings.shape[0]} embeddings to: {output_pt_path}")

if __name__ == "__main__":
    # Process Train
    prepare_and_embed(
        csv_path=OUTPUT_TRAIN_CSV_PATH,
        output_pt_path=os.path.join(BASE_DATA_DIR, "train_biomedvlp_cxr_bert_embeddings.pt")
    )

    # Process Val
    prepare_and_embed(
        csv_path=OUTPUT_VAL_CSV_PATH,
        output_pt_path=os.path.join(BASE_DATA_DIR, "val_biomedvlp_cxr_bert_embeddings.pt")
    )
    print("\nDone! Both files are ready for Soft-CLIP training.")