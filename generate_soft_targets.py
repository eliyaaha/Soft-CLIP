import os
import ast
import re
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# Configuration
BASE_DATA_DIR = "/groups/orentsur_group/work/omertole/mimic_data"
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def safe_literal_eval(val):
    try: return ast.literal_eval(val)
    except (ValueError, SyntaxError): return val

def extract_sections(text):
    if not isinstance(text, str): return ""
    match = re.search(r'Impression:\s*(.*)', text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text

def prepare_and_embed(csv_path, output_pt_path):
    print(f"\n--- Processing: {os.path.basename(csv_path)} ---")
    df = pd.read_csv(csv_path)
    
    # 1. Same logic as notebook (Flattening)
    df['image'] = df['image'].apply(safe_literal_eval)
    df['text'] = df['text'].apply(safe_literal_eval)
    df_flat = df.explode('image').reset_index(drop=True)
    df_flat['text'] = df_flat['text'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else x)
    df_flat['impression_clean'] = df_flat['text'].apply(extract_sections)
    
    texts = df_flat['impression_clean'].tolist()
    print(f"Total rows to embed: {len(texts):,}")

    # 2. Extract Embeddings
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    
    dataset = torch.utils.data.TensorDataset(torch.arange(len(texts))) # Just to batch indices
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    all_embeddings = []
    print("Running BioClinicalBERT inference...")
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
        csv_path=os.path.join(BASE_DATA_DIR, "mimic_cxr_aug_train.csv"),
        output_pt_path=os.path.join(BASE_DATA_DIR, "train_bioclinicalbert_embeddings.pt")
    )
    
    # Process Val
    prepare_and_embed(
        csv_path=os.path.join(BASE_DATA_DIR, "mimic_cxr_aug_validate.csv"),
        output_pt_path=os.path.join(BASE_DATA_DIR, "val_bioclinicalbert_embeddings.pt")
    )
    print("\nDone! Both files are ready for Soft-CLIP training.")