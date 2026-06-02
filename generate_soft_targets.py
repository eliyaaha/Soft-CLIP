import os
import ast
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# Configuration
BASE_DATA_DIR = "/groups/orentsur_group/work/omertole/mimic_data"
TRAIN_CSV_PATH = os.path.join(BASE_DATA_DIR, "mimic_cxr_aug_train.csv")
OUTPUT_PATH = os.path.join(BASE_DATA_DIR, "train_bioclinicalbert_embeddings.pt")
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def safe_literal_eval(val):
    try: return ast.literal_eval(val)
    except (ValueError, SyntaxError): return val

class TextOnlyDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx): return self.texts[idx]

def generate_embeddings():
    print("Loading data...")
    df = pd.read_csv(TRAIN_CSV_PATH)
    
    # Process text column exactly as in training
    df['text'] = df['text'].apply(safe_literal_eval)
    df['text'] = df['text'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else x)
    
    # Simple regex to get Impression section
    import re
    def get_impression(text):
        if not isinstance(text, str): return ""
        match = re.search(r'Impression:\s*(.*)', text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else text

    texts = df['text'].apply(get_impression).tolist()
    
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    
    dataset = TextOnlyDataset(texts)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    all_embeddings = []
    
    print("Extracting embeddings...")
    with torch.no_grad():
        for batch_texts in tqdm(loader):
            inputs = tokenizer(
                list(batch_texts), 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=128
            ).to(device)
            
            outputs = model(**inputs)
            # Use CLS token for sentence representation
            embeddings = outputs.last_hidden_state[:, 0, :]
            # L2 Normalize
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            all_embeddings.append(embeddings.cpu())
            
    final_embeddings = torch.cat(all_embeddings, dim=0)
    torch.save(final_embeddings, OUTPUT_PATH)
    print(f"Successfully saved embeddings to {OUTPUT_PATH}")

if __name__ == "__main__":
    generate_embeddings()