# -*- coding: utf-8 -*-

import os
import ast
import re
import pandas as pd

# Define paths for the SLURM environment
BASE_DATA_DIR = "/groups/orentsur_group/work/omertole/mimic_data"
TRAIN_CSV_PATH = os.path.join(BASE_DATA_DIR, "mimic_cxr_aug_train.csv")
VAL_CSV_PATH = os.path.join(BASE_DATA_DIR, "mimic_cxr_aug_validate.csv")

# Define output paths for the preprocessed datasets
OUTPUT_TRAIN_CSV_PATH = os.path.join(BASE_DATA_DIR, "mimic_cxr_processed_train.csv")
OUTPUT_VAL_CSV_PATH = os.path.join(BASE_DATA_DIR, "mimic_cxr_processed_validate.csv")


def safe_literal_eval(val):
    """Safely convert string representations of lists to actual Python lists."""
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return val


def extract_sections(text):
    """Extract Findings/Find and Impression sections from raw text using regex."""
    if not isinstance(text, str):
        return "", ""

    # Flexible pattern for Impression variations (e.g., "Impression:", "Impressions", "Impression \ Impression")
    impression_pattern = r'Impression(?:s|(?:\s*\\\s*Impression))?:?'

    # Matches Findings/Find, and stops when reaching any variation of Impression or end of string
    findings_match = re.search(fr'Finding?s?:?\s*(.*?)(?={impression_pattern}|$)', text, re.DOTALL | re.IGNORECASE)
    
    # Matches the content after any variation of Impression
    impression_match = re.search(fr'{impression_pattern}\s*(.*)', text, re.DOTALL | re.IGNORECASE)

    findings = findings_match.group(1).strip() if findings_match else ""
    
    # If there is an impression match, use it. Otherwise, if no 'Impression' keyword exists, the whole text might be findings.
    if impression_match:
        impression = impression_match.group(1).strip()
    else:
        # If 'Impression:' is nowhere to be found, we don't assume the whole text is Impression
        impression = ""

    return findings, impression


def prepare_dataframe(csv_path):
    """Load, convert, extract studies, explode, and clean the dataset split."""
    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)

    new_rows = []

    for _, row in df.iterrows():
        # Safely evaluate the string representations into Python lists
        images = safe_literal_eval(row.get('image', '[]'))
        texts = safe_literal_eval(row.get('text', '[]'))
        
        # Specific view columns containing lists of paths
        ap_list = safe_literal_eval(row.get('AP', '[]'))
        pa_list = safe_literal_eval(row.get('PA', '[]'))
        lateral_list = safe_literal_eval(row.get('Lateral', '[]'))
        
        # Check if text_augment exists and parse it
        texts_aug = []
        if 'text_augment' in row and pd.notna(row['text_augment']):
            texts_aug = safe_literal_eval(row['text_augment'])
            
        if not isinstance(images, list) or not isinstance(texts, list):
            continue

        # Dictionary to group image metadata by study_id 
        # List to track the order of study_ids as they appear
        study_to_data = {}
        ordered_study_ids = []

        for img_path in images:
            # Extract study_id using regex (matches 's' followed by digits inside slashes)
            match = re.search(r'/s(\d+)/', img_path)
            if match:
                study_id = "s" + match.group(1) 
                
                # Identify the specific view for this individual image based on the columns
                current_view = None
                if isinstance(ap_list, list) and img_path in ap_list:
                    current_view = 'AP'
                elif isinstance(pa_list, list) and img_path in pa_list:
                    current_view = 'PA'
                elif isinstance(lateral_list, list) and img_path in lateral_list:
                    current_view = 'Lateral'
                
                # If it is a new study_id, initialize its list and record the order
                if study_id not in study_to_data:
                    study_to_data[study_id] = []
                    ordered_study_ids.append(study_id)
                
                # Append a dictionary containing both the image path and its specific mapped view
                study_to_data[study_id].append({
                    'image_path': img_path,
                    'view_type': current_view
                })

        # Create a new row for each unique study_id
        for i, study_id in enumerate(ordered_study_ids):
            # Extract the corresponding text based on the order of appearance
            study_text = texts[i] if i < len(texts) else ""
            study_text_aug = texts_aug[i] if i < len(texts_aug) else ""

            new_rows.append({
                'subject_id': row['subject_id'],
                'study_id': study_id,
                'image_meta': study_to_data[study_id],  # This is a list of dicts
                'text': study_text,
                'text_augment': study_text_aug,
            })

    # Convert the processed rows back into a new DataFrame (Study Level)
    df_studies = pd.DataFrame(new_rows)
    
    if df_studies.empty:
        print("Warning: Processed DataFrame is empty.")
        return df_studies

    # Explode the 'image_meta' column so each individual image gets its own row
    df_flat = df_studies.explode('image_meta').reset_index(drop=True)

    # Separate the dictionary inside 'image_meta' into individual columns
    df_flat['image'] = df_flat['image_meta'].apply(lambda x: x['image_path'] if isinstance(x, dict) else None)
    df_flat['view'] = df_flat['image_meta'].apply(lambda x: x['view_type'] if isinstance(x, dict) else None)
    
    # Drop the temporary metadata column
    df_flat = df_flat.drop(columns=['image_meta'])

    # --- FIXED SECTION: Safe extraction using list comprehension ---
    extracted_data = [extract_sections(x) for x in df_flat['text']]
    df_extracted = pd.DataFrame(extracted_data, columns=['findings_clean', 'impression_clean'], index=df_flat.index)
    df_flat = pd.concat([df_flat, df_extracted], axis=1)

    # Filter out rows where both sections are empty strings
    initial_count = len(df_flat)
    df_flat = df_flat[(df_flat['findings_clean'] != "") | (df_flat['impression_clean'] != "")]
    dropped_count = initial_count - len(df_flat)
    print(f"Dropped {dropped_count:,} rows where both Findings and Impression were empty.")
    # ---------------------------------------------------------------

    # Reorder columns to make the dataframe intuitive and organized
    ordered_cols = [
        'subject_id', 'study_id', 'image', 'view', 'AP', 'PA', 'Lateral',
        'text', 'text_augment', 'findings_clean', 'impression_clean'
    ]
    existing_cols = [col for col in ordered_cols if col in df_flat.columns]
    df_flat = df_flat[existing_cols]

    print(f"Processing complete. Total flattened rows remaining: {len(df_flat):,}")
    return df_flat


def main():
    # Process and save the training set
    print("--- Processing Training Dataset ---")
    df_train_processed = prepare_dataframe(TRAIN_CSV_PATH)
    df_train_processed.to_csv(OUTPUT_TRAIN_CSV_PATH, index=False)
    print(f"Saved processed training dataset to: {OUTPUT_TRAIN_CSV_PATH}\n")

    # Process and save the validation set
    print("--- Processing Validation Dataset ---")
    df_val_processed = prepare_dataframe(VAL_CSV_PATH)
    df_val_processed.to_csv(OUTPUT_VAL_CSV_PATH, index=False)
    print(f"Saved processed validation dataset to: {OUTPUT_VAL_CSV_PATH}\n")


if __name__ == "__main__":
    main()