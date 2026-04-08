import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

def extract_cohort():
    """Identifies the target cohort (AFib positive vs negative) from diagnoses."""
    print("Loading Admissions and Diagnoses...")
    
    # In a real environment, you'd ensure these files exist before running.
    # We will wrap in try-except for robustness.
    try:
        admissions = pd.read_csv(os.path.join(RAW_DIR, "ADMISSIONS.csv"), usecols=['SUBJECT_ID', 'HADM_ID', 'ADMITTIME'])
        diagnoses = pd.read_csv(os.path.join(RAW_DIR, "DIAGNOSES_ICD.csv"), usecols=['SUBJECT_ID', 'HADM_ID', 'ICD9_CODE'])
    except FileNotFoundError as e:
        print(f"Error: Missing raw data file. Please ensure MIMIC CSVs are in {RAW_DIR}")
        print(e)
        return pd.DataFrame()
    
    # AFib ICD-9 code is typically 42731
    diagnoses['AFIB_LABEL'] = diagnoses['ICD9_CODE'].apply(lambda x: 1 if str(x).startswith('42731') else 0)
    
    # Aggregate to admission level (1 if AFib anytime during admission, else 0)
    cohort = diagnoses.groupby(['SUBJECT_ID', 'HADM_ID'])['AFIB_LABEL'].max().reset_index()
    cohort = pd.merge(cohort, admissions, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    cohort['ADMITTIME'] = pd.to_datetime(cohort['ADMITTIME'])
    return cohort

def process_events_chunked(filename, cohort, item_ids, chunksize=1000000):
    """Reads large event files in chunks, filters for 24h window, and aggregates."""
    print(f"Processing {filename} in chunks...")
    filepath = os.path.join(RAW_DIR, filename)
    
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found. Skipping.")
        return pd.DataFrame()
    
    aggregated_chunks = []
    
    for chunk in pd.read_csv(filepath, chunksize=chunksize, usecols=['SUBJECT_ID', 'HADM_ID', 'ITEMID', 'CHARTTIME', 'VALUENUM']):
        # Filter for relevant features (Heart rate, BP, Potassium, etc.)
        chunk = chunk[chunk['ITEMID'].isin(item_ids)]
        if chunk.empty: continue
            
        chunk['CHARTTIME'] = pd.to_datetime(chunk['CHARTTIME'])
        
        # Merge with cohort to get admit time and label
        merged = pd.merge(chunk, cohort, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
        
        # Filter for first 24 hours (86400 seconds)
        merged['TIME_DIFF'] = (merged['CHARTTIME'] - merged['ADMITTIME']).dt.total_seconds()
        windowed = merged[(merged['TIME_DIFF'] >= 0) & (merged['TIME_DIFF'] <= 86400)]
        
        # Aggregate stats (mean, min, max) per patient, per item
        agg = windowed.groupby(['SUBJECT_ID', 'HADM_ID', 'ITEMID'])['VALUENUM'].agg(['mean', 'min', 'max']).reset_index()
        aggregated_chunks.append(agg)

    if not aggregated_chunks:
        return pd.DataFrame()

    # Final aggregation across all chunks
    final_agg = pd.concat(aggregated_chunks).groupby(['SUBJECT_ID', 'HADM_ID', 'ITEMID']).mean().reset_index()
    
    # Pivot to wide format (one row per admission)
    wide_df = final_agg.pivot_table(index=['SUBJECT_ID', 'HADM_ID'], columns='ITEMID', values=['mean', 'min', 'max'])
    wide_df.columns = [f"{stat}_{item}" for stat, item in wide_df.columns]
    return wide_df.reset_index()

def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    cohort = extract_cohort()
    
    if cohort.empty:
        print("Cohort extraction failed. Exiting pipeline.")
        return

    # Example ITEMIDs (e.g., 211=Heart Rate, 50971=Potassium - mapping depends on MIMIC version)
    vitals_ids = [211, 51, 8368, 52] # Mock IDs for HR, BP, etc.
    labs_ids = [50971, 50912]        # Mock IDs for Potassium, Creatinine, etc.
    
    vitals_df = process_events_chunked("CHARTEVENTS.csv", cohort, vitals_ids)
    labs_df = process_events_chunked("LABEVENTS.csv", cohort, labs_ids)
    
    # Merge everything
    final_data = pd.merge(cohort[['SUBJECT_ID', 'HADM_ID', 'AFIB_LABEL']], vitals_df, on=['SUBJECT_ID', 'HADM_ID'], how='left')
    final_data = pd.merge(final_data, labs_df, on=['SUBJECT_ID', 'HADM_ID'], how='left')
    
    # Basic Median Imputation (Revised to use numeric_only=True to prevent warnings)
    numeric_cols = final_data.select_dtypes(include=[np.number]).columns
    final_data[numeric_cols] = final_data[numeric_cols].fillna(final_data[numeric_cols].median())
    
    # Drop rows that still have NaNs (e.g., entirely empty columns) or fill with 0 as fallback
    final_data.fillna(0, inplace=True)

    output_path = os.path.join(PROCESSED_DIR, "afib_24h_dataset.csv")
    final_data.to_csv(output_path, index=False)
    print(f"Preprocessing complete. Saved to {output_path}")

if __name__ == "__main__":
    main()
