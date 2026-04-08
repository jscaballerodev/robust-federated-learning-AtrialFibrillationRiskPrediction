import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_mock_mimic():
    """Generates synthetic MIMIC-like CSV data for testing the pipeline."""
    os.makedirs('data/raw', exist_ok=True)
    num_patients = 1000
    print(f"Generating mock data for {num_patients} patients...")

    np.random.seed(42)
    subject_ids = np.arange(1, num_patients + 1)
    hadm_ids = subject_ids + 100000

    # 1. ADMISSIONS.csv
    admittimes = [datetime(2050, 1, 1) + timedelta(days=np.random.randint(0, 365)) for _ in range(num_patients)]
    admissions = pd.DataFrame({'SUBJECT_ID': subject_ids, 'HADM_ID': hadm_ids, 'ADMITTIME': admittimes})
    admissions.to_csv('data/raw/ADMISSIONS.csv', index=False)
    print("Generated ADMISSIONS.csv")

    # 2. DIAGNOSES_ICD.csv (Simulate ~15% AFib prevalence)
    is_afib = np.random.rand(num_patients) < 0.15
    # 42731 is AFib, 25000 is Diabetes (dummy alternative)
    icd_codes = np.where(is_afib, '42731', '25000') 
    diagnoses = pd.DataFrame({'SUBJECT_ID': subject_ids, 'HADM_ID': hadm_ids, 'ICD9_CODE': icd_codes})
    diagnoses.to_csv('data/raw/DIAGNOSES_ICD.csv', index=False)
    print("Generated DIAGNOSES_ICD.csv")

    # 3. CHARTEVENTS.csv (Vitals)
    chart_data = []
    vitals_ids = [211, 51, 8368, 52]
    for i in range(num_patients):
        sid = subject_ids[i]
        hid = hadm_ids[i]
        admit = admittimes[i]
        # Generate 15 chart events per patient
        for _ in range(15):
            item = np.random.choice(vitals_ids)
            # Events within 0 to 48 hours of admission
            offset = timedelta(hours=np.random.uniform(0, 48)) 
            # Differentiate values a bit based on ITEMID and AFib status for ML signal
            base_val = 80 if item == 211 else 120 
            noise = np.random.normal(0, 15)
            # Add some synthetic signal: AFib patients might have slightly higher HR (211)
            signal = 15 if (is_afib[i] and item == 211) else 0 
            val = base_val + noise + signal
            
            chart_data.append([sid, hid, item, admit + offset, val])
            
    chartevents = pd.DataFrame(chart_data, columns=['SUBJECT_ID', 'HADM_ID', 'ITEMID', 'CHARTTIME', 'VALUENUM'])
    chartevents.to_csv('data/raw/CHARTEVENTS.csv', index=False)
    print("Generated CHARTEVENTS.csv")

    # 4. LABEVENTS.csv (Labs)
    lab_data = []
    labs_ids = [50971, 50912]
    for i in range(num_patients):
        sid = subject_ids[i]
        hid = hadm_ids[i]
        admit = admittimes[i]
        # Generate 5 lab events per patient
        for _ in range(5):
            item = np.random.choice(labs_ids)
            offset = timedelta(hours=np.random.uniform(0, 48))
            base_val = 4.0 if item == 50971 else 1.0
            val = np.random.normal(base_val, 0.5)
            lab_data.append([sid, hid, item, admit + offset, val])
            
    labevents = pd.DataFrame(lab_data, columns=['SUBJECT_ID', 'HADM_ID', 'ITEMID', 'CHARTTIME', 'VALUENUM'])
    labevents.to_csv('data/raw/LABEVENTS.csv', index=False)
    print("Generated LABEVENTS.csv")
    print("Mock data generation complete!")

if __name__ == "__main__":
    generate_mock_mimic()
