import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score

PROCESSED_DATA = "data/processed/afib_24h_dataset.csv"
MODEL_DIR = "models"
SUMMARY_FILE = "project_summary.json"

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # 1. Load Data
    if not os.path.exists(PROCESSED_DATA):
        print(f"Error: Processed data not found at {PROCESSED_DATA}")
        return

    print("Loading processed data...")
    df = pd.read_csv(PROCESSED_DATA)
    
    if len(df) == 0:
        print("Error: Processed dataset is empty.")
        return

    X = df.drop(columns=['SUBJECT_ID', 'HADM_ID', 'AFIB_LABEL'])
    y = df['AFIB_LABEL']
    
    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 3. Train Centralized XGBoost
    print("Training Baseline XGBoost...")
    # scale_pos_weight is important for class imbalance
    # Handle edge case where there might be 0 positive cases in the split
    if y_train.sum() == 0:
        print("Error: No positive cases found in the training split. Cannot calculate scale_pos_weight.")
        return
        
    scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum() 
    
    model = xgb.XGBClassifier(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=5, 
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='auc'
    )
    model.fit(X_train, y_train)
    
    # 4. Evaluate
    preds_proba = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)
    
    auroc = roc_auc_score(y_test, preds_proba)
    f1 = f1_score(y_test, preds)
    
    print(f"XGBoost Baseline - AUROC: {auroc:.4f}, F1: {f1:.4f}")
    
    # 5. Save Model Artifact
    model_path = os.path.join(MODEL_DIR, "baseline_xgb.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Saved baseline model to {model_path}")
        
    # 6. Update Project Summary JSON
    summary = {}
    if os.path.exists(SUMMARY_FILE):
        with open(SUMMARY_FILE, 'r') as f:
            summary = json.load(f)
    
    # Initialize basic structure if not exists
    if "project_name" not in summary:
        summary["project_name"] = "Robust Federated Learning for Atrial Fibrillation Risk Prediction"
        summary["data_context"] = {
            "total_patients": len(df),
            "af_prevalence_percent": round(float(y.mean()) * 100, 2),
            "time_window": "First 24 hours of ICU admission"
        }
    if "metrics" not in summary:
        summary["metrics"] = {}
        
    # Update XGBoost metrics
    summary["metrics"]["centralized_xgboost"] = {
        "AUROC": round(float(auroc), 4),
        "F1": round(float(f1), 4)
    }
    
    with open(SUMMARY_FILE, "w") as f:
        json.dump(summary, f, indent=4)
        
    return auroc, f1

if __name__ == "__main__":
    main()
