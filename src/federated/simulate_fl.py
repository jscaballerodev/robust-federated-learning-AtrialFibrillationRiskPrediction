import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
import copy

PROCESSED_DATA = "data/processed/afib_24h_dataset.csv"
MODEL_DIR = "models"
SUMMARY_FILE = "project_summary.json"
NUM_CLIENTS = 10
MALICIOUS_FRACTION = 0.2  # 20% of clients are malicious
TRIM_FRACTION = 0.2       # For trimmed mean
ROUNDS = 20
LOCAL_EPOCHS = 3

# Define a simple MLP for tabular data
class TabularMLP(nn.Module):
    def __init__(self, input_dim):
        super(TabularMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.sigmoid(self.fc3(x))

def get_data_loaders(num_clients):
    if not os.path.exists(PROCESSED_DATA):
        raise FileNotFoundError(f"Processed data not found at {PROCESSED_DATA}")
        
    df = pd.read_csv(PROCESSED_DATA)
    if len(df) == 0:
        raise ValueError("Processed dataset is empty.")

    X = df.drop(columns=['SUBJECT_ID', 'HADM_ID', 'AFIB_LABEL']).values
    y = df['AFIB_LABEL'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Split training data among clients (IID for simplicity)
    client_data = []
    chunk_size = len(X_train) // num_clients
    for i in range(num_clients):
        X_chunk = X_train[i*chunk_size : (i+1)*chunk_size]
        y_chunk = y_train[i*chunk_size : (i+1)*chunk_size]
        dataset = TensorDataset(torch.FloatTensor(X_chunk), torch.FloatTensor(y_chunk).unsqueeze(1))
        client_data.append(DataLoader(dataset, batch_size=32, shuffle=True))
        
    test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test).unsqueeze(1)), batch_size=128)
    return client_data, test_loader, X_train.shape[1], len(df), float(y.mean())

def train_client(model, dataloader, is_malicious=False):
    """Trains a local model. If malicious, injects an attack (Label Flipping)."""
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()
    
    for epoch in range(LOCAL_EPOCHS):
        for data, target in dataloader:
            if is_malicious:
                target = 1.0 - target # ATTACK: Label Flipping!
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    return model.state_dict()

def trimmed_mean_aggregation(client_weights, trim_frac):
    """Robust Aggregation: Trimmed Mean to discard extreme weights (poisoning defense)."""
    global_weights = copy.deepcopy(client_weights[0])
    num_to_trim = int(len(client_weights) * trim_frac)
    
    for key in global_weights.keys():
        # Stack the weights for this layer from all clients
        stacked_weights = torch.stack([w[key] for w in client_weights], dim=0)
        
        if num_to_trim > 0:
            # Sort along the client dimension
            sorted_weights, _ = torch.sort(stacked_weights, dim=0)
            # Trim the top and bottom values
            trimmed_weights = sorted_weights[num_to_trim : -num_to_trim]
            # Take the mean of the remaining
            global_weights[key] = torch.mean(trimmed_weights, dim=0)
        else:
            global_weights[key] = torch.mean(stacked_weights, dim=0)
            
    return global_weights

def evaluate(model, test_loader):
    model.eval()
    y_true, y_pred_proba = [], []
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            y_true.extend(target.numpy())
            y_pred_proba.extend(output.numpy())
            
    y_true = np.array(y_true).flatten()
    y_pred_proba = np.array(y_pred_proba).flatten()
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Handle case with only 1 class in test set
    if len(np.unique(y_true)) == 1:
        return 0.0, 0.0
        
    return roc_auc_score(y_true, y_pred_proba), f1_score(y_true, y_pred)

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    try:
        client_loaders, test_loader, input_dim, total_patients, af_prevalence = get_data_loaders(NUM_CLIENTS)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
        
    global_model = TabularMLP(input_dim)
    
    num_malicious = int(NUM_CLIENTS * MALICIOUS_FRACTION)
    malicious_clients = set(range(num_malicious)) # First 'N' clients are malicious
    
    print(f"Simulating FL with {NUM_CLIENTS} clients. {num_malicious} are malicious (Label Flipping).")
    
    auroc, f1 = 0.0, 0.0
    for round_num in range(ROUNDS):
        client_weights = []
        for client_id in range(NUM_CLIENTS):
            local_model = TabularMLP(input_dim)
            local_model.load_state_dict(global_model.state_dict())
            
            is_malic = client_id in malicious_clients
            weights = train_client(local_model, client_loaders[client_id], is_malicious=is_malic)
            client_weights.append(weights)
            
        # Robust Aggregation
        aggregated_weights = trimmed_mean_aggregation(client_weights, trim_frac=TRIM_FRACTION)
        global_model.load_state_dict(aggregated_weights)
        
        auroc, f1 = evaluate(global_model, test_loader)
        print(f"Round {round_num+1}/{ROUNDS} | AUROC: {auroc:.4f} | F1: {f1:.4f}")
        
    # Save Model
    model_path = os.path.join(MODEL_DIR, "global_fl_model.pth")
    torch.save(global_model.state_dict(), model_path)
    print(f"Saved global FL model to {model_path}")
    
    # --- Generate/Update Project Summary JSON ---
    summary = {}
    if os.path.exists(SUMMARY_FILE):
        with open(SUMMARY_FILE, 'r') as f:
            summary = json.load(f)
            
    summary["project_name"] = "Robust Federated Learning for Atrial Fibrillation Risk Prediction"
    summary["data_context"] = {
        "total_patients": total_patients,
        "af_prevalence_percent": round(af_prevalence * 100, 2),
        "time_window": "First 24 hours of ICU admission"
    }
    summary["medical_context"] = "Prediction of AFib risk using 24-hour vital signs and lab values from ICU admissions."
    summary["architecture_explanation"] = (f"Federated Learning simulation with {NUM_CLIENTS} clients. "
                                           f"{MALICIOUS_FRACTION*100}% of clients execute a label-flipping poisoning attack. "
                                           "A Trimmed Mean aggregation algorithm defends against the poisoned updates.")
    
    if "metrics" not in summary:
        summary["metrics"] = {}
        
    summary["metrics"]["federated_robust_mlp"] = {
        "AUROC": round(float(auroc), 4),
        "F1": round(float(f1), 4)
    }
    
    with open(SUMMARY_FILE, "w") as f:
        json.dump(summary, f, indent=4)
        
    print(f"Updated {SUMMARY_FILE}")

if __name__ == "__main__":
    main()
