"""
scripts/federated_simulation.py
================================
Step 9.1: Deterministic Local Federated Learning Simulation for FEDrA.

Core Functions:
1. Load and align the canonical dataset features (URL: 22, HTML: 12, Visual: 1280 -> Fused: 1314).
2. Deterministic client dataset partitioning (IID vs Label-Skew Non-IID).
3. Local client training on Fusion MLP (377,857 parameters, 4 dense layers: 1314 -> 256 -> 128 -> 64 -> 1).
4. Delta computation: ΔW_i = W_i - W_t.
5. Standard Federated Averaging (FedAvg): W_(t+1) = W_t + Σ (n_i / Σ n_j) ΔW_i.
6. Multi-round evaluation strictly against the 198-sample held-out test split.
7. Zero raw data leakage into FL artifacts.
8. 100% deterministic reproducibility under fixed random seeds.

NOTE: This is a standalone local Python simulation for validating mathematical FL optimization.
It does NOT execute in browser or connect to network servers.
"""

import os
import sys
import json
import copy
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Ensure scripts directory is on sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from url_features import CANONICAL_URL_FEATURE_NAMES, URL_FEATURE_DIM

FEAT_DIR = os.path.join(BASE_DIR, "Dataset", "features")
MANIFEST_PATH = os.path.join(BASE_DIR, "Dataset", "manifest.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
FUSION_MODEL_PATH = os.path.join(MODELS_DIR, "fusion_model.pkl")
DEFAULT_ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts", "federated")

WEIGHTS = {"url": 0.4, "html": 0.3, "visual": 0.3}


# ── 1. Model Definition (Fusion MLP PyTorch Equivalent) ───────────────────────
class FusionMLPNet(nn.Module):
    """
    Exact PyTorch equivalent of Scikit-Learn MLPClassifier(hidden_layer_sizes=(256, 128, 64), activation='relu').
    Topology: 1314 -> 256 -> 128 -> 64 -> 1
    Total Trainable Parameters: 377,857
    """
    def __init__(self, input_dim: int = 1314):
        super(FusionMLPNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 64)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        logits = self.fc4(x)
        return logits.squeeze(-1)

    def predict_proba(self, x: torch.Tensor) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs

    def get_parameter_dict(self) -> dict:
        """Returns deep copy of state dict as numpy arrays."""
        return {k: v.cpu().numpy().copy() for k, v in self.state_dict().items()}

    def load_parameter_dict(self, param_dict: dict):
        """Loads state dict from numpy arrays."""
        state = {k: torch.tensor(v, dtype=torch.float32) for k, v in param_dict.items()}
        self.load_state_dict(state)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def init_from_sklearn_weights(sklearn_mlp) -> FusionMLPNet:
    """Initializes FusionMLPNet with weights from a trained Scikit-Learn MLPClassifier."""
    net = FusionMLPNet(input_dim=1314)
    state = net.state_dict()
    
    # Layer 1: sklearn coefs_[0] is [1314, 256], PyTorch fc1.weight is [256, 1314]
    state['fc1.weight'] = torch.tensor(sklearn_mlp.coefs_[0].T, dtype=torch.float32)
    state['fc1.bias'] = torch.tensor(sklearn_mlp.intercepts_[0], dtype=torch.float32)
    
    # Layer 2: [256, 128] -> [128, 256]
    state['fc2.weight'] = torch.tensor(sklearn_mlp.coefs_[1].T, dtype=torch.float32)
    state['fc2.bias'] = torch.tensor(sklearn_mlp.intercepts_[1], dtype=torch.float32)
    
    # Layer 3: [128, 64] -> [64, 128]
    state['fc3.weight'] = torch.tensor(sklearn_mlp.coefs_[2].T, dtype=torch.float32)
    state['fc3.bias'] = torch.tensor(sklearn_mlp.intercepts_[2], dtype=torch.float32)
    
    # Layer 4: [64, 1] -> [1, 64]
    state['fc4.weight'] = torch.tensor(sklearn_mlp.coefs_[3].T, dtype=torch.float32)
    state['fc4.bias'] = torch.tensor(sklearn_mlp.intercepts_[3], dtype=torch.float32)
    
    net.load_state_dict(state)
    return net


# ── 2. Data Loading & Feature Building ────────────────────────────────────────
def load_dataset():
    """
    Loads all features aligned to manifest, splits into 80/20 train/test,
    and applies canonical StandardScaler normalization + weighting.
    """
    manifest = pd.read_csv(MANIFEST_PATH).sort_values("sample_id").reset_index(drop=True)
    y = manifest["label"].values.astype(np.float32)

    # URL features (22)
    url_df = pd.read_csv(os.path.join(FEAT_DIR, "url_features.csv")).sort_values("sample_id").reset_index(drop=True)
    feature_cols = [c for c in CANONICAL_URL_FEATURE_NAMES if c in url_df.columns]
    X_url = url_df[feature_cols].values.astype(np.float64)

    # HTML features (12)
    html_df = pd.read_csv(os.path.join(FEAT_DIR, "html_features.csv")).sort_values("sample_id").reset_index(drop=True)
    X_html = html_df.drop(columns=["sample_id"]).values.astype(np.float64)

    # Visual embeddings (1280)
    X_visual = np.load(os.path.join(FEAT_DIR, "visual_embeddings.npy")).astype(np.float64)

    indices = np.arange(len(y))
    idx_train, idx_test, y_train, y_test = train_test_split(
        indices, y, test_size=0.20, stratify=y, random_state=42
    )

    train_parts = []
    test_parts = []
    for X, w in [(X_url, WEIGHTS["url"]), (X_html, WEIGHTS["html"]), (X_visual, WEIGHTS["visual"])]:
        sc = StandardScaler()
        X_tr = sc.fit_transform(X[idx_train]) * w
        X_te = sc.transform(X[idx_test]) * w
        train_parts.append(X_tr)
        test_parts.append(X_te)

    X_train_fused = np.hstack(train_parts).astype(np.float32)
    X_test_fused = np.hstack(test_parts).astype(np.float32)

    return X_train_fused, y_train, X_test_fused, y_test, idx_train, idx_test


# ── 3. Client Partitioning (IID vs Label-Skew Non-IID) ────────────────────────
def partition_data_iid(X_train: np.ndarray, y_train: np.ndarray, num_clients: int = 5, seed: int = 42) -> list:
    """
    IID Partitioning:
    Evenly distributes the 792 training samples across `num_clients` using a deterministic shuffle.
    """
    rng = np.random.RandomState(seed)
    n_samples = len(y_train)
    shuffled_idx = rng.permutation(n_samples)

    client_splits = np.array_split(shuffled_idx, num_clients)
    partitions = []
    for i, split in enumerate(client_splits):
        partitions.append({
            "client_id": f"client_{i}",
            "indices": split.tolist(),
            "X": X_train[split],
            "y": y_train[split],
            "num_samples": len(split),
            "num_phish": int(np.sum(y_train[split] == 1)),
            "num_legit": int(np.sum(y_train[split] == 0)),
            "phish_ratio": float(np.mean(y_train[split] == 1))
        })
    return partitions


def partition_data_non_iid(X_train: np.ndarray, y_train: np.ndarray, num_clients: int = 5, seed: int = 42) -> list:
    """
    Non-IID (Label-Skew) Partitioning:
    Partitions the 792 training samples (520 Phish, 272 Legit) with heterogeneous class proportions across clients:
      - Client 0: 180 Phish, 10 Legit (94.7% Phish)
      - Client 1: 150 Phish, 25 Legit (85.7% Phish)
      - Client 2: 100 Phish, 40 Legit (71.4% Phish)
      - Client 3: 60 Phish, 97 Legit  (38.2% Phish)
      - Client 4: 30 Phish, 100 Legit (23.1% Phish)
    Total: 520 Phish, 272 Legit. Zero duplicates. Zero holdout leakage.
    """
    rng = np.random.RandomState(seed)
    phish_idx = np.where(y_train == 1)[0]
    legit_idx = np.where(y_train == 0)[0]

    assert len(phish_idx) == 520, f"Expected 520 phishing samples, got {len(phish_idx)}"
    assert len(legit_idx) == 272, f"Expected 272 legitimate samples, got {len(legit_idx)}"

    phish_shuffled = rng.permutation(phish_idx)
    legit_shuffled = rng.permutation(legit_idx)

    # Specific partition counts for 5 clients
    phish_counts = [180, 150, 100, 60, 30]
    legit_counts = [10, 25, 40, 97, 100]
    assert sum(phish_counts) == 520
    assert sum(legit_counts) == 272

    partitions = []
    p_offset = 0
    l_offset = 0

    for i in range(num_clients):
        p_c = phish_counts[i]
        l_c = legit_counts[i]

        p_slice = phish_shuffled[p_offset : p_offset + p_c]
        l_slice = legit_shuffled[l_offset : l_offset + l_c]
        p_offset += p_c
        l_offset += l_c

        c_indices = np.concatenate([p_slice, l_slice])
        c_indices = rng.permutation(c_indices) # local shuffle

        partitions.append({
            "client_id": f"client_{i}",
            "indices": c_indices.tolist(),
            "X": X_train[c_indices],
            "y": y_train[c_indices],
            "num_samples": len(c_indices),
            "num_phish": int(p_c),
            "num_legit": int(l_c),
            "phish_ratio": float(p_c / (p_c + l_c))
        })

    return partitions


# ── 4. Local Client Training Routine ──────────────────────────────────────────
def train_client_local(
    global_model: FusionMLPNet,
    client_data: dict,
    lr: float = 0.001,
    local_epochs: int = 2,
    batch_size: int = 32,
    seed: int = 42
) -> dict:
    """
    Deterministic local client training:
    1. Copies global model W_t.
    2. Evaluates local loss before training.
    3. Executes local_epochs of mini-batch gradient descent (Adam/BCE).
    4. Evaluates local loss after training.
    5. Computes parameter delta: ΔW_i = W_i - W_t.
    6. Returns clean FL update contract payload (0 raw data).
    """
    torch.manual_seed(seed)
    local_net = copy.deepcopy(global_model)
    local_net.train()

    X_t = torch.tensor(client_data["X"], dtype=torch.float32)
    y_t = torch.tensor(client_data["y"], dtype=torch.float32)
    dataset = TensorDataset(X_t, y_t)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(local_net.parameters(), lr=lr)

    # Initial local loss
    local_net.eval()
    with torch.no_grad():
        initial_logits = local_net(X_t)
        loss_before = criterion(initial_logits, y_t).item()

    # Local training loop
    local_net.train()
    for _ in range(local_epochs):
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            logits = local_net(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

    # Final local loss
    local_net.eval()
    with torch.no_grad():
        final_logits = local_net(X_t)
        loss_after = criterion(final_logits, y_t).item()

    # Compute parameter deltas ΔW_i = W_i - W_t
    w_global = global_model.get_parameter_dict()
    w_local = local_net.get_parameter_dict()

    delta_weights = {}
    delta_sq_sum = 0.0
    for k in w_global.keys():
        d = w_local[k] - w_global[k]
        delta_weights[k] = d
        delta_sq_sum += np.sum(d ** 2)

    delta_norm = float(np.sqrt(delta_sq_sum))

    return {
        "client_id": client_data["client_id"],
        "num_examples": client_data["num_samples"],
        "local_loss_before": round(float(loss_before), 5),
        "local_loss_after": round(float(loss_after), 5),
        "delta_norm": round(float(delta_norm), 6),
        "delta_weights": delta_weights
    }


# ── 5. Standard Federated Averaging (FedAvg) ──────────────────────────────────
def aggregate_fedavg(global_model: FusionMLPNet, client_updates: list) -> FusionMLPNet:
    """
    Standard FedAvg Aggregator:
    W_(t+1) = W_t + Σ (n_i / Σ n_j) ΔW_i
    """
    total_samples = sum(u["num_examples"] for u in client_updates)
    base_params = global_model.get_parameter_dict()
    new_params = copy.deepcopy(base_params)

    for k in base_params.keys():
        weighted_delta = np.zeros_like(base_params[k])
        for u in client_updates:
            weight = u["num_examples"] / total_samples
            weighted_delta += weight * u["delta_weights"][k]
        new_params[k] = base_params[k] + weighted_delta

    updated_global_model = copy.deepcopy(global_model)
    updated_global_model.load_parameter_dict(new_params)
    return updated_global_model


# ── 6. Evaluation on Held-Out Test Set ─────────────────────────────────────────
def evaluate_global_model(model: FusionMLPNet, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Evaluates global model against the static holdout test set (198 samples).
    """
    model.eval()
    X_t = torch.tensor(X_test, dtype=torch.float32)
    y_t = torch.tensor(y_test, dtype=torch.float32)

    criterion = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        logits = model(X_t)
        loss = criterion(logits, y_t).item()
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    auc = roc_auc_score(y_test, probs)

    return {
        "loss": round(float(loss), 5),
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "auc": round(float(auc), 4)
    }


# ── 7. Simulation Orchestrator ────────────────────────────────────────────────
def run_simulation(
    partition_mode: str = "iid",
    num_clients: int = 5,
    num_rounds: int = 3,
    local_epochs: int = 2,
    lr: float = 0.001,
    seed: int = 42,
    artifact_dir: str = DEFAULT_ARTIFACT_DIR
) -> dict:
    """
    Runs a deterministic multi-round FL simulation.
    """
    os.makedirs(artifact_dir, exist_ok=True)
    
    # 1. Load data
    X_train, y_train, X_test, y_test, _, _ = load_dataset()

    # 2. Partition
    if partition_mode == "iid":
        partitions = partition_data_iid(X_train, y_train, num_clients=num_clients, seed=seed)
    elif partition_mode == "non_iid":
        partitions = partition_data_non_iid(X_train, y_train, num_clients=num_clients, seed=seed)
    else:
        raise ValueError(f"Unknown partition_mode: {partition_mode}")

    # 3. Initialize Global Model from production baseline weights
    fusion_bundle = joblib.load(FUSION_MODEL_PATH)
    global_model = init_from_sklearn_weights(fusion_bundle["model"])

    # Initial evaluation (Round 0)
    initial_metrics = evaluate_global_model(global_model, X_test, y_test)

    rounds_history = []
    
    # FL Rounds Loop
    for r in range(1, num_rounds + 1):
        client_updates = []
        client_metrics_summary = []

        for c_data in partitions:
            # Deterministic client seed
            client_seed = seed + (r * 100) + int(c_data["client_id"].split("_")[1])
            update = train_client_local(
                global_model=global_model,
                client_data=c_data,
                lr=lr,
                local_epochs=local_epochs,
                batch_size=32,
                seed=client_seed
            )
            client_updates.append(update)
            client_metrics_summary.append({
                "client_id": update["client_id"],
                "num_examples": update["num_examples"],
                "local_loss_before": update["local_loss_before"],
                "local_loss_after": update["local_loss_after"],
                "delta_norm": update["delta_norm"]
            })

        # Aggregation (FedAvg)
        global_model = aggregate_fedavg(global_model, client_updates)

        # Evaluate aggregated model on holdout set
        global_metrics = evaluate_global_model(global_model, X_test, y_test)

        rounds_history.append({
            "round_id": r,
            "clients": client_metrics_summary,
            "mean_delta_norm": round(float(np.mean([u["delta_norm"] for u in client_updates])), 6),
            "global_metrics": global_metrics
        })

    result_payload = {
        "simulation_mode": partition_mode,
        "config": {
            "num_clients": num_clients,
            "num_rounds": num_rounds,
            "local_epochs": local_epochs,
            "learning_rate": lr,
            "seed": seed,
            "model_architecture": "FusionMLPNet(1314->256->128->64->1)",
            "total_parameters": global_model.count_parameters()
        },
        "client_partitions": [
            {
                "client_id": p["client_id"],
                "num_samples": p["num_samples"],
                "num_phish": p["num_phish"],
                "num_legit": p["num_legit"],
                "phish_ratio": round(p["phish_ratio"], 4)
            } for p in partitions
        ],
        "initial_metrics_round_0": initial_metrics,
        "rounds": rounds_history,
        "final_metrics": rounds_history[-1]["global_metrics"]
    }

    output_path = os.path.join(artifact_dir, f"{partition_mode}_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_payload, f, indent=2)

    return result_payload


def run_all_experiments():
    """Runs both IID and Non-IID experiments and saves config and results."""
    os.makedirs(DEFAULT_ARTIFACT_DIR, exist_ok=True)
    
    config = {
        "experiment_name": "FEDrA_Step9_1_Local_Simulation",
        "timestamp": "2026-10-08",
        "num_clients": 5,
        "num_rounds": 3,
        "local_epochs": 2,
        "learning_rate": 0.001,
        "seed": 42,
        "model": "Fusion MLP (1314 dims, 377,857 params)",
        "dataset": "990 samples (792 train, 198 holdout)"
    }
    with open(os.path.join(DEFAULT_ARTIFACT_DIR, "simulation_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("\n" + "=" * 65)
    print("  RUNNING EXPERIMENT A: IID (5 clients, 3 rounds, seed=42)")
    print("=" * 65)
    iid_res = run_simulation(partition_mode="iid", seed=42)
    print(f"  Round 0 (Baseline) -> Acc: {iid_res['initial_metrics_round_0']['accuracy']:.4f} | Loss: {iid_res['initial_metrics_round_0']['loss']:.4f} | AUC: {iid_res['initial_metrics_round_0']['auc']:.4f}")
    for r in iid_res["rounds"]:
        m = r["global_metrics"]
        print(f"  Round {r['round_id']} (FedAvg)     -> Acc: {m['accuracy']:.4f} | Loss: {m['loss']:.4f} | Prec: {m['precision']:.4f} | Rec: {m['recall']:.4f} | AUC: {m['auc']:.4f} | DeltaNorm: {r['mean_delta_norm']:.5f}")

    print("\n" + "=" * 65)
    print("  RUNNING EXPERIMENT B: Non-IID Label-Skew (5 clients, 3 rounds, seed=42)")
    print("=" * 65)
    non_iid_res = run_simulation(partition_mode="non_iid", seed=42)
    print(f"  Round 0 (Baseline) -> Acc: {non_iid_res['initial_metrics_round_0']['accuracy']:.4f} | Loss: {non_iid_res['initial_metrics_round_0']['loss']:.4f} | AUC: {non_iid_res['initial_metrics_round_0']['auc']:.4f}")
    for r in non_iid_res["rounds"]:
        m = r["global_metrics"]
        print(f"  Round {r['round_id']} (FedAvg)     -> Acc: {m['accuracy']:.4f} | Loss: {m['loss']:.4f} | Prec: {m['precision']:.4f} | Rec: {m['recall']:.4f} | AUC: {m['auc']:.4f} | DeltaNorm: {r['mean_delta_norm']:.5f}")

    return iid_res, non_iid_res


if __name__ == "__main__":
    run_all_experiments()
