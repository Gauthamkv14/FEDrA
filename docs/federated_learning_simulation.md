# FEDrA Step 9.1: Deterministic Local Federated Learning Simulation

## 1. Overview & Strict Scope Boundary

> [!IMPORTANT]
> **SIMULATION & PROTOTYPE ONLY**
>
> This document details the **Step 9.1 Local Simulation** for Federated Learning (FL) in the FEDrA framework.
> 
> - **In-Scope**: Standalone Python simulation validating local client optimization, gradient/weight delta calculation, standard Federated Averaging (FedAvg), optimization behavior under IID and Non-IID distributions, evaluation against a static held-out test set, and verification of zero raw-data leakage in FL payloads.
> - **Strictly Out-of-Scope**: Zero browser-side JavaScript training, zero Web Workers / WebGPU training, zero client-server networking or HTTP endpoints, zero Secure Aggregation / Diffie-Hellman masking, zero Differential Privacy noise injection, zero dynamic model downloading or hot-swapping into the Chrome extension, and zero collection of real user browsing data.
> - **Production Model Status**: All production inference models (`models/fusion_model.pkl`, `models/onnx/fusion_model.onnx`, and `extension/models/`) remain **100% frozen and unmodified**.

---

## 2. Trainable Architecture: Fusion MLP

The simulation targets FEDrA's **Multimodal Fusion MLP**, which combines normalized features across URL, HTML, and visual modalities.

### Architecture Topology & Weight Transfer (Case A: Production Baseline Initialization)
- **Initialization Source**: Directly transferred from the trained Scikit-Learn `MLPClassifier` in `models/fusion_model.pkl` via `init_from_sklearn_weights()`.
- **Parameter Transfer**:
  - `coefs_[0]` ($1314 \times 256$) $\rightarrow$ `fc1.weight` ($256 \times 1314$)
  - `intercepts_[0]` ($256$) $\rightarrow$ `fc1.bias` ($256$)
  - `coefs_[1]` ($256 \times 128$) $\rightarrow$ `fc2.weight` ($128 \times 256$)
  - `intercepts_[1]` ($128$) $\rightarrow$ `fc2.bias` ($128$)
  - `coefs_[2]` ($128 \times 64$) $\rightarrow$ `fc3.weight` ($64 \times 128$)
  - `intercepts_[2]` ($64$) $\rightarrow$ `fc3.bias` ($64$)
  - `coefs_[3]` ($64 \times 1$) $\rightarrow$ `fc4.weight` ($1 \times 64$)
  - `intercepts_[3]` ($1$) $\rightarrow$ `fc4.bias` ($1$)
- **Parity Verification**: Max probability difference between Scikit-Learn `MLPClassifier` and PyTorch `FusionMLPNet` on the holdout test set is $2.37 \times 10^{-7}$. Round 0 precisely reproduces the existing trained model baseline (Acc: `0.8333`, Loss: `0.4285`, Prec: `0.8819`, Rec: `0.8615`, F1: `0.8716`, AUC: `0.9233`).
- **Input Dimension**: $1314$
  - URL features: $22$ (Weight: $0.4$)
  - HTML features: $12$ (Weight: $0.3$)
  - Visual MobileNetV2 embeddings: $1280$ (Weight: $0.3$)
- **Hidden Layers**:
  - `Dense(1314 -> 256)` + `ReLU` ($1314 \times 256 + 256 = 336,640$ params)
  - `Dense(256 -> 128)` + `ReLU` ($256 \times 128 + 128 = 32,896$ params)
  - `Dense(128 -> 64)` + `ReLU` ($128 \times 64 + 64 = 8,256$ params)
  - `Dense(64 -> 1)` ($64 \times 1 + 1 = 65$ params)
- **Total Trainable Parameters**: **377,857**
- **Activation / Loss**:
  - Training: Raw logit output paired with `BCEWithLogitsLoss`.
  - Inference: Sigmoid activation $\sigma(z) = \frac{1}{1 + e^{-z}}$ producing $P(\text{Phishing})$.

---

## 3. Dataset Partitioning & Experimental Regimes

The FEDrA canonical dataset (990 labeled samples from `Dataset/manifest.csv`) is partitioned with an 80/20 stratified split (`random_state=42`):
- **Training Set**: $792$ samples ($520$ Phishing, $272$ Legitimate; $65.65\%$ Phishing ratio).
- **Held-Out Test Set**: $198$ samples ($130$ Phishing, $68$ Legitimate; $65.65\%$ Phishing ratio).

### Regime A: IID (Independent & Identically Distributed)
The $792$ training samples are uniformly distributed across 5 simulated clients:
- **Client 0**: 159 samples ($107$ Phish, $52$ Legit — $67.3\%$ Phish)
- **Client 1**: 159 samples ($93$ Phish, $66$ Legit — $58.5\%$ Phish)
- **Client 2**: 158 samples ($109$ Phish, $49$ Legit — $69.0\%$ Phish)
- **Client 3**: 158 samples ($110$ Phish, $48$ Legit — $69.6\%$ Phish)
- **Client 4**: 158 samples ($101$ Phish, $57$ Legit — $63.9\%$ Phish)
- **Total**: $792$ samples. Zero overlap. Uniform balance.

### Regime B: Non-IID (Deliberate Label Skew)
Simulates heterogeneous user browsing behavior with severe class imbalance across 5 clients:
- **Client 0 (High Phishing Encounter)**: $180$ Phish + $10$ Legit = $190$ samples ($94.7\%$ Phish)
- **Client 1 (High Phishing Skew)**: $150$ Phish + $25$ Legit = $175$ samples ($85.7\%$ Phish)
- **Client 2 (Moderate Phishing)**: $100$ Phish + $40$ Legit = $140$ samples ($71.4\%$ Phish)
- **Client 3 (Legitimate Skew)**: $60$ Phish + $97$ Legit = $157$ samples ($38.2\%$ Phish)
- **Client 4 (High Legitimate Encounter)**: $30$ Phish + $100$ Legit = $130$ samples ($23.1\%$ Phish)
- **Total**: $520$ Phish, $272$ Legit = $792$ samples. Zero duplicate assignments, zero test set leakage.

---

## 4. Local Training & FedAvg Aggregation Formulation

### Local Client Optimization
Each participating client $i$ receives the current global model weights $W_t$. The client trains locally for $E=2$ epochs using mini-batch Adam optimization ($\text{lr}=0.001$, batch size $B=32$):

$$\mathcal{L}_{\text{BCE}}(z, y) = - \left[ y \log \sigma(z) + (1 - y) \log (1 - \sigma(z)) \right]$$

Upon completion of local optimization, the client computes parameter updates:

$$\Delta W_i = W_i^{(E)} - W_t$$

The Euclidean norm $\|\Delta W_i\|_2 = \sqrt{\sum (\Delta W_i)^2}$ is computed to monitor parameter update magnitudes.

### Federated Averaging (FedAvg)
The central aggregator computes the new global parameter vector $W_{t+1}$ using sample-weighted averaging:

$$W_{t+1} = W_t + \sum_{i=1}^{K} \frac{n_i}{N_{\text{total}}} \Delta W_i \quad \text{where } N_{\text{total}} = \sum_{j=1}^{K} n_j$$

*Simulated transmission*: In this local simulation, only parameter delta tensors $\Delta W_i$ and sample counts $n_i$ are transferred to the aggregator; raw feature matrices remain strictly local to client dictionaries.

---

## 5. Simulation Results Summary

> [!NOTE]
> **Performance Interpretation**
> Both IID and Non-IID experiments produced measurable changes in global performance over three FedAvg rounds, with final AUC exceeding the initial experimental baseline ($0.9233 \rightarrow 0.9325$ on IID, $0.9233 \rightarrow 0.9328$ on Non-IID). Three rounds are insufficient to establish formal convergence.

### Experiment A: IID Benchmark (5 Clients, 3 Rounds, Seed=42)

| Round | Global Loss | Global Accuracy | Global Precision | Global Recall | Global F1 | Global AUC | Mean Delta Norm |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0 (Baseline)** | 0.4285 | 83.33% | 0.8819 | 0.8615 | 0.8716 | 0.9233 | — |
| **1** | 0.4066 | 83.33% | 0.8880 | 0.8538 | 0.8706 | 0.9308 | 1.87576 |
| **2** | 0.4585 | 83.84% | 0.8769 | 0.8769 | 0.8769 | 0.9300 | 1.85010 |
| **3** | 0.4724 | 83.84% | 0.8952 | 0.8538 | 0.8740 | 0.9325 | 1.78726 |

### Experiment B: Non-IID Label-Skew (5 Clients, 3 Rounds, Seed=42)

| Round | Global Loss | Global Accuracy | Global Precision | Global Recall | Global F1 | Global AUC | Mean Delta Norm |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0 (Baseline)** | 0.4285 | 83.33% | 0.8819 | 0.8615 | 0.8716 | 0.9233 | — |
| **1** | 0.4426 | 84.34% | 0.9024 | 0.8538 | 0.8775 | 0.9181 | 1.93756 |
| **2** | 0.4291 | 83.84% | 0.8712 | 0.8846 | 0.8779 | 0.9256 | 1.98432 |
| **3** | 0.4411 | 86.87% | 0.8662 | 0.9462 | 0.9044 | 0.9328 | 1.96094 |

---

## 6. Privacy & Data Leakage Verification

All outputs stored in `artifacts/federated/` (`simulation_config.json`, `iid_results.json`, `non_iid_results.json`) were strictly audited:
1. **Zero Raw URLs**: Scanned and verified no URLs or domain names exist in payloads.
2. **Zero HTML / DOM Trees**: Scanned and verified no markup or text content exists in payloads.
3. **Zero Visual Embeddings / Screenshots**: Scanned and verified no raw feature vectors or image bytes exist in payloads.
4. **Permitted Payload Schema**:
   ```json
   {
     "client_id": "client_0",
     "num_examples": 159,
     "local_loss_before": 0.03931,
     "local_loss_after": 0.02227,
     "delta_norm": 2.032386
   }
   ```

---

## 7. What This Simulation Does NOT Prove

To maintain strict academic and engineering rigor, we explicitly declare the limitations of this simulation:
1. **No Browser Training Feasibility**: Does not evaluate JavaScript training performance, WebAssembly execution speed, WebGPU compute availability, or memory constraints inside Chrome extension workers.
2. **No Network Robustness**: Does not evaluate socket timeouts, client dropouts, asymmetric upload/download bandwidth, or server-side load under thousands of concurrent clients.
3. **No Cryptographic Security Guarantee**: Does not apply real Diffie-Hellman SecAgg encryption masks or prove unobservability against an untrusted aggregation server.
4. **No Formal Differential Privacy**: Does not add $(\epsilon, \delta)$-DP Gaussian mechanisms or bound privacy budgets.
5. **No Production Hot-Swapping**: Does not implement cryptographic model signing or dynamic runtime model replacement in browser extensions.
