# FEDrA — Federated Learning Architecture Specification

**Status:** DESIGNED (Conceptual Architecture, Protocols, and Threat Models — Implementation Planned for Step 9+)  
**Author:** FEDrA Engineering & Research Team  
**Date:** 2026-10-08  
**Scope:** Architecture, Update Contracts, Privacy Boundaries, Threat Model, and Model Promotion Pipeline for Client-Collaborative Phishing Detection.

---

## 1. Executive Summary & Core Philosophy

FEDrA (*Federated Detection of Ransomware & Phishing*) is designed from the ground up to eliminate the fundamental privacy flaw of traditional web security scanners: **transmitting user browsing histories, visited URLs, full DOM contents, and screenshots to a central cloud server**.

In Phase 1 (Steps 1–7), FEDrA successfully established an authoritative, browser-local multimodal inference pipeline using WebAssembly and ONNX Runtime:
```text
Browser Tab (Active Navigation)
       │
       ▼
[ Browser-Native Feature Extraction ]
  ├── URL Lexical Vector (22 dims)
  ├── HTML DOM Vector (12 dims)
  └── Screenshot Tensor (1 × 3 × 224 × 224)
       │
       ▼
[ Local In-Browser ONNX Inference ]
  ├── url_baseline.onnx (22 dims)
  ├── html_baseline.onnx (12 dims)
  ├── mobilenet_v2_visual.onnx (1280 dims + 7×7 spatial map)
  ├── image_baseline.onnx (1280 dims)
  └── fusion_model.onnx (1314 dims)
       │
       ▼
[ Local Explainability & Verdict ]
  ├── Exact Linear Feature Attribution (c_i = w_i · z_i)
  ├── Analytical Visual Grad-CAM (alpha_k = w_k / (49 · sigma_k))
  ├── Multimodal Explanation Fusion (schema_version: "1.0")
  └── Extension Popup UI
```

**The Step 8 Goal:** Design the Federated Learning (FL) architecture that enables client devices to collaboratively train and improve the detection models against emerging zero-day phishing campaigns **without ever transmitting raw URLs, HTML source, screenshots, or browsing activity to any centralized infrastructure**.

---

## 2. In-Depth Current Model Architecture Audit

Before defining the federation mechanics, we audit the exact trained model artifacts, tensor dimensions, parameter counts, and mathematical formulations established across Steps 1–7:

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   FEDrA MODEL INVENTORY (OPSET 17)                                     │
├──────────────────────┬─────────────┬───────────┬──────────────┬───────────────┬────────────────────────┤
│ Component            │ Model Class │ Input Dim │ Output Interf│ Total Params  │ Artifact Size & Format │
├──────────────────────┼─────────────┼───────────┼──────────────┼───────────────┼────────────────────────┤
│ URL Baseline         │ LogReg (bal)│ 22        │ 2-class prob │ 23 params     │ 866 B (ONNX) / 2.4 KB  │
│ HTML Baseline        │ LogReg (bal)│ 12        │ 2-class prob │ 13 params     │ 666 B (ONNX) / 1.7 KB  │
│ Visual Extractor     │ MobileNetV2 │ 3×224×224 │ 1280 + 7×7   │ 2,223,872     │ 8.86 MB (ONNX opset 17)│
│ Image Baseline       │ LogReg (bal)│ 1280      │ 2-class prob │ 1,281 params  │ 26.0 KB (ONNX) / 42 KB │
│ Fusion MLP           │ 4-layer MLP │ 1314      │ 2-class prob │ 377,857 params│ 1.51 MB (ONNX) / 9.1 MB│
└──────────────────────┴─────────────┴───────────┴──────────────┴───────────────┴────────────────────────┘
```

### Authoritative Architecture & Output Dimension Resolution:
To eliminate any ambiguity between the internal network topology and the inference runtime interface:
1. **Internal Network Topology (`MLPClassifier`):**
   - Binary classifier configured with `hidden_layer_sizes=(256, 128, 64)`, `activation="relu"`, and single-unit sigmoid output (`n_outputs_ = 1`, `out_activation_ = "logistic"`).
   - Layer 1: Weight $[1314, 256]$ (336,384 params) + Bias $[256]$ (256 params) = 336,640 params.
   - Layer 2: Weight $[256, 128]$ (32,768 params) + Bias $[128]$ (128 params) = 32,896 params.
   - Layer 3: Weight $[128, 64]$ (8,192 params) + Bias $[64]$ (64 params) = 8,256 params.
   - Output Layer: Weight $[64, 1]$ (64 params) + Bias $[1]$ (1 param) = 65 params.
   - **Total Trainable Network Parameters:** $336,640 + 32,896 + 8,256 + 65 = 377,857$ parameters.
2. **Inference Runtime Output Contract (`fusion_model.onnx` / Scikit-Learn API):**
   - In accordance with Scikit-Learn binary classification and standard ONNX runtime export, the model produces two output nodes:
     - `label`: Predicted class index ($0$ or $1$, shape $[1]$, `int64`).
     - `probabilities`: Normalized 2-class probability distribution $[P(\text{Legitimate}), P(\text{Phishing})] = [1 - \sigma(z), \sigma(z)]$ (shape $[1, 2]$, `float32`).

---

## 3. Federation Unit Decision & Feasibility Matrix

We systematically analyze which model components are appropriate for federated training:

| Component | Architecture & Dims | Federation Feasibility | Update Payload Size | Client Compute Profile | Privacy Risk Profile | Architectural Decision |
|---|---|---|---|---|---|---|
| **Fusion MLP** | 4-layer MLP (1314 inputs, 377K params) | **Feasible (Primary)** | ~1.51 MB (dense) / ~380 KB (quantized) | Moderate feedforward backprop | Low-to-Moderate (multimodal embeddings) | **YES — Primary Multimodal FL Target (DESIGNED)** |
| **URL Baseline** | Logistic Regression (22 dims) | **High (Lightweight)** | 92 B | Negligible linear SGD | Minimal (22 weights) | **YES — Tier 1 Lightweight Target (DESIGNED)** |
| **HTML Baseline** | Logistic Regression (12 dims) | **High (Lightweight)** | 52 B | Negligible linear SGD | Minimal (12 weights) | **YES — Tier 1 Lightweight Target (DESIGNED)** |
| **Image Baseline** | Logistic Regression (1280 dims) | **Moderate** | 5.1 KB | Low linear SGD | Low | **YES — Tier 2 Target (DESIGNED)** |
| **MobileNetV2** | Deep ConvNet (53 layers, 2.22M params) | **Not Feasible for Client FL** | 8.86 MB | High (>2s conv autograd in browser) | High (visual feature inversion) | **NO — Frozen Pretrained Backbone (NOT FEDERATED)** |

### Rationale for Freezing MobileNetV2:
1. **Compute & Battery Impact:** Running autograd/backpropagation through 53 convolutional and depthwise-separable layers in client browsers leads to tab lag, high RAM allocation (~150 MB), and severe CPU utilization.
2. **Bandwidth Overhead:** Transmitting 8.86 MB per client per round across mobile/broadband connections is bandwidth-prohibitive.
3. **Representational Role:** MobileNetV2 acts purely as a generic visual representation extractor trained on millions of diverse images. Phishing-specific decision boundaries are learned in the downstream **Image Baseline** (5.1 KB) and **Fusion MLP** (1.51 MB).

---

## 4. Strict Decoupling: Inference vs Training Paths

```text
══════════════════════════════════════════════════════════════════════════════════
[SYNCHRONOUS REAL-TIME INFERENCE PATH — 100% LOCAL & OFFLINE]
══════════════════════════════════════════════════════════════════════════════════
User visits page ──► content.js ──► background.js ──► ONNX Inference ──► Popup UI
  • Real-time execution: < 60 ms
  • Network requests: 0 (Zero external calls)
  • External dependencies: None (Completely functional offline)
  • FL status: Passive consumer of local model artifacts

══════════════════════════════════════════════════════════════════════════════════
[ASYNCHRONOUS FEDERATED LEARNING PATH — IDLE BACKGROUND PROTOCOL (DESIGNED)]
══════════════════════════════════════════════════════════════════════════════════
Local label buffer (user feedback / verified encounters)
       │
       ▼ (Only when device is Idle + AC Power + Unmetered Wi-Fi)
Local Forward-Backward Pass (Python Simulation in Step 9; Web Worker in Step 10+)
       │
       ▼
Compute Parameter Deltas: ΔW = W_local - W_global
       │
       ▼
L2 Norm Gradient Clipping & Privacy Safeguards
       │
       ▼ (Encrypted update payload)
FL Coordinator (Aggregation Gate ──► Holdout Validation ──► Promotion Pipeline)
```

**Non-Negotiable Rule:** The browser extension **NEVER** contacts the FL Coordinator during page classification. Phishing detection latency is completely decoupled from federated training rounds.

---

## 5. Client Data Boundary & Privacy Contract

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT PRIVACY BOUNDARY                                   │
├────────────────────────────────────────────────────────┬───────────────────────────────┤
│ DATA THAT STRICTLY NEVER LEAVES CLIENT DEVICE          │ DATA PERMITTED IN FL CONTRACT │
├────────────────────────────────────────────────────────┼───────────────────────────────┤
│ ❌ Visited URLs, full query params, domain names       │ ✅ Model weight deltas (ΔW)   │
│ ❌ Raw HTML source code, DOM structure, script bodies  │ ✅ Protocol & Model version IDs│
│ ❌ Page screenshots, render canvases, favicon bitmaps   │ ✅ Aggregation round identifier│
│ ❌ Browsing history, navigation timestamps             │ ✅ Bucketed sample count      │
│ ❌ User cookie stores, auth tokens, IP address         │                               │
│ ❌ User association with specific phishing incidents   │                               │
└────────────────────────────────────────────────────────┴───────────────────────────────┘
```

---

## 6. Client FL Update JSON Contract Specification & Field Classification

In accordance with data minimization principles, every transmitted field is strictly classified:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "FEDrA_Client_FL_Update_Contract",
  "type": "object",
  "required": [
    "protocol_version",
    "model_id",
    "architecture_version",
    "base_model_version",
    "training_round_id",
    "update_type",
    "num_examples_bucket",
    "parameter_delta"
  ],
  "properties": {
    "protocol_version": { "type": "string", "enum": ["1.0"] },
    "model_id": { 
      "type": "string", 
      "enum": ["fedra-fusion-mlp", "fedra-url-baseline", "fedra-html-baseline", "fedra-image-baseline"] 
    },
    "architecture_version": { "type": "string", "enum": ["1.0"] },
    "base_model_version": { "type": "string", "pattern": "^v[0-9]+\\.[0-9]+\\.[0-9]+$" },
    "training_round_id": { "type": "integer", "minimum": 1 },
    "update_type": { 
      "type": "string", 
      "enum": ["model_delta_dense", "model_delta_quantized_int8", "model_delta_topk_sparse"] 
    },
    "num_examples_bucket": { 
      "type": "string", 
      "enum": ["1-10", "11-50", "51-100", "101-200", "200+"],
      "description": "Bucketed sample size to prevent exact browsing volume fingerprinting"
    },
    "parameter_delta": {
      "type": "object",
      "required": ["format", "layer_deltas"],
      "properties": {
        "format": { "type": "string", "enum": ["float32_base64", "bfloat16_base64"] },
        "layer_deltas": {
          "type": "object",
          "description": "Key-value dictionary mapping layer names to delta weight vectors"
        }
      }
    },
    "local_validation_summary": {
      "type": "object",
      "description": "Optional client-side loss reduction flag for self-filtering free-riders",
      "properties": {
        "loss_decreased": { "type": "boolean" }
      }
    },
    "round_enrollment_token": {
      "type": "string",
      "description": "Ephemeral single-round blinded token for Sybil mitigation; discarded immediately upon round completion. Does NOT persist or track clients."
    }
  },
  "additionalProperties": false
}
```

### Field Classification Matrix:

| Field Name | Classification | Purpose & Privacy Lifecycle |
|---|---|---|
| `protocol_version` | **REQUIRED** | Ensures wire format compatibility. |
| `model_id` | **REQUIRED** | Identifies target model component. |
| `architecture_version` | **REQUIRED** | Prevents incompatible tensor dimensions. |
| `base_model_version` | **REQUIRED** | Identifies the starting global weights $W_t$. |
| `training_round_id` | **REQUIRED** | Prevents replay attacks across FL rounds. |
| `update_type` | **REQUIRED** | Declares encoding/compression format. |
| `num_examples_bucket` | **REQUIRED** | Enables weighted averaging while bucketing exact browsing counts. |
| `parameter_delta` | **REQUIRED** | The core model update $\Delta W$. |
| `local_validation_summary`| **OPTIONAL** | Client self-check flag; avoids leaking exact loss values. |
| `round_enrollment_token` | **FUTURE / OPTIONAL** | Single-use ephemeral blinded token. Discarded immediately. |
| *Exact Sample Count ($n_i$)* | **REMOVED** | Replaced with coarse buckets to prevent browsing volume tracking. |
| *Persistent Client IDs* | **REMOVED** | Strictly prohibited to prevent cross-round client fingerprinting. |

---

## 7. Aggregation Strategy & Mathematics

### Federated Averaging (FedAvg):
For round $t$, with enrolled client cohort $\mathcal{S}_t \subseteq \{1, \dots, K\}$:
$$W_{t+1} = W_t + \sum_{i \in \mathcal{S}_t} \alpha_i \, \Delta W_i^{(t)}$$

### Privacy-Preserving Weighting Modification:
To prevent user volume tracking, FEDrA defines **Clamped Bucketed Weighting**:
$$\tilde{n}_i = \text{BucketValue}(\text{num\_examples\_bucket}_i), \quad \alpha_i = \frac{\tilde{n}_i}{\sum_{j \in \mathcal{S}_t} \tilde{n}_j}$$
For high-security cohorts, $\alpha_i = \frac{1}{|\mathcal{S}_t|}$ (Uniform Federated Averaging) is applied.

---

## 8. Secure Aggregation & Differential Privacy Roadmap

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        FEDrA THREE-TIER PRIVACY DEFENSE DEPTH                          │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ TIER 1: FEDERATED LEARNING (Data Localization — DESIGNED)                               │
│   • Raw URLs, DOM, and Screenshots never leave the browser.                            │
│   • Only model parameter deltas (ΔW) are exchanged.                                    │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ TIER 2: SECURE AGGREGATION (SecAgg / SecAgg+ — DESIGNED / FUTURE)                      │
│   • Pairwise Diffie-Hellman blinding masks individual updates: ΔW_i + Σ r_ij           │
│   • Blinding masks cancel out during server summation (Σ masks = 0).                   │
│   • Server learns ONLY the aggregate Σ ΔW_i; individual client deltas are zero-knowledge│
├────────────────────────────────────────────────────────────────────────────────────────┤
│ TIER 3: DIFFERENTIAL PRIVACY (DP-FedAvg — DESIGNED / FUTURE)                           │
│   • Candidate mechanism: update clipping + calibrated Gaussian noise injection.        │
│   • Requires formal privacy accountant and explicit (ε, δ) privacy budget.             │
│   • Formal DP guarantees are NOT implemented or validated yet.                         │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Comprehensive Threat Model & Mitigations

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                       FEDrA FL THREAT MATRIX                                           │
├──────────────────────┬──────────────────────────────────┬────────────────────────┬─────────────────────┤
│ Threat Vector        │ Description / Adversary Goal     │ Initial Mitigation     │ Future Mitigation   │
├──────────────────────┼──────────────────────────────────┼────────────────────────┼─────────────────────┤
│ Model Poisoning      │ Malicious clients submit extreme │ L2 gradient clipping;  │ Coordinate-wise     │
│                      │ weights to corrupt detector.     │ Trimmed-Mean Aggreg.   │ Krum / Bulyan FL    │
├──────────────────────┼──────────────────────────────────┼────────────────────────┼─────────────────────┤
│ Backdoor / Trojan    │ Updates craft targeted evasion   │ Automated holdout zero-│ Spectral anomaly /  │
│                      │ for specific phishing domains.   │ day regression gates.  │ Activation cluster. │
├──────────────────────┼──────────────────────────────────┼────────────────────────┼─────────────────────┤
│ Sybil Attacks        │ Adversary creates 1,000 bots to  │ Rate limiting; round   │ Proof-of-Work;      │
│                      │ dominate aggregation weights.    │ enrollment quotas.     │ Device attestation. │
├──────────────────────┼──────────────────────────────────┼────────────────────────┼─────────────────────┤
│ Update Inversion     │ Honest-but-curious server tries  │ L2 clipping; gradient  │ Cryptographic       │
│                      │ to reconstruct training samples. │ perturbation (LDP).    │ SecAgg+ protocol.   │
├──────────────────────┼──────────────────────────────────┼────────────────────────┼─────────────────────┤
│ Free-Rider Clients   │ Clients submit empty/noise       │ Self-validation checks │ Proof-of-Gradient   │
│                      │ updates to consume models.       │ on local update deltas.│ verification tokens │
├──────────────────────┼──────────────────────────────────┼────────────────────────┼─────────────────────┤
│ Server Compromise    │ Attacker hacks coordinator to    │ Cryptographic PKI      │ Multi-signature     │
│                      │ push malicious models to clients │ manifest signatures.   │ transparency log.   │
└──────────────────────┴──────────────────────────────────┴────────────────────────┴─────────────────────┘
```

---

## 10. Multi-Gate Model Validation & Promotion Pipeline

A candidate aggregated model $W_{t+1}$ is **NEVER** deployed automatically. It must pass 5 mandatory validation gates built around the project's actual evaluation assets:

```text
 Candidate Global Model W_(t+1)
               │
               ▼
 ┌───────────────────────────────────────────────────────────────────────┐
 │ GATE A: CORE REGRESSION GATE (Held-Out Test Set — 198 Samples)        │
 │ • Accuracy  ≥ Baseline - 0.0050  (PROPOSED / TO BE CALIBRATED)        │
 │ • Phishing Precision ≥ Baseline - 0.0050                              │
 │ • Phishing Recall    ≥ Baseline  (Zero-tolerance for recall drop)     │
 │ • ROC-AUC Score      ≥ Baseline - 0.0020                              │
 └───────────────────────────────────┬───────────────────────────────────┘
                                     │ PASS
                                     ▼
 ┌───────────────────────────────────────────────────────────────────────┐
 │ GATE B: ZERO-DAY EVALUATION GATE (OpenPhish Live Feed — 25 Samples)   │
 │ • Detection rate must show improvement or statistically acceptable    │
 │   non-regression relative to baseline (76.0% baseline).               │
 └───────────────────────────────────┬───────────────────────────────────┘
                                     │ PASS
                                     ▼
 ┌───────────────────────────────────────────────────────────────────────┐
 │ GATE C: LEGITIMATE-PAGE FALSE POSITIVE GATE (68 Legit Test Samples)   │
 │ • False-positive rate on legitimate test split must remain bounded    │
 │   (FPR ≤ 0.0150, zero acceptable drift on validated safe pages).      │
 └───────────────────────────────────┬───────────────────────────────────┘
                                     │ PASS
                                     ▼
 ┌───────────────────────────────────────────────────────────────────────┐
 │ GATE D: ONNX INTEGRITY & RUNTIME SANITY                               │
 │ • Zero NaN, Infinite, or degenerate weight values.                    │
 │ • Bounded L2 parameter shift: ||W_(t+1) - W_t||_2 ≤ δ_max.            │
 │ • Exact numerical parity upon ONNX opset 17 export (< 10⁻⁵ delta).    │
 └───────────────────────────────────┬───────────────────────────────────┘
                                     │ PASS
                                     ▼
 ┌───────────────────────────────────────────────────────────────────────┐
 │ GATE E: SECURITY & ANOMALY VERIFICATION                               │
 │ • Coordinate-wise outlier rejection against poisoning/backdoors.      │
 └───────────────────────────────────┬───────────────────────────────────┘
                                     │ PASS
                                     ▼
 ┌───────────────────────────────────────────────────────────────────────┐
 │ ARTIFACT SIGNING & STAGED RELEASE                                     │
 │ • Generate SHA-256 hash & sign release manifest with ED25519 PKI.     │
 └───────────────────────────────────────────────────────────────────────┘
```

### Automated Rollback Protocol:
If client telemetry reports an anomaly (e.g. spike in false-positive reports), the browser client immediately discards $W_{t+1}$ and reverts to the previous cached local baseline $W_t$.

---

## 11. Model Versioning & Synchronized Dependency Graph

To prevent breaking browser inference, every model update is pinned to an immutable version tuple:

```text
FEDrA Model Version Contract:
─────────────────────────────
• Architecture Version : 1.0 (MLP: 1314 -> 256 -> 128 -> 64 -> 1)
• Schema Version       : URL-v1 (22 dims), HTML-v1 (12 dims), Visual-v1 (1280 dims)
• Global Model Version : v1.4.0-round42
• Min Client Runtime   : 1.0.0
• Synchronized Hashes  :
    - url_baseline.onnx        (SHA-256)
    - html_baseline.onnx       (SHA-256)
    - mobilenet_v2_visual.onnx (SHA-256)
    - image_baseline.onnx      (SHA-256)
    - fusion_model.onnx        (SHA-256)
    - modality_scalers.json    (SHA-256)
    - attribution_params.json  (SHA-256)
```

---

## 12. Browser Compatibility Analysis

- **Production Browser Client (Chrome Extension)**:
  - Runtime: JavaScript ES6 + WebAssembly.
  - In-Browser Inference: Fully implemented and validated (< 60 ms).
  - **Browser-Side FL Training Status:** **NOT IMPLEMENTED (DESIGNED ONLY)**.
  - Proposed Implementation (Step 10+): Custom lightweight JS SGD kernel for URL/HTML models; Web Worker / Wasm backprop kernel for Fusion MLP.
  - MobileNetV2: Strictly inference-only (feature extraction).
- **Simulation / Server Prototype (Python)**:
  - Runtime: Python 3.10+ (PyTorch, Scikit-Learn, NumPy).
  - Planned for Step 9: Multi-client federated training simulation across dataset partitions.

---

## 13. System Architecture Diagram

```text
                             ┌───────────────────────────────────┐
                             │       FL COORDINATOR SERVER       │
                             │            (DESIGNED)             │
                             │  • Round Scheduling               │
                             │  • Client Enrollment & Quotas     │
                             │  • Weighted Clamped Aggregation   │
                             │  • 5-Gate Validation Pipeline     │
                             │  • ED25519 Artifact Signing       │
                             └─────────────────┬─────────────────┘
                                               │
                                     Signed Global Model
                                       (v1.0.0-round_t)
                                               │
               ┌───────────────────────────────┼───────────────────────────────┐
               │                               │                               │
               ▼                               ▼                               ▼
       ┌───────────────┐               ┌───────────────┐               ┌───────────────┐
       │   CLIENT A    │               │   CLIENT B    │               │   CLIENT C    │
       │ (Browser Ext) │               │ (Browser Ext) │               │ (Browser Ext) │
       ├───────────────┤               ├───────────────┤               ├───────────────┤
       │  LOCAL DATA   │               │  LOCAL DATA   │               │  LOCAL DATA   │
       │ (NEVER SENT)  │               │ (NEVER SENT)  │               │ (NEVER SENT)  │
       │  • URLs       │               │  • URLs       │               │  • URLs       │
       │  • DOM        │               │  • DOM        │               │  • DOM        │
       │  • Images     │               │  • Images     │               │  • Images     │
       ├───────────────┤               ├───────────────┤               ├───────────────┤
       │ LOCAL TRAIN   │               │ LOCAL TRAIN   │               │ LOCAL TRAIN   │
       │  (PROPOSED)   │               │  (PROPOSED)   │               │  (PROPOSED)   │
       │     ΔW_A      │               │     ΔW_B      │               │     ΔW_C      │
       └───────┬───────┘               └───────┬───────┘               └───────┬───────┘
               │                               │                               │
               │ (L2 Clipped + Masked Delta)   │ (L2 Clipped + Masked Delta)   │ (L2 Clipped + Masked Delta)
               └───────────────────────────────┼───────────────────────────────┘
                                               │
                                               ▼
                                  ┌──────────────────────────┐
                                  │   SECURE AGGREGATION     │
                                  │  (Trimmed-Mean / FedAvg) │
                                  └────────────┬─────────────┘
                                               │
                                               ▼
                                  ┌──────────────────────────┐
                                  │  CANDIDATE GLOBAL MODEL  │
                                  │       W_(t+1)            │
                                  └────────────┬─────────────┘
                                               │
                                               ▼
                                  ┌──────────────────────────┐
                                  │  5-GATE VALIDATION SUITE │
                                  │(Gates A, B, C, D, and E) │
                                  └────────────┬─────────────┘
                                               │
                                               ▼
                                  ┌──────────────────────────┐
                                  │ SIGNED ONNX DISTRIBUTION │
                                  │  (Verified In-Browser)   │
                                  └──────────────────────────┘
```

---

## 14. Explicit Implementation Boundaries

| Sub-system | Architectural Status | Implemented in Step 8? | Target Milestone |
|---|---|---|---|
| **FL Architecture & Protocols** | **DESIGNED** | Specification Complete | Step 8 |
| **Model Candidate Selection & Math** | **DESIGNED** | Audited against Codebase | Step 8 |
| **JSON Update Contract Specification** | **DESIGNED** | Schema Defined & Verified | Step 8 |
| **Threat Model & Mitigation Matrix** | **DESIGNED** | Specification Complete | Step 8 |
| **Validation Gate Specification** | **DESIGNED** | Gating Thresholds Aligned | Step 8 |
| **FL Simulation Runtime (Python)** | **PENDING** | Not Implemented in Step 8 | Step 9 (Future) |
| **Browser-Side Backpropagation Kernel**| **PENDING** | Not Implemented in Step 8 | Step 10+ (Future) |
| **SecAgg Cryptographic Protocol** | **PENDING** | Not Implemented in Step 8 | Step 10+ (Future) |
| **Differential Privacy Mechanism** | **PENDING** | Not Implemented in Step 8 | Step 10+ (Future) |
| **Dynamic In-Browser Model Downloader**| **PENDING** | Not Implemented in Step 8 | Step 11+ (Future) |

---
