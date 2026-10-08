# FEDrA — Architecture Specification

This document details both the **Verified Current Architecture** (as actually implemented in the codebase following Step 5 ONNX Export & Numerical Parity Validation) and the **Intended Target Architecture** (the target design outlined in project specifications).

---

# PART 1: ACTUAL CURRENT ARCHITECTURE (STEP 7.5 BROWSER-INTEGRATED EXPLAINABILITY)

FEDrA currently operates with **full browser-native ONNX inference, model-faithful structured feature attribution, visual Grad-CAM explainability, and integrated multimodal evidence synthesis** across all modalities executing locally in-browser via ONNX Runtime Web (WASM) and JavaScript, while the Python Flask server and Selenium fallbacks remain fully functional as side-by-side verification and fallback backends:

1. **Client-Side In-Browser Feature Extraction & Complete ONNX Inference:**
   - The user's active Chrome tab extracts URL features (22 dims via `extension/url_features.js`) and HTML features (12 dims via `extension/html_features.js`) directly in the Content Script.
   - `background.js` captures visible tab screenshots via `chrome.tabs.captureVisibleTab()`.
   - `background.js` (MV3 Service Worker) executes local browser-side inference via `extension/inference.js` using bundled `onnxruntime-web` WASM across all 5 models:
     - `extension/models/url_baseline.onnx` (22 dims -> URL prediction & phishing probability)
     - `extension/models/html_baseline.onnx` (12 dims -> HTML prediction & phishing probability)
     - `extension/models/mobilenet_v2_visual.onnx` (Preprocessed NCHW [1, 3, 224, 224] screenshot -> dual output: 1280 visual embedding + $1280 \times 7 \times 7$ spatial feature activations)
     - `extension/models/image_baseline.onnx` (1280 dims -> Image prediction & phishing probability)
     - `extension/models/fusion_model.onnx` (Concatenated & scaled [22*0.4, 12*0.3, 1280*0.3] -> 1314 dims -> Authoritative Multimodal Verdict)
     - All sessions and modality scalers (`modality_scalers.json`) are cached and reused across tab navigations (~28ms visual latency, ~0.5ms image baseline, ~0.8ms fusion, ~0.2ms lexical/DOM latency).
2. **Model-Faithful Structured Feature Attribution (Step 7.1):**
   - `extension/attribution.js` computes exact linear logit decomposition for URL (22) and HTML (12) Logistic Regression models:
     $$c_i = w_i \cdot \left(\frac{x_i - \mu_i}{\sigma_i}\right), \quad z = b_0 + \sum c_i, \quad P(\text{Phishing}) = \sigma(z)$$
   - Deterministically maps feature contributions ($c_i > 0$ toward Phishing, $c_i < 0$ toward Legitimate) to human-readable explanation reasons.
   - Operates entirely client-side with sub-millisecond execution (< 0.05ms) and zero external dependencies.
3. **Visual Grad-CAM Explainability Engine (Step 7.2):**
   - MobileNetV2 Layer 18 final convolutional block outputs spatial feature map $A \in \mathbb{R}^{1280 \times 7 \times 7}$ where spatial area $\Omega = 7 \times 7 = 49$.
   - Global Average Pooling: $v_k = \frac{1}{49} \sum_{i,j} A_{k,i,j}$.
   - Image Baseline logit: $y^{\text{phish}} = b + \sum_k w_k \cdot \left(\frac{v_k - \mu_k}{\sigma_k}\right)$.
   - Gradient w.r.t spatial activations: $\frac{\partial y^{\text{phish}}}{\partial A_{k,i,j}} = \frac{w_k}{49 \cdot \sigma_k}$.
   - Exact analytical Grad-CAM channel importance weights:
     $$\alpha_k^{\text{phish}} = \frac{1}{49} \sum_{i,j} \frac{\partial y^{\text{phish}}}{\partial A_{k,i,j}} = \frac{w_k}{49 \cdot \sigma_k}$$
   - Spatial importance map computation:
     $$L_{\text{Grad-CAM}}(i, j) = \text{ReLU}\left(\sum_{k=1}^{1280} \alpha_k \cdot A_{k, i, j}\right)$$
   - Bilinear upsampling from $7 \times 7$ to $224 \times 224$ with min-max normalization to $[0.0, 1.0]$.
   - Client-side execution in `extension/attribution.js` (`computeVisualGradCam`, `explainVisualGradCam`) with ~3.3ms latency and exact numerical parity against PyTorch Autograd Grad-CAM ($7.22 \times 10^{-9}$ max diff).
4. **Browser Integration of Multimodal Explainability Pipeline (Steps 7.3, 7.4 & 7.5):**
   - `background.js` orchestrates `FedraAttribution.explainMultimodalPipeline()` strictly downstream of the authoritative prediction pipeline (`features -> ONNX models -> fusion -> final prediction -> explainMultimodalPipeline`).
   - Generates a unified, schema-conformant analysis result (`schema_version: "1.0"`) containing authoritative `prediction`, `modalities`, `explanation`, and `timings`.
   - **Strict Downstream Error Isolation:** Any failure in feature attribution or Grad-CAM computation is captured and reported in the explanation metadata without disrupting or altering the authoritative detector prediction.
   - **Modality Degradation Robustness:** Gracefully handles missing modalities (e.g. dead sites, missing screenshots) without throwing exceptions, setting explicit error codes (`*_EXPLANATION_UNAVAILABLE`).
   - **Deterministic Cross-Modal Agreement:** Evaluates modality consistency (`ALL_PHISHING`, `ALL_LEGITIMATE`, `MIXED`, `PARTIAL_*`, `UNAVAILABLE`).
   - **Zero External Overhead:** Pure in-browser JavaScript execution (~2.0ms latency), 0 external APIs, 0 LLMs, 0 network requests.
5. **Server-Side Verification & Fallback (Flask Backend):**
   - The payload is dispatched to Flask backend (`scripts/api_server.py`) for side-by-side telemetry. If Flask is offline, the browser functions autonomously in standalone local mode.
6. **Fallback / CLI Path (Server-Side Selenium):** For non-browser CLI evaluation (`test_fusion.py`, `zero_day_eval.py`), the server performs fallback extraction using headless Chrome via Selenium.


```
══════════════════════════════════════════════════════════════════════════════════════════
               PRIMARY PATH: BROWSER-NATIVE FEATURE EXTRACTION (CHROME EXTENSION)
══════════════════════════════════════════════════════════════════════════════════════════
                    ┌──────────────────────────────────────────────┐
                    │               USER'S BROWSER                 │
                    │               (Google Chrome)                │
                    └──────────────────────┬───────────────────────┘
                                           │
                         ┌─────────────────┴─────────────────┐
                         ▼                                   ▼
                 [ content.js ]                     [ background.js ]
         ┌───────────────┴───────────────┐        (captureVisibleTab)
         ▼                               ▼           (Base64 PNG)
  [ url_features.js ]           [ html_features.js ]         │
  (22 Canonical Dims)           (12 DOM Features)            │
         │                               │                   │
         └───────────────┬───────────────┘                   │
                         ▼                                   │
              [ Client Feature Vectors ]                     │
                         │                                   │
                         └─────────────────┬─────────────────┘
                                           │ HTTP POST http://localhost:5000/analyze
                                           │ Payload: { url, url_features, html_features, screenshot, timings }
                                           ▼
══════════════════════════════════════════════════════════════════════════════════════════
                    FLASK API BACKEND — INFERENCE ENGINE (api_server.py)
                 [ ZERO Selenium Instances | ZERO Server DNS Lookups ]
══════════════════════════════════════════════════════════════════════════════════════════
                                           │
                         ┌─────────────────┼─────────────────┐
                         ▼                 ▼                 ▼
               [ Client URL Vector ] [ Client HTML Vector ] [ Visual Embedding ]
                (22 canonical dims)   (12 DOM features)     (PyTorch MobileNetV2)
                         │                 │                 │
                         ▼                 ▼                 ▼
                    StandardScaler    StandardScaler       [ GAP ]
                       (22 dims)         (12 dims)           │
                     × Weight 0.4      × Weight 0.3          ▼
                         │                 │            [ 1280-dim Vector ]
                         │                 │                 │
                         │                 │           StandardScaler
                         │                 │            × Weight 0.3
                         │                 │                 │
                         └────────┬────────┴─────────────────┘
                                  ▼
                       [ Concatenation Layer ]
                    (np.hstack -> 1314 dimensions)
                                  │
                                  ▼
                      [ Fusion MLP Classifier ]
                        models/fusion_model.pkl
                       (256 -> 128 -> 64 -> Sigmoid)
                                  │
                                  ▼
                    [ Phishing Verdict & Reasons ]
                     (Latency: 20ms - 170ms total)
                                  │
                                  ▼
                         [ JSON Response ]
══════════════════════════════════════════════════════════════════════════════════════════
                                  │
                                  ▼
                           [ background.js ]
                     (Stores in chrome.storage.local)
                                  │
                         ┌────────┴────────┐
                         ▼                 ▼
                  [ popup.html/js ]  [ Alert Notification ]
                   (Status Gauges)    (Fired on Phishing)
```

---

### Component Breakdown of Implemented Architecture

1. **Browser Frontend (`extension/`):**
   - `manifest.json`: Manifest V3 configuration registering `url_features.js`, `html_features.js`, and `content.js` with required permissions (`activeTab`, `tabs`, `scripting`, `storage`, `notifications`, `webNavigation`).
   - `url_features.js`: Pure JavaScript extractor of the 22 canonical lexical URL features (100% parity with Python across validated corpus).
   - `html_features.js`: Pure JavaScript extractor of the 12 canonical DOM features (100% parity with Python across validated corpus).
   - `content.js`: Injected script running at `document_idle` with DOM stabilization readiness (~250ms). Extracts URL features and live DOM features directly in-browser.
   - `background.js`: Service worker receiving feature vectors and DOM payload, capturing tab screenshot via `chrome.tabs.captureVisibleTab()`, dispatching unified payload to Flask backend, merging end-to-end telemetry ($T_1..T_7$), and issuing desktop alerts.
   - `popup.html` / `popup.js`: Popup UI rendering scanning, safe, phishing, and dead-site cards with a 3-second polling reload.

2. **Backend Server (`scripts/api_server.py`):**
   - Endpoints: `POST /analyze`, `GET /health`.
   - Host Requirements: Python 3.10+, Flask, PyTorch, Torchvision, Scikit-Learn, Joblib, Pillow, BeautifulSoup4.
   - **Browser-Native Path:** Accepts client-provided URL (22 dims) and HTML (12 dims) feature vectors directly, decodes base64 screenshot in-memory, and runs visual encoder + fusion MLP. **Spawns 0 Selenium processes and completes inference in 20ms–170ms.**
   - **Fallback Path:** Uses server-side URL/HTML extraction and Selenium headless Chrome only when request lacks client features/HTML (e.g. legacy CLI tests).

3. **Machine Learning Pipeline & Validated ONNX Artifacts:**
   - Visual Modality: Frozen PyTorch `mobilenet_v2` feature extractor $\rightarrow$ Global Average Pooling $\rightarrow$ 1280 dimensions. Exported as `models/onnx/mobilenet_v2_visual.onnx`.
   - HTML Modality: 12 DOM features via JavaScript (`extension/html_features.js`) or BeautifulSoup server fallback. Exported as `models/onnx/html_baseline.onnx`.
   - URL Modality: Canonical 22 lexical features via JavaScript (`extension/url_features.js`) or Python (`scripts/url_features.py`). Exported as `models/onnx/url_baseline.onnx`.
   - Fusion: Weighted horizontal concatenation $(22 \times 0.4 + 12 \times 0.3 + 1280 \times 0.3 = 1314\text{ dims}) \rightarrow$ `MLPClassifier(256, 128, 64)`. Exported as `models/onnx/fusion_model.onnx`.

---

# PART 2: DNS AUDIT & POLICY ANALYSIS

An audit of all DNS operations in the repository reveals:

| Location | Operation | Purpose | Affects ML Model Features? | Active in Primary Browser-Native Path? |
|---|---|---|---|---|
| `scripts/api_server.py:check_dns()` | `socket.getaddrinfo(hostname, None)` | Host reachability check | **No** (0 DNS features in ML model) | **No** (Bypassed when client HTML is provided) |
| `scripts/test_fusion.py:check_dns()` | `socket.getaddrinfo(hostname, None)` | Pre-flight probe before Selenium | **No** (0 DNS features in ML model) | **No** (CLI script only) |
| `scripts/zero_day_eval.py:check_dns()`| `socket.getaddrinfo(hostname, None)` | Pre-flight probe before Selenium | **No** (0 DNS features in ML model) | **No** (Benchmark script only) |
| `extension/content.js` | Text substring check (`DNS_PROBE`, etc.) | Error page text detection | **No** (Pure string check on DOM) | **Yes** (Client-side dead-site detector) |

**Conclusion:** The ML feature vector is 100% DNS-free. The Flask backend in Step 3 no longer performs server-side DNS queries on browser-native requests.

---

# PART 3: INTENDED TARGET ARCHITECTURE (DESIRED FUTURE STATE)

The intended target architecture is a **fully client-side, zero-server, privacy-preserving browser extension** where all capture, feature extraction, neural inference, and explainability run locally within the user's browser runtime via ONNX Runtime Web.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        INTENDED CLIENT-SIDE BROWSER ARCHITECTURE                       │
│                               (Manifest V3 Extension)                                  │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                        │
│                                  Active Browser Tab                                    │
│                                           │                                            │
│                    ┌──────────────────────┼──────────────────────┐                     │
│                    ▼                      ▼                      ▼                     │
│               [ Page URL ]           [ Live DOM ]     [ chrome.tabs.captureVisibleTab ]│
│                    │                      │                      │                     │
│                    ▼                      ▼                      ▼                     │
│           [ URL Feature Ext. ]   [ DOM Feature Ext. ]   [ Visual Preprocessing (Canvas)│
│            (url_features.js)      (html_features.js)      (224x224 RGB Normalization)  │
│                    │                      │                      │                     │
│                    ▼                      ▼                      ▼                     │
│             [ Fixed Vector ]       [ Fixed Vector ]     [ ONNX MobileNetV2 Encoder ]   │
│                (22 dims)               (12 dims)                 (1280 dims)           │
│                    │                      │                      │                     │
│                    └──────────────────────┼──────────────────────┘                     │
│                                           ▼                                            │
│                            [ In-Browser Scaler & Fusion ]                              │
│                                (ONNX Runtime Web / Wasm)                               │
│                                           │                                            │
│                                           ▼                                            │
│                             [ ONNX Fusion MLP Classifier ]                             │
│                                (Client-side inference)                                 │
│                                           │                                            │
│                                           ▼                                            │
│                           [ Phishing Probability Verdict ]                             │
│                                           │                                            │
│                        ┌──────────────────┴──────────────────┐                         │
│                        ▼                                     ▼                         │
│              [ Client-Side SHAP ]                  [ Client-Side Grad-CAM ]            │
│         (Tabular Feature Attribution)              (Visual Saliency Heatmap)           │
│                        │                                     │                         │
│                        └──────────────────┬──────────────────┘                         │
│                                           ▼                                            │
│                                [ Interactive Popup UI ]                                │
│                       (Instant Warning + Explainability Gauges)                        │
│                                                                                        │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### Architectural Progress & Evolution

| Architectural Aspect | Old Baseline (Step 0) | Step 3 Implemented | Step 4 & 4.5 Implemented | Step 5 Implemented | Target Architecture (Step 6+) |
|---|---|---|---|---|---|
| **DOM Acquisition** | Selenium re-fetch in Python | Browser-native (`content.js` full DOM) | Browser-native (`content.js`) | Browser-native (`content.js`) | Browser-native (`content.js`) |
| **Screenshot Acquisition** | Selenium `save_screenshot` | `chrome.tabs.captureVisibleTab()` base64 | `chrome.tabs.captureVisibleTab()` | `chrome.tabs.captureVisibleTab()` | `chrome.tabs.captureVisibleTab()` |
| **URL Feature Extraction** | Server-side Python | Server-side Python | **Browser-Native (`url_features.js`)** | **Browser-Native (`url_features.js`)** | In-Browser JS/WASM |
| **HTML Feature Extraction** | Server-side BeautifulSoup | Server-side BeautifulSoup | **Browser-Native (`html_features.js`)** | **Browser-Native (`html_features.js`)** | In-Browser JS/DOM |
| **ONNX Model Artifacts** | None | None | None | **Exported & Numerically Validated (`models/onnx/`)** | Local Browser Runtime |
| **Inference Location** | Python Flask Backend | Python Flask Backend (In-Memory) | Hybrid (Client features + Flask MLP) | Hybrid (Active) + Validated ONNX Suite | In-Browser (`onnxruntime-web`) |
| **Inference Latency** | ~5.7s – 6.1s | ~22ms – 168ms (Backend total) | ~15ms – 150ms total | **~15ms – 150ms total** | < 100ms (Local client-side) |
| **Selenium Dependency** | Mandatory for every request | Fallback only | Fallback only | Fallback only | Completely removed |
| **Server DNS Lookups** | Mandatory for every request | Bypassed on browser path | Bypassed on browser path | Bypassed on browser path | Completely removed |
| **Federated Learning** | None | None | None | None | **DESIGNED (Step 8 Protocol, Contracts, Threat Model)** |

---

# PART 3: FEDERATED LEARNING ARCHITECTURE (STEP 8 SPECIFICATION — DESIGNED)

FEDrA defines a privacy-preserving federated model updating architecture enabling client devices to collaboratively train downstream detection models without uploading raw browsing data:

1. **Federated Targets:**
   - **Primary Target:** Fusion MLP classifier (1314 inputs $\rightarrow$ 377,857 params, ~1.51 MB). Topology: 4 dense layers ($256 \rightarrow 128 \rightarrow 64 \rightarrow 1$ sigmoid unit), outputting 2-class probability distribution ($[P(0), P(1)]$) at runtime interface.
   - **Secondary Target:** URL (22 dims) and HTML (12 dims) linear baselines.
   - **Frozen Component:** MobileNetV2 feature extractor remains frozen on client devices (zero conv autograd in browser).
2. **Inference vs Training Decoupling:**
   - In-browser inference is 100% local, offline, and synchronous (< 60 ms).
   - Federated learning is asynchronous, idle-scheduled, and executes in background workers.
   - **Browser-side FL training is NOT IMPLEMENTED (DESIGNED only).**
3. **Client Data Boundary & Contract:**
   - URLs, DOM, screenshots, and browsing activity strictly NEVER leave the client.
   - Transmitted updates are restricted to parameter deltas ($\Delta W$), bucketed sample counts, and version metadata.
4. **Three-Tier Privacy Defense:**
   - Tier 1: Federated Learning (data localization — DESIGNED).
   - Tier 2: Secure Aggregation (SecAgg / SecAgg+ pairwise blinding — DESIGNED / FUTURE).
   - Tier 3: Differential Privacy (clipping + Gaussian noise + privacy accountant — DESIGNED / FUTURE; formal guarantees not yet implemented).
5. **Multi-Gate Model Promotion Pipeline (PROPOSED / TO BE CALIBRATED):**
   - Gate A: Core Regression Gate on 198-sample holdout test set (no unacceptable drop in accuracy, recall, precision, AUC).
   - Gate B: Zero-Day Evaluation Gate on established 25-sample OpenPhish live feed (non-regression relative to 76.0% baseline).
   - Gate C: Legitimate False-Positive Gate on 68-sample safe test split (bounded FPR $\le 1.5\%$).
   - Gate D: ONNX Integrity & Numerical Stability (0 NaN/Inf, bounded parameter shift, verified opset 17 export).
   - Gate E: Security / Anomaly verification against poisoning and backdoor updates.
   - Rollback: Automatic client fallback to previous known-good model $W_t$ if telemetry detects performance degradation.
6. **Detailed Documentation:** Refer to [`docs/federated_learning_architecture.md`](file:///c:/Users/User/OneDrive/Desktop/Mini%20Projects/Mini%20Project-%203rd%20Year/FEDrA/docs/federated_learning_architecture.md) for full contracts and threat matrices.


