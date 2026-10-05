# FEDrA — Architecture Specification

This document details both the **Verified Current Architecture** (as actually implemented in the codebase following Step 4 Browser-Native Feature Extraction) and the **Intended Target Architecture** (the target design outlined in project specifications).

---

# PART 1: ACTUAL CURRENT ARCHITECTURE (STEP 4 VERIFIED IMPLEMENTATION)

FEDrA currently operates on an **optimized client-side feature extraction + hybrid inference architecture**:
1. **Primary Runtime Path (Browser-Native Feature Extraction):** The user's active Chrome tab extracts URL features (22 dims via `extension/url_features.js`) and HTML features (12 dims via `extension/html_features.js`) directly within the browser context. `background.js` captures the visible tab screenshot via `chrome.tabs.captureVisibleTab()`. The pre-computed feature vectors, HTML, and base64 screenshot are forwarded to `api_server.py`.
2. **Backend Ingestion:** `scripts/api_server.py` uses client-extracted feature vectors when provided (bypassing server-side HTML/URL parsing), extracts the visual embedding (1280 dims) in-memory, and runs the 1314-dim fusion MLP in ~20ms.
3. **Fallback / CLI Path (Server-Side Selenium):** For non-browser CLI evaluation (`test_fusion.py`, `zero_day_eval.py`), the server performs fallback extraction using headless Chrome via Selenium.

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
   - `url_features.js`: Pure JavaScript extractor of the 22 canonical lexical URL features (100% parity with Python).
   - `html_features.js`: Pure JavaScript extractor of the 12 canonical DOM features (100% parity with Python).
   - `content.js`: Injected script running at `document_idle` with DOM stabilization readiness (~250ms). Extracts URL features and live DOM features directly in-browser, benchmarked via high-precision performance timers.
   - `background.js`: Service worker receiving feature vectors and DOM payload, capturing tab screenshot via `chrome.tabs.captureVisibleTab()`, dispatching unified payload to Flask backend, merging end-to-end telemetry ($T_1..T_7$), and issuing desktop alerts.
   - `popup.html` / `popup.js`: Popup UI rendering scanning, safe, phishing, and dead-site cards with a 3-second polling reload.

2. **Backend Server (`scripts/api_server.py`):**
   - Endpoints: `POST /analyze`, `GET /health`.
   - Host Requirements: Python 3.10+, Flask, PyTorch, Torchvision, Scikit-Learn, Joblib, Pillow, BeautifulSoup4.
   - **Browser-Native Path:** Accepts client-provided URL (22 dims) and HTML (12 dims) feature vectors directly, decodes base64 screenshot in-memory, and runs visual encoder + fusion MLP. **Spawns 0 Selenium processes and completes inference in 20ms–170ms.**
   - **Fallback Path:** Uses server-side URL/HTML extraction and Selenium headless Chrome only when request lacks client features/HTML (e.g. legacy CLI tests).

3. **Machine Learning Pipeline:**
   - Visual Modality: Frozen PyTorch `mobilenet_v2` feature extractor $\rightarrow$ Global Average Pooling $\rightarrow$ 1280 dimensions.
   - HTML Modality: 12 DOM features via JavaScript (`extension/html_features.js`) or BeautifulSoup server fallback.
   - URL Modality: Canonical 22 lexical features via JavaScript (`extension/url_features.js`) or Python (`scripts/url_features.py`).
   - Fusion: Weighted horizontal concatenation $(22 \times 0.4 + 12 \times 0.3 + 1280 \times 0.3 = 1314\text{ dims}) \rightarrow$ `MLPClassifier(256, 128, 64)`.

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

The intended target architecture is a **fully client-side, zero-server, privacy-preserving browser extension** where all capture, feature extraction, neural inference, and explainability run locally within the user's browser runtime.

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

| Architectural Aspect | Old Baseline (Step 0) | Step 3 Implemented | Step 4 Implemented | Target Architecture (Step 6+) |
|---|---|---|---|---|
| **DOM Acquisition** | Selenium re-fetch in Python | Browser-native (`content.js` full DOM) | Browser-native (`content.js`) | Browser-native (`content.js`) |
| **Screenshot Acquisition** | Selenium `save_screenshot` | `chrome.tabs.captureVisibleTab()` base64 | `chrome.tabs.captureVisibleTab()` | `chrome.tabs.captureVisibleTab()` |
| **URL Feature Extraction** | Server-side Python | Server-side Python | **Browser-Native (`url_features.js`)** | In-Browser JS/WASM |
| **HTML Feature Extraction** | Server-side BeautifulSoup | Server-side BeautifulSoup | **Browser-Native (`html_features.js`)** | In-Browser JS/DOM |
| **Inference Location** | Python Flask Backend | Python Flask Backend (In-Memory) | Hybrid (Client features + Flask MLP) | In-Browser (`onnxruntime-web`) |
| **Inference Latency** | ~5.7s – 6.1s | ~22ms – 168ms (Backend total) | **~15ms – 150ms total** | < 100ms (Local client-side) |
| **Selenium Dependency** | Mandatory for every request | Fallback only | Fallback only | Completely removed |
| **Server DNS Lookups** | Mandatory for every request | Bypassed on browser path | Bypassed on browser path | Completely removed |
