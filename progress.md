# FEDrA — Development Progress & Roadmap

This document tracks the verified implementation status, known blockers, and the prioritized roadmap for FEDrA.

---

## Status Legend
- `[x]` Verified Complete — Tested and verified in code/artifacts.
- `[~]` Partially Complete — Implemented in part; requires integration or completion.
- `[ ]` Pending — Planned work not yet started.
- `[!]` Blocked / Critical Issue — Requires immediate resolution before proceeding.

---

## Current Verified State

### Data & Exploration
- `[x]` Dataset collection script (`legitamate.py`)
- `[x]` Global sample indexing (`scripts/build_manifest.py` $\rightarrow$ `Dataset/manifest.csv` with 990 verified samples)
- `[x]` Exploratory Data Analysis & visual plots (`notebooks/eda.ipynb`, `notebooks/figures/*.png`)

### Feature Extraction (Batch & In-Browser)
- `[x]` URL feature extraction (`scripts/extract_url_features.py` + `scripts/url_features.py` $\rightarrow$ `Dataset/features/url_features.csv` — canonical 22 features, zero mismatch)
- `[x]` In-Browser URL feature extraction (`extension/url_features.js` — 22 canonical features, 100% parity with Python)
- `[x]` HTML feature extraction (`scripts/extract_html_features.py` $\rightarrow$ `Dataset/features/html_features.csv` — 12 features)
- `[x]` In-Browser HTML feature extraction (`extension/html_features.js` — 12 DOM features, 100% parity with Python)
- `[x]` Visual embedding extraction (`scripts/extract_visual_embeddings.py` $\rightarrow$ `Dataset/features/visual_embeddings.npy` — 1280-dim MobileNetV2 GAP)

### Model Training & Artifacts
- `[x]` Baseline training (`scripts/train_baselines.py` $\rightarrow$ `models/url_baseline.pkl` [22 dims], `models/html_baseline.pkl` [12 dims], `models/image_baseline.pkl` [1280 dims], `baseline_metrics.json`)
- `[x]` Fusion MLP training (`scripts/train_fusion.py` $\rightarrow$ `models/fusion_model.pkl` [1314 dims = $22 + 12 + 1280$], `models/fusion_metrics.json`)

### Standalone Inference & Evaluation
- `[x]` Single URL CLI baseline evaluator (`scripts/test_single_url.py` — canonical 22 dims)
- `[x]` Single URL CLI fusion evaluator (`scripts/test_fusion.py` — canonical 1314 fused dims, 0 padding)
- `[x]` Zero-day benchmark on live OpenPhish feed (`scripts/zero_day_eval.py` $\rightarrow$ `notebooks/zero_day_results.csv`, 76.0% detection)

### Server & Interface
- `[x]` Flask backend server (`scripts/api_server.py` — accepts pre-computed browser feature vectors, with server fallback)
- `[x]` Standalone web dashboard (`extension/index.html`)
- `[x]` Chrome Extension MV3 UI & background relay (`extension/manifest.json`, `background.js`, `popup.html`, `popup.js`)
- `[x]` Browser-Native DOM & Screenshot Capture (`extension/content.js`, `extension/background.js`)
- `[x]` Extension error/dead-site handling (`onErrorOccurred` listener and DOM error signature checks)
- `[~]` Decision explanation generation (Heuristic boolean checks in `build_reasons()`; SHAP/Grad-CAM missing)

---

## Prioritized Implementation Roadmap

### 1. URL Feature-Schema Integrity
- `[x]` **Step 1: Forensic Analysis (Complete):** Identified exact root causes, mapped feature-by-feature discrepancies, and proposed 22-dimensional candidate canonical schema in `feature_schema.md`.
- `[x]` **Step 2: Implementation (Complete):** Created `scripts/url_features.py`, regenerated `Dataset/features/url_features.csv` (22 dims), retrained `url_baseline.pkl` (22 dims) and `fusion_model.pkl` (1314 dims), aligned all 4 inference scripts without zero-padding, preserved historical metrics in `models/historical_metrics_pre_schema_fix.json`.

### 2. Re-train & Re-evaluate Affected Models
- `[x]` Re-extract canonical URL features for the 990 dataset samples (`Dataset/features/url_features.csv`).
- `[x]` Retrain `url_baseline.pkl` (22 dims) and `fusion_model.pkl` (1314 dims) with the aligned schema.
- `[x]` Update `models/baseline_metrics.json` and `models/fusion_metrics.json`.
- `[x]` Re-run `scripts/zero_day_eval.py` and log updated metrics (76.0% accuracy on 25 zero-day samples).

### 3. Browser-Native Page Acquisition
- `[x]` Implement DOM acquisition in `content.js` without naive 30k truncation (principled 2MB safety threshold).
- `[x]` Implement dynamic DOM readiness strategy (~250ms stabilization delay replacing arbitrary 2000ms pause).
- `[x]` Implement visible tab screenshot capture in `background.js` via `chrome.tabs.captureVisibleTab()`.
- `[x]` Eliminate Selenium page re-fetching on the primary browser detection path in `api_server.py`.
- `[x]` Implement in-memory base64 screenshot decoding in Flask backend.
- `[x]` Audit all DNS calls; confirm ML features are 100% DNS-free and bypass server DNS queries on browser-native path.
- `[x]` Add end-to-end timing telemetry ($T_1..T_7$) measuring readiness, capture, transfer, feature extraction, and inference.

### 4. Browser-Native Feature Extraction (Step 4 Complete)
- `[x]` Create pure JavaScript 22-feature canonical URL extractor (`extension/url_features.js`).
- `[x]` Create pure JavaScript 12-feature canonical HTML/DOM extractor (`extension/html_features.js`).
- `[x]` Register JS feature modules in `manifest.json` content scripts.
- `[x]` Integrate in-browser feature extraction in `content.js` with micro-benchmark telemetry.
- `[x]` Update `background.js` to dispatch client-extracted vectors to Flask server.
- `[x]` Update `api_server.py` to ingest client feature vectors directly with server-side fallback.

### 4.5 Browser Feature Extraction Hardening & Parity Validation (Step 4.5 Complete)
- `[x]` Expand parity test corpus with 33 edge-case URLs (multi-part ccTLDs, IPv4, ports, '@', blank query values, punycode) and 5 complex DOM structures.
- `[x]` Refine JavaScript query token parser in `url_features.js` to strictly match Python `urllib.parse.parse_qsl` blank-value discard semantics.
- `[x]` Verify 100% mathematical and semantic parity across validated test corpus (726 URL checks, 60 HTML checks, 0 mismatches).
- `[x]` Verify end-to-end backend integration on `browser_native`, `server_selenium_fallback`, and `client_dead_site` paths.
- `[x]` Ensure OS environment portability (removed machine-specific hardcoded paths from documentation).
- `[x]` Audit codebase for zero unexpected DNS/WHOIS queries and zero non-canonical dimensions.

### 5. In-Browser Model Conversion & Inference (Next Step)
- `[ ]` Create model export script (`scripts/export_onnx.py`) to convert MobileNetV2, scalers, and Fusion MLP to ONNX (`.onnx`).
- `[ ]` Integrate `onnxruntime-web` into Chrome Extension.
- `[ ]` Implement in-browser tensor normalization and execution, eliminating Python backend requirement.

### 6. SHAP Explainability Engine
- `[ ]` Implement `shap` feature attribution pipeline for URL and HTML structured inputs.
- `[ ]` Integrate top feature attribution weights into the extension UI.

### 7. Grad-CAM Visual Explainability Engine
- `[ ]` Implement Grad-CAM heatmap generation on the last convolutional layer of MobileNetV2.
- `[ ]` Expose visual overlay heatmaps to the UI to highlight suspicious webpage regions.

### 8. End-to-End Validation
- `[ ]` Run full validation across static holdout test set (198 samples) and zero-day live feed.
- `[ ]` Benchmark in-browser inference latency (target: < 100ms).
- `[ ]` Benchmark browser memory and resource footprint.
