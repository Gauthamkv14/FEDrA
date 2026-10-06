# FEDrA — Current State Memory

**Last Updated:** 2026-10-06  
**Current Development Phase:** Phase 1 (Step 6.3 — Browser Image Baseline & Fusion ONNX Migration)  
**Current Task:** Step 6.3 Complete — Browser-Native Image Baseline & Fusion ONNX Inference. Bundled `image_baseline.onnx`, `fusion_model.onnx`, and `modality_scalers.json` into `extension/models/`. Extended `extension/inference.js` with `runImageBaselineInference()`, `preprocessFusionInput()`, `runFusionInference()`, and `runFullBrowserPipeline()`. In-browser inference is now capable of producing the complete final multimodal verdict across all 5 models locally via `onnxruntime-web` WASM. Side-by-side verification with Flask API backend confirmed 100% class agreement and micro-scale numerical parity across 50 dataset fixtures (Image baseline max prob diff: $4.01 \times 10^{-7}$, Fusion MLP max prob diff: $1.17 \times 10^{-7}$). Defensive failure handling (6/6 PASS) and legacy fallbacks (Flask and Selenium) verified.  
**Next Task:** Step 6.4 — Client-Authoritative Decision & Server Decoupling, or Explainability (SHAP/Grad-CAM).

---

## 1. Project Identity & Objective
- **Name:** FEDrA (Federated Detection of Ransomware & Phishing) — Phase 1
- **Objective:** Real-time multimodal phishing detection analyzing URL structure, HTML content, and visual screenshots.

---

## 2. Verified Current Architecture
- **Architecture Type:** Full Browser-Native ONNX Inference with Retained Server Side-by-Side Verification / Fallback.
- **Frontend / Client:** Chrome Extension (Manifest V3) executing browser-native URL feature extraction (22 dims), HTML feature extraction (12 dims), tab screenshot capture, and local in-browser ONNX inference for all 5 models (`url_baseline.onnx`, `html_baseline.onnx`, `mobilenet_v2_visual.onnx`, `image_baseline.onnx`, `fusion_model.onnx`) in `background.js` via `extension/inference.js`.
- **Backend / Host:** Python Flask API (`scripts/api_server.py`) running on `http://localhost:5000` (retained as reference backend, side-by-side comparison, and defensive fallback).
- **Acquisition at Runtime:** 
  - **Normal Path:** Browser-native capture directly from active Chrome tab. Zero Selenium spawned. Zero server DNS lookups.
  - **Fallback Path:** Headless Chrome via Selenium (reserved for CLI tests or missing client payload).
- **Inference Engines:** 
  - Browser: `onnxruntime-web` (WebAssembly provider) executing all 5 models locally.
  - Host: Scikit-Learn `MLPClassifier` + `LogisticRegression` (with PyTorch MobileNetV2 fallback).


---

## 3. Dataset & Splits
- **Total Samples:** 990 samples indexed in `Dataset/manifest.csv`.
  - **Legitimate (Label 0):** 340 samples (`Dataset/legit_dataset/`).
  - **Phishing (Label 1):** 650 samples (`Dataset/phised_dataset/`).
  - **Class Split:** 65.66% Phishing / 34.34% Legitimate.
- **Train / Test Split:** Stratified 80/20 split (`random_state=42`):
  - Train: 792 samples (520 Phishing, 272 Legit).
  - Test: 198 samples (130 Phishing, 68 Legit).
- **Zero-Day Holdout:** 25 live URLs evaluated from OpenPhish feed (`notebooks/zero_day_results.csv`, 76.0% detection rate).

---

## 4. Models & Feature Dimensions

| Component | Architecture / Model | Feature Dimension | Input Details | Python Artifact | ONNX Artifact (`models/onnx/`) |
|---|---|---|---|---|---|
| **URL Baseline** | `LogisticRegression(class_weight='balanced')` | 22 dims | 22 lexical features (`scripts/url_features.py`) | `models/url_baseline.pkl` | `url_baseline.onnx` (866 B) |
| **HTML Baseline** | `LogisticRegression(class_weight='balanced')` | 12 dims | 12 DOM structural features | `models/html_baseline.pkl` | `html_baseline.onnx` (666 B) |
| **Visual Baseline**| `LogisticRegression(class_weight='balanced')` | 1280 dims | Pretrained MobileNetV2 GAP embeddings | `models/image_baseline.pkl` | `image_baseline.onnx` (26 KB) |
| **Visual Extractor**| MobileNetV2 (Feature Extractor + GAP) | `[1, 3, 224, 224]` -> 1280 | RGB normalized image tensor | `torchvision.models.mobilenet_v2` | `mobilenet_v2_visual.onnx` (8.86 MB) |
| **Fusion Model** | `MLPClassifier(hidden_layers=(256, 128, 64))` | 1314 dims | Concatenation of weighted scaled vectors ($22 + 12 + 1280$) | `models/fusion_model.pkl` | `fusion_model.onnx` (1.51 MB) |

- **Modality Weights:** URL = `0.4`, HTML = `0.3`, Visual = `0.3`.
- **Fusion Vector:** $(22 \times 0.4) + (12 \times 0.3) + (1280 \times 0.3) = 1314$ dimensions.
- **Model Serialization:** Python: `joblib.dump()`, ONNX: opset 17 (`skl2onnx` + `torch.onnx`).
- **ONNX Contracts & Scalers:** `models/onnx/model_contracts.json`, `models/onnx/modality_scalers.json`.
- **Performance Metrics (Test Split):**
  - URL Baseline: Acc `0.9798`, AUC `0.9985`, F1 `0.9847`
  - HTML Baseline: Acc `0.8586`, AUC `0.9304`, F1 `0.8906`
  - Visual Baseline: Acc `0.8434`, AUC `0.9022`, F1 `0.8755`
  - Fusion Model: Acc `0.9899`, AUC `0.9997`, F1 `0.9924`

---

## 5. Current Inference Path & Extension Behavior
1. User visits webpage in Chrome.
2. `content.js` runs at `document_idle`, stabilizes DOM readiness (~250ms), extracts full `outerHTML` (up to 2MB cap), and sends message to `background.js`.
3. `background.js` captures visible tab screenshot via `chrome.tabs.captureVisibleTab()`, constructs payload with timing telemetry, and issues HTTP `POST` to `http://localhost:5000/analyze`.
4. Flask server (`api_server.py`):
   - Receives browser DOM and base64 screenshot.
   - Extracts URL (22 dims), HTML (12 dims), and Visual (1280 dims) in-memory with zero Selenium re-fetching and zero server DNS queries.
   - Runs fusion model (1314 dims) in ~20ms–170ms.
   - If client reports dead-site navigation error, runs URL-only model.
   - Generates heuristic reason strings via `build_reasons()`.
   - Returns JSON payload with complete diagnostic timings.
5. `background.js` stores response in `chrome.storage.local` (`last_result`) and fires desktop notification on phishing.
6. `popup.js` reads `last_result`, verifies tab hostname, and renders status cards.

---

## 6. Current Explainability State
- **SHAP:** `NOT IMPLEMENTED` (0 occurrences in code).
- **Grad-CAM:** `NOT IMPLEMENTED` (0 occurrences in code).
- **Actual Explainability:** Rule-based heuristics in `build_reasons()` inside `scripts/api_server.py` evaluating static boolean checks on URL/HTML metadata.

---

## 7. Major Known Contradictions & Critical Issues
1. **[RESOLVED IN STEP 2] URL Training vs. Inference Feature Schema Mismatch:**
   - Successfully unified on canonical 22-dimensional feature schema via `scripts/url_features.py`.
2. **[RESOLVED IN STEP 3] Selenium Dependency on Normal Acquisition Path:**
   - Active Chrome tab now acquires DOM and screenshot natively; Selenium is bypassed on normal detection and kept only as fallback for headless CLI tests.
3. **MLP Claim for Baselines vs. Actual Logistic Regression:**
   - Documentation states URL/HTML baselines are MLPs; actual saved models are `LogisticRegression`.
4. **SHAP & Grad-CAM Claims vs. Rule-Based Heuristics:**
   - Documentation asserts SHAP and Grad-CAM explainability; actual code uses static `if/elif` string generation.
5. **DNS-Free Feature State:**
   - All 22 URL, 12 HTML, and 1280 Visual features are 100% DNS-free. Server-side DNS queries are completely bypassed on the browser-native path.
6. **Documentation Cleanup:**
   - Stale `CLAUDE.md` (which contained unresolved merge conflict markers on lines 4–8 and 147–152) has been retired and preserved as `CLAUDE_ARCHIVED.md`. All active instructions, constraints, and schemas are governed by `agent.md`, `memory.md`, `architecture.md`, `progress.md`, and `feature_schema.md`.

---

## 8. Important Constraints
- `Dataset/` is strictly read-only.
- Never use folder names as features (prevents domain vs hash label leakage).
- All models must be saved/loaded with `joblib`.
- Development branch is `development`.
