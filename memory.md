# FEDrA — Current State Memory

**Last Updated:** 2026-10-06  
**Current Development Phase:** Phase 1 (Integrity Establishment & Hardened In-Browser Feature Extraction)  
**Current Task:** Step 4.5 Complete — Browser Feature Extraction Hardening & Git Release Validation. Pure JavaScript feature extraction (`extension/url_features.js` and `extension/html_features.js`) verified with 100% parity across an expanded test corpus (726 URL feature checks, 60 HTML feature checks). Backend integration verified across `browser_native`, `server_selenium_fallback`, and `client_dead_site` paths. Machine-specific paths removed for full OS portability.  
**Next Task:** Step 5 — In-Browser Inference Engine & ONNX Model Conversion (or SHAP/Grad-CAM explainability).

---

## 1. Project Identity & Objective
- **Name:** FEDrA (Federated Detection of Ransomware & Phishing) — Phase 1
- **Objective:** Real-time multimodal phishing detection analyzing URL structure, HTML content, and visual screenshots.

---

## 2. Verified Current Architecture
- **Architecture Type:** Optimized Hybrid Client-Server (Moving to Client-Side In-Browser Inference).
- **Frontend / Client:** Chrome Extension (Manifest V3) executing browser-native URL feature extraction (22 dims) and HTML feature extraction (12 dims) directly in Content Script (`content.js`, `url_features.js`, `html_features.js`), capturing visible tab screenshot via `chrome.tabs.captureVisibleTab()`.
- **Backend / Host:** Python Flask API (`scripts/api_server.py`) running on `http://localhost:5000` (ingests client feature vectors when available, with server fallback).
- **Acquisition at Runtime:** 
  - **Normal Path:** Browser-native capture directly from active Chrome tab. Zero Selenium spawned. Zero server DNS lookups.
  - **Fallback Path:** Headless Chrome via Selenium (reserved for CLI tests or missing client payload).
- **Inference Engine:** Python host environment using PyTorch (`mobilenet_v2`) and Scikit-Learn (`MLPClassifier` / `LogisticRegression`).


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

| Component | Architecture / Model | Feature Dimension | Input Details | Saved Artifact |
|---|---|---|---|---|
| **URL Baseline** | `LogisticRegression(class_weight='balanced')` | 22 dims | 22 lexical features (`scripts/url_features.py`) | `models/url_baseline.pkl` |
| **HTML Baseline** | `LogisticRegression(class_weight='balanced')` | 12 dims | 12 DOM structural features | `models/html_baseline.pkl` |
| **Visual Baseline**| `LogisticRegression(class_weight='balanced')` | 1280 dims | Pretrained MobileNetV2 GAP embeddings | `models/image_baseline.pkl` |
| **Fusion Model** | `MLPClassifier(hidden_layers=(256, 128, 64))` | 1314 dims | Concatenation of weighted scaled vectors ($22 + 12 + 1280$) | `models/fusion_model.pkl` |

- **Modality Weights:** URL = `0.4`, HTML = `0.3`, Visual = `0.3`.
- **Fusion Vector:** $(22 \times 0.4) + (12 \times 0.3) + (1280 \times 0.3) = 1314$ dimensions.
- **Model Serialization:** `joblib.dump()` / `joblib.load()`.
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
