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

### 5. ONNX Model Export & Numerical Parity Validation (Step 5 Complete)
- `[x]` Create model export script (`scripts/export_onnx.py`) to convert MobileNetV2, scalers, and Fusion MLP to ONNX (`.onnx`).
- `[x]` Export `models/onnx/url_baseline.onnx` (StandardScaler + LogisticRegression on 22 dims, 866 bytes).
- `[x]` Export `models/onnx/html_baseline.onnx` (StandardScaler + LogisticRegression on 12 dims, 666 bytes).
- `[x]` Export `models/onnx/image_baseline.onnx` (StandardScaler + LogisticRegression on 1280 dims, 26 KB).
- `[x]` Export `models/onnx/mobilenet_v2_visual.onnx` (MobileNetV2 feature extractor + GAP to 1280 dims, 8.86 MB).
- `[x]` Export `models/onnx/fusion_model.onnx` (MLPClassifier 1314 -> 256 -> 128 -> 64 -> 2, 1.51 MB).
- `[x]` Export `models/onnx/modality_scalers.json` (scaling parameters and modality weights) and `model_contracts.json`.
- `[x]` Validate numerical parity against Python across 50 representative samples (`models/onnx/validation_baseline.json`) — 100% class agreement, max probability diff $\le 2.32 \times 10^{-6}$.

### 5.1 ONNX Runtime Web Compatibility Validation (Step 5.1 Complete)
- `[x]` Execute all 5 ONNX models directly using `onnxruntime-web` with the WebAssembly (WASM) engine.
- `[x]` Validate exact input/output tensor contracts across all models with zero custom or unsupported operator errors.
- `[x]` Verify MobileNetV2 preprocessing contract (`[1, 3, 224, 224]`, NCHW, ImageNet normalization).
- `[x]` Verify manual modality scaling ($W_{\text{url}}=0.4$, $W_{\text{html}}=0.3$, $W_{\text{visual}}=0.3$) and 1314-dimensional concatenation logic.
- `[x]` Verify 100% class prediction agreement and micro-scale numerical diffs ($< 4.01 \times 10^{-7}$) across the 50 validation fixtures in `onnxruntime-web` WASM.

### 6.1 Browser URL + HTML ONNX Inference (Step 6.1 Complete)
- `[x]` Bundle `onnxruntime-web` WASM runtime into `extension/libs/onnxruntime-web/`.
- `[x]` Bundle `url_baseline.onnx` and `html_baseline.onnx` into `extension/models/`.
- `[x]` Implement focused browser inference module (`extension/inference.js`) with cached ONNX sessions and safe error handling.
- `[x]` Update `extension/manifest.json` with `web_accessible_resources` for models and WASM libs.
- `[x]` Integrate local URL (22) and HTML (12) ONNX inference into `extension/background.js` MV3 service worker.
- `[x]` Retain server-side visual and multimodal fusion inference for production verdict and diagnostic comparison.
- `[x]` Verify 100% classification agreement and numerical parity across 50 validation fixtures (URL: max diff $2.08 \times 10^{-7}$, HTML: max diff $7.31 \times 10^{-8}$).
- `[x]` Verify full regression suite (786 feature checks, 22/12/1280/1314 contract, browser-native, dead-site, and Selenium fallback).

### 6.2 Browser-Native MobileNetV2 Visual Feature Extraction (Step 6.2 Complete)
- `[x]` Bundle `models/onnx/mobilenet_v2_visual.onnx` into `extension/models/`.
- `[x]` Implement browser screenshot preprocessing in `extension/inference.js` (`Resize(256) -> CenterCrop(224) -> ImageNet Normalization -> NCHW Float32Array`).
- `[x]` Extend `extension/inference.js` with cached `mobilenetSession` and `runVisualInference()`.
- `[x]` Update `extension/background.js` to run local visual extraction on tab screenshot and forward 1280-dim embedding to Flask.
- `[x]` Update `scripts/api_server.py` to validate and ingest client 1280 visual embedding with automatic server PyTorch fallback.
- `[x]` Validate visual embedding parity against PyTorch reference across 50 dataset samples (Max abs diff: $1.89 \times 10^{-5}$, Mean abs diff: $7.61 \times 10^{-7}$).
- `[x]` Verify complete regression suite (786 feature checks, URL/HTML in-browser ONNX, Flask fusion, Selenium fallback).

### 6.3 Browser Image Baseline & Fusion ONNX Inference (Step 6.3 Complete & Validated)
- `[x]` Bundle `image_baseline.onnx`, `fusion_model.onnx`, and `modality_scalers.json` into `extension/models/`.
- `[x]` Implement `runImageBaselineInference()` (1280 raw visual embedding input with embedded StandardScaler) in `extension/inference.js`.
- `[x]` Implement `preprocessFusionInput()` and `runFusionInference()` ($[22 \times 0.4, 12 \times 0.3, 1280 \times 0.3] \rightarrow 1314$ dims) in `extension/inference.js`.
- `[x]` Implement `runFullBrowserPipeline()` to orchestrate complete local detection flow.
- `[x]` Update `extension/background.js` to execute local image baseline and fusion ONNX inference alongside Flask side-by-side comparison.
- `[x]` Validate Image Baseline numerical parity across 50 dataset fixtures (Max prob diff: $4.01 \times 10^{-7}$, Class agreement: 100%).
- `[x]` Validate Fusion MLP numerical parity across 50 dataset fixtures (Max prob diff: $1.17 \times 10^{-7}$, Class agreement: 100%).
- `[x]` Verify full regression suite, defensive failure rejection (6/6 PASS), and fallback paths (Flask and Selenium retained).

### 6.4 Client-Authoritative Decision & Server Decoupling (Next Step)
- `[ ]` Finalize browser-local verdict as authoritative detection path.
- `[ ]` Transition Flask server to optional audit/logging mode.

### 7.1 Model-Faithful Structured Feature Attribution (Step 7.1 Complete & Validated)
- `[x]` Inspect URL (22) and HTML (12) Logistic Regression models and identify exact linear logit decomposition method ($c_i = w_i \cdot z_i$).
- `[x]` Export model coefficients, intercepts, scalers, and description templates to `models/onnx/attribution_parameters.json` and `extension/models/attribution_parameters.json`.
- `[x]` Implement browser-native attribution engine (`extension/attribution.js`) providing `explainUrlPrediction()`, `explainHtmlPrediction()`, and `explainStructuredModalities()`.
- `[x]` Validate mathematical reconstruction ($|z - (b_0 + \sum c_i)| < 10^{-14}$) across 50 dataset fixtures for both URL and HTML models.
- `[x]` Validate Python vs JavaScript numerical parity ($1.43 \times 10^{-6}$ max abs diff) and 100% ranking agreement.
- `[x]` Verify 100% deterministic output and sub-millisecond execution (< 0.05ms).
- `[x]` Integrate attribution into `extension/background.js` telemetry without modifying detector predictions.

### 7.2 Grad-CAM Visual Explainability Engine (Step 7.2 Complete & Validated)
- `[x]` Derived exact analytical Grad-CAM formulation for MobileNetV2 Layer 18 feature map ($1280 \times 7 \times 7$) $\rightarrow$ GAP $\rightarrow$ StandardScaler $\rightarrow$ LogisticRegression classifier ($\alpha_k = w_k / (49 \cdot \sigma_k)$).
- `[x]` Proven exact mathematical equivalence against PyTorch Autograd Grad-CAM ($7.22 \times 10^{-9}$ max diff).
- `[x]` Exported `mobilenet_v2_visual.onnx` with dual outputs (`visual_embedding` $[1, 1280]$ and `spatial_features` $[1, 1280, 7, 7]$) preserving 100% backward compatibility.
- `[x]` Exported exact visual Grad-CAM weights $\alpha_k$ to `attribution_parameters.json` under `gradcam_channel_weights`.
- `[x]` Implemented high-performance browser-native Grad-CAM engine in `extension/attribution.js` (`computeVisualGradCam`, `explainVisualGradCam`, `interpolate7x7To224x224`).
- `[x]` Validated Python vs JavaScript numerical parity across 50 visual samples (Grid max diff: $1.27 \times 10^{-7}$, Heatmap max diff: $1.37 \times 10^{-6}$, Peak regions agreement: 100%).
- `[x]` Measured in-browser execution latency: Avg 3.34 ms, P95 5.74 ms.
- `[x]` Verified 100% prediction invariance across URL, HTML, MobileNetV2, Image Baseline, and Fusion models.

### 7.3 Multimodal Explanation Fusion (Step 7.3 Complete & Validated)
- `[x]` Designed structured explanation contract (`schema_version: "1.0"`) synthesizing URL linear attribution, HTML linear attribution, and Visual Grad-CAM.
- `[x]` Enforced strict architectural separation between model evidence and multimodal summary; fusion MLP classifier remains authoritative.
- `[x]` Implemented deterministic Top-N ($N=3$) absolute attribution ranking with raw and scaled values, exact contribution, direction, and descriptions.
- `[x]` Implemented visual summary contract (Grad-CAM method, target class 1, dimensions, min/max/mean stats, peak regions, objective language).
- `[x]` Implemented deterministic Cross-Modal Agreement evaluation (`ALL_PHISHING`, `ALL_LEGITIMATE`, `MIXED`, `PARTIAL_*`, `UNAVAILABLE`).
- `[x]` Implemented in `scripts/explain_features.py` and browser extension `extension/attribution.js` (`fuseMultimodalExplanations`).

### 7.4 Explanation Consistency & Evaluation (Step 7.4 Complete & Validated)
- `[x]` Verified 100% prediction invariance across all 5 models (URL, HTML, MobileNetV2, Image Baseline, Fusion MLP).
- `[x]` Verified exact evidence reuse (Step 7.1 linear logit decomposition and Step 7.2 Grad-CAM without recomputation).
- `[x]` Verified 100% directional consistency ($c_i > 0 \implies \text{phishing}$, $c_i < 0 \implies \text{legitimate}$).
- `[x]` Verified cross-modal agreement states across canonical test cases (A, B, C1, C2, D1, D2, D3, E).
- `[x]` Verified graceful missing-modality handling with explicit error codes (`*_EXPLANATION_UNAVAILABLE`).
- `[x]` Verified 100% byte-identical determinism across 10 repeated evaluation runs.
- `[x]` Verified feature perturbation directional sanity check (reducing top phishing feature decreases logit).
### 7.5 Browser Integration of the Explainability Pipeline (Step 7.5 Complete & Validated)
- `[x]` Integrated `FedraAttribution.explainMultimodalPipeline()` downstream of authoritative prediction in `extension/background.js`.
- `[x]` Configured browser analysis result contract containing authoritative `prediction`, `modalities`, `explanation` (`schema_version: "1.0"`), and comprehensive `timings`.
- `[x]` Enforced strict downstream error isolation: explainability exceptions never alter or block prediction verdicts.
- `[x]` Handled all modality degradation permutations (Full 3 modalities, URL+HTML, URL only, Visual only, Missing inputs).
- `[x]` Validated 100% prediction invariance across dataset test fixtures with and without explainability.
- `[x]` Benchmarked in-browser execution latency: JS P50 = 2.00 ms, P95 = 3.00 ms, Max = 4.00 ms (well within < 20 ms budget).
- `[x]` Maintained full backward compatibility for extension storage (`last_result`), alerts, and fallback pathways.

### 7.6 Extension UI Explainability Integration (Step 7.6 Complete & Validated)
- `[x]` Designed clean, accessible, CSP-compliant extension popup UI (`extension/popup.html`) with zero external fonts/CDNs and dark cybersecurity theme.
- `[x]` Implemented strictly rendering-only controller (`extension/popup.js`) with zero ONNX inference, zero feature extraction, and zero Grad-CAM recomputation.
- `[x]` Rendered authoritative final verdict, safety confidence/phishing probability progress bar, and risk level pill badge.
- `[x]` Rendered cross-modal agreement banner with consensus chip (`ALL_PHISHING`, `ALL_LEGITIMATE`, `MIXED`, `PARTIAL_*`, `UNAVAILABLE`) and factual summary text.
- `[x]` Rendered structured modality evidence cards for URL structure and Page DOM content featuring top-3 contributing features with direction tags and model-faithful descriptions.
- `[x]` Rendered interactive visual layout Grad-CAM card with dynamic 7x7 spatial heatmap grid and peak activation region coordinates.
- `[x]` Implemented collapsible Technical Diagnostics accordion drawer displaying granular inference latencies, probability breakdown, decision source, and schema version.
- `[x]` Implemented "Return to Safety" navigation handler for phishing and unreachable sites.
- `[x]` Validated all 6 UI states and live headless Chrome MV3 extension runtime in `scratch/test_step7_6_ui.py`.

### 8. End-to-End Validation
- `[ ]` Run full validation across static holdout test set (198 samples) and zero-day live feed.
- `[ ]` Benchmark in-browser inference latency (target: < 100ms).
- `[ ]` Benchmark browser memory and resource footprint.


