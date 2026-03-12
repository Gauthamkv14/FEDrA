# FEDrA – Project Context (Phase-1)
**Last Updated:** 2026-03-08
**Repo:** https://github.com/Gauthamkv14/FEDrA.git
**Status:** Zero-Day Evaluation Done → Flask API + Web UI Next

> **FEDrA** = **FE**derated **D**etection of **Ra**nsomware & Phishing
> Phase-1 Goal: Explainable, multimodal browser extension for real-time zero-day phishing detection.

---

## 🤖 Claude Code (Antigravity) Instructions

This file is the single source of truth for the AI agent working on this project.

### Ground Rules for Agent
- **Never use folder names as features** — label leakage risk (see Dataset Notes)
- **Always work on `development` branch** — never commit to `main`
- **Always activate `fedra` conda env** before running any Python
- **Read this file fully** before starting any task
- **One task at a time** — complete and test before moving to next
- **Never delete dataset folders** — treat `Dataset/` as read-only raw data
- **All models are saved and loaded using joblib (not pickle).**
- **Always use joblib.dump() to save and joblib.load() to load .pkl files.**

### How to Run Tasks
Agent should follow this pattern for every task:
1. Read relevant section of CLAUDE.md
2. Check current branch (`git branch`)
3. Do the work
4. Test the output
5. Commit to `development` with a clear message
6. Report what was done and what's next

---

## 🎯 Core Objective

Detect phishing websites in real time using a browser extension (Manifest V3) analyzing:
- **URL structure**
- **HTML/DOM content**
- **Visual layout (screenshot)**

Requirements:
- Detect **zero-day attacks** (no blacklists)
- **Human-readable explanations** (SHAP + Grad-CAM)
- Run **locally** in the browser (no server API calls during inference)
- **Lightweight** — feasible on modest hardware

---

## ✅ Finalized Decisions

| Component           | Decision                                                        |
|---------------------|-----------------------------------------------------------------|
| Deployment          | Browser Extension (Chrome/Edge, Manifest V3)                   |
| Inference Mode      | Real-time, client-side                                          |
| Modalities          | URL + HTML/DOM + Screenshot                                     |
| Model Architecture  | MLP (URL) + MLP (HTML) + CNN (Image) → Concat → MLP Classifier |
| Visual Encoder      | ResNet18 / MobileNetV2 (candidates)                            |
| Fusion Strategy     | Feature-level concatenation                                     |
| Explainability      | SHAP (URL/HTML), Grad-CAM (Image)                              |
| Training Approach   | Offline training + model distillation                           |
| Export Format       | ONNX or TensorFlow.js                                           |
| Constraints         | No DNS, no paid APIs, no server inference                       |

---

## 📁 Project Structure

```
FEDrA/
├── CLAUDE.md                        ← you are here
├── .gitignore
├── legitamate.py                    ← legit dataset collection script
├── Dataset/
│   ├── legit_dataset/               # 340 folders, named by domain (www.google.com)
│   │   └── <domain>/
│   │       ├── metadata             # contains the actual URL string
│   │       ├── page.html
│   │       └── screenshot
│   ├── phised_dataset/              # 650 folders, named by hash ID (0002_bafybei...link)
│   │   └── <id>/
│   │       ├── metadata
│   │       ├── page.html
│   │       └── screenshot
│   ├── manifest.csv                 ← DONE (Step 2)
│   └── features/                   ← DONE (Step 5)
│       ├── url_features.csv
│       ├── html_features.csv
│       └── visual_embeddings.npy
├── notebooks/
│   ├── eda.ipynb                   ← DONE (Step 3)
│   └── zero_day_results.csv        ← DONE (Step 8)
├── scripts/
│   ├── build_manifest.py
│   ├── extract_url_features.py
│   ├── extract_html_features.py
│   ├── extract_visual_embeddings.py
│   ├── train_baselines.py          ← Step 6
│   ├── train_fusion.py             ← Step 7
│   ├── test_single_url.py          ← Step 6 inference
│   ├── test_fusion.py              ← Step 7 inference
│   ├── zero_day_eval.py            ← Step 8
│   └── api_server.py               ← Step 9 (Flask API)
├── models/
│   ├── url_baseline.pkl
│   ├── html_baseline.pkl
│   ├── image_baseline.pkl
│   ├── fusion_model.pkl
│   ├── baseline_metrics.json
│   └── fusion_metrics.json
└── extension/
    └── index.html                  ← Web UI
```

---

## ⚠️ Dataset Notes & Bias Prevention

- Folder names **must NEVER be used as model features** — severe label leakage:
  - Legit folders = real domain names (`www.google.com`) → obvious class 0 pattern
  - Phishing folders = hash IDs (`0002_bafybei...`) → obvious class 1 pattern
- Folder names are **only used as file paths** to locate `metadata`, `page.html`, `screenshot`
- The **URL string comes from the metadata file**, not the folder name
- `sample_id` in manifest must be a **neutral integer**, never the folder name

---

## 🔧 Current State

- [x] SRS drafted
- [x] Architecture diagram completed
- [x] Legitimate dataset collected (340 samples)
- [x] Phishing dataset available (650 samples)
- [x] Project folder structure created
- [x] Python environment set up (`fedra` conda env)
- [x] Git repo initialized → https://github.com/Gauthamkv14/FEDrA.git
- [x] `manifest.csv` built
- [x] EDA completed
- [x] Feature extraction pipeline implemented
- [x] Training pipeline baselines trained and single URL inference working
- [x] Fusion model trained and inference script implemented
- [x] Zero-day evaluation complete (2026-03-08) — 25 OpenPhish URLs, 60% detection, 5.71s avg latency
- [x] Extension framework rewritten and debug logging added (2026-03-12)
- [x] Implemented dead site detection using `webNavigation.onErrorOccurred` (2026-03-12)

---

## 🚀 Immediate Next Steps (Ordered)

### ✅ Step 1 — Environment (DONE)
```bash
conda activate fedra
# packages: pandas, numpy, scikit-learn, torch, torchvision,
#           pillow, beautifulsoup4, shap, tqdm, jupyter, flask, flask-cors
```

---

### ✅ Step 2 — Build manifest.csv (DONE)
**Script:** `scripts/build_manifest.py`

Output schema:
```
sample_id | label | url | html_path | screenshot_path
```

---

### ✅ Step 3 — EDA (DONE)
**Notebook:** `notebooks/eda.ipynb`

---

### 🔲 Step 4 — Feature Schema (No code — decision needed)

**URL Features:**
- URL length, number of subdomains, presence of IP address
- Use of HTTPS, number of special chars (`@`, `-`, `//`)
- Domain token entropy, TLD type
- Count of URL parameters
- Presence of suspicious parameters (login, redirect, verify, secure)

**HTML Features:**
- Number of forms, inputs, iframes
- Number of external links/scripts
- Presence of password fields, meta redirects
- Script-to-content ratio, favicon mismatch flag
- Number of external script tags
- Presence of auto-submission forms
- Ratio of input fields to submit buttons
- Number of unique external domains linked

**Visual:** Raw screenshot → CNN encoder (no manual features)

---

### ✅ Step 5 — Feature Extraction Pipeline (DONE)
Three scripts reading from `manifest.csv`, writing to `Dataset/features/`:

| Script | Input | Output |
|--------|-------|--------|
| `extract_url_features.py` | URL string from manifest | `url_features.csv` |
| `extract_html_features.py` | `page.html` via BeautifulSoup | `html_features.csv` |
| `extract_visual_embeddings.py` | screenshot via frozen CNN | `visual_embeddings.npy` |

---

### ✅ Step 6 — Per-Modality Baseline & Inference Testing (DONE)
Train and evaluate each modality independently before fusion:
- URL-only MLP → Precision, Recall, F1, AUC
- HTML-only MLP → same metrics
- Image-only CNN → same metrics

Tested using inference script `scripts/test_single_url.py` which loads all 3 `.pkl` baseline models.

---

### ✅ Step 7 — Fusion Model (DONE)
Concatenate all three embeddings → MLP classifier → sigmoid output.

**Scripts:** `scripts/train_fusion.py` (training), `scripts/test_fusion.py` (inference)
**Model:** `models/fusion_model.pkl`

Fusion metrics (198 test samples):
- Precision: 0.9219 | Recall: 0.9077 | F1: 0.9147 | AUC: 0.9292

---

### ✅ Step 8 — Zero-Day Evaluation (DONE 2026-03-08)
25 live URLs from OpenPhish feed:
- Detection rate: 60% (15/25)
- Avg latency: 5.71s/URL
- Results: `notebooks/zero_day_results.csv`

---

### 🔲 Step 9 — Flask API + Web UI
**Scripts:** `scripts/api_server.py` + `extension/index.html`

- Flask backend: POST /analyze → JSON response with prediction + reasons
- Standalone HTML frontend with dark cybersecurity theme
- Run server: `/opt/anaconda3/envs/fedra/bin/python scripts/api_server.py`
- Open: `extension/index.html` in Chrome

---

### 🔲 Step 10 — Export
Distill and export model to ONNX or TensorFlow.js for browser deployment.

---

## 🛠️ Unresolved Decisions

| Item                        | Options                              | Priority |
|-----------------------------|--------------------------------------|----------|
| Handle class imbalance      | Weighted loss vs SMOTE vs oversample | High     |
| Embedding dimensions        | 64 or 128-dim per modality           | Medium   |
| Browser inference backend   | TensorFlow.js vs ONNX Runtime Web    | Later    |
| Performance thresholds      | Latency < 1s, Memory < 100MB?        | Later    |

---

## ⚠️ Key Constraints (Non-Negotiable)

- ❌ No external API calls during runtime
- ❌ No DNS-based features in Phase-1
- ❌ No cloud-based inference
- ❌ Folder names must never be model features
- ✅ Explainability required (SHAP + Grad-CAM)
- ✅ Zero-day detection capability
- ✅ Lightweight for in-browser execution
- ✅ Keep Phase-1 lean — no overengineering

---

## 🌿 Git Workflow

**Branches:**
- `main` — stable, protected. Only updated via PR from `development`
- `development` — active working branch for everyone

**Everyone's daily workflow:**
```bash
conda activate fedra
git checkout development
git pull origin development
# ... do work ...
git add .
git commit -m "feat/fix/docs: describe what you did"
git push origin development
```

**Merging to main (only repo owner — @Gauthamkv14):**
1. Go to https://github.com/Gauthamkv14/FEDrA/compare/main...development
2. Open Pull Request
3. Review changes
4. Merge

**Commit message conventions:**
- `feat:` new feature or script
- `fix:` bug fix
- `docs:` CLAUDE.md or readme updates
- `data:` dataset or manifest changes
- `model:` training or evaluation changes

---

## 📦 Data Status

| Dataset    | Samples | Folder Naming        | Label | Status       |
|------------|---------|----------------------|-------|--------------|
| Phishing   | 650     | `0001_<hash>.domain` | 1     | ✅ Available |
| Legitimate | 340     | `www.domain.com`     | 0     | ✅ Available |
| Combined   | 990     | Via manifest.csv     | —     | ✅ Available |

**Class imbalance:** 66% phishing / 34% legit → addressed via class weighting in training.

---

## 🔮 Phase-2 (Out of Scope for Now)

- Federated learning for privacy-preserving updates
- DNS / SSL certificate analysis
- Active learning loop
- Infrastructure fingerprinting (CDN, hosting)