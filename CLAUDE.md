# FEDrA – Project Context (Phase-1)
**Last Updated:** 2026-03-03  
**Status:** Implementation Prep → Dataset Pipeline  

> **FEDrA** = **FE**derated **D**etection of **Ra**nsomware & Phishing (Phase-1 focuses on phishing)  
> Phase-1 Goal: Build an **explainable**, **multimodal browser extension** for real-time zero-day phishing detection.

---

## 🎯 Core Objective

Detect phishing websites in real time using a browser extension (Manifest V3) that analyzes:
- **URL structure**
- **HTML/DOM content**
- **Visual layout (screenshot)**

Requirements:
- Detect **zero-day attacks** (no blacklists)
- Provide **human-readable explanations** (SHAP + Grad-CAM)
- Run **locally** in the browser (no server API calls during inference)
- Be **lightweight** and feasible on modest hardware

---

## ✅ Finalized Decisions

| Component              | Decision                                                         |
|------------------------|------------------------------------------------------------------|
| Deployment             | Browser Extension (Chrome/Edge, Manifest V3)                    |
| Inference Mode         | Real-time, client-side                                           |
| Modalities             | URL + HTML/DOM + Screenshot (Visual)                            |
| Model Architecture     | MLP (URL) + MLP (HTML) + CNN (Image) → Concat → MLP Classifier  |
| Visual Encoder         | ResNet18 / MobileNetV2 (candidates)                             |
| Fusion Strategy        | Feature-level concatenation                                      |
| Explainability         | SHAP (URL/HTML), Grad-CAM (Image)                               |
| Training Approach      | Offline training + model distillation                            |
| Export Format          | ONNX or TensorFlow.js                                            |
| Constraints            | No DNS, no paid APIs, no server inference                        |

---

## 📁 Current Project Structure

```
FEDrA/
├── CLAUDE.md
└── Dataset/
    ├── legit_dataset/        # 300 folders — named by domain (e.g. www.google.com)
    │   └── <domain>/
    │       ├── metadata      # URL, timestamp, etc.
    │       ├── page.html     # Raw HTML
    │       └── screenshot    # Visual capture
    └── phised_dataset/       # 860 folders — numbered IDs (e.g. 0002_bafybei...link)
        └── <id>/
            ├── metadata
            ├── page.html
            └── screenshot
```

**Total samples before merge:** 1,160 (300 legit + 860 phishing)  
**Label convention:** `0` = legitimate, `1` = phishing

---

## ⚠️ Dataset Notes & Bias Prevention

- Folder names **must never be used as model features** — they would introduce severe label leakage:
  - Legit folders use real domain names (`www.google.com`) → obvious pattern for class 0
  - Phishing folders use opaque hash-like IDs → obvious pattern for class 1
- Folder names are used **only as sample identifiers** during dataset construction
- The actual features extracted (URL string from metadata, HTML content, screenshot pixels) are what the model trains on
- URL string extracted from **metadata file** — not from folder name

---

## 🔧 Current State

- [x] SRS drafted  
- [x] High-level architecture diagram completed  
- [x] Legitimate website dataset collected (300 samples) ✅  
- [x] Phishing dataset available (860 samples) ✅  
- [x] Project folder structure created (`FEDrA/Dataset/...`)  
- [ ] Python/ML environment not yet set up  
- [ ] Datasets not yet merged or preprocessed  
- [ ] Feature extraction pipeline not implemented  
- [ ] Training pipeline not built  
- [ ] Baseline model undecided  

➡️ **We are now at the threshold of implementation — environment setup is the immediate next step.**

---

## ⚠️ Key Constraints (Non-Negotiable)

- ❌ No external API calls during runtime
- ❌ No DNS-based features in Phase-1
- ❌ No cloud-based inference
- ❌ Folder names must never be model features (bias/leakage risk)
- ✅ Must support explainability (no black-box models)
- ✅ Must detect zero-day threats (holdout unseen sites)
- ✅ Lightweight enough for in-browser execution
- ✅ Avoid overengineering — keep MVP lean

---

## 🚀 Immediate Next Steps (In Order)

### Step 1 — Set Up Python Environment
```bash
# Install Miniconda (recommended for ML projects)
# Then create a dedicated environment:
conda create -n fedra python=3.10
conda activate fedra
pip install pandas numpy scikit-learn torch torchvision pillow beautifulsoup4 shap tqdm jupyter
```
> Alternatives: plain `venv` + pip, or Google Colab for prototyping if local hardware is limited.

---

### Step 2 — Build the Dataset Manifest (CSV Index)
Write a script (`scripts/build_manifest.py`) that:
1. Iterates over `legit_dataset/` → assigns `label = 0`
2. Iterates over `phised_dataset/` → assigns `label = 1`
3. For each sample folder, reads the **metadata file** to extract the URL string
4. Records: `sample_id`, `label`, `url`, `html_path`, `screenshot_path`
5. Outputs: `Dataset/manifest.csv`

**Critical:** `sample_id` must be a neutral integer index — **never** the folder name.

Expected output schema:
```
sample_id | label | url | html_path | screenshot_path
0         | 0     | https://www.google.com | .../page.html | .../screenshot.png
1         | 1     | https://... | .../page.html | .../screenshot.png
```

---

### Step 3 — Exploratory Data Analysis (EDA)
Using `notebooks/eda.ipynb`:
- Class distribution (how balanced is 300 vs 860?)
- Check for missing/corrupt files (missing HTML, broken screenshots)
- URL length distribution per class
- HTML file size distribution per class
- Screenshot resolution consistency

> **Class imbalance note:** 860 phishing vs 300 legit (~74/26 split). Plan to use weighted loss or oversampling (SMOTE on features, or duplicate legit samples).

---

### Step 4 — Define Feature Extraction Schema
Before building the pipeline, lock in exactly what features are extracted per modality:

**URL Features (from metadata, not folder name):**
- URL length
- Number of subdomains
- Presence of IP address
- Use of HTTPS
- Number of special characters (`@`, `-`, `//`, etc.)
- Domain token entropy
- TLD type

**HTML Features:**
- Number of forms, inputs, iframes
- Number of external links / scripts
- Presence of password fields
- Meta redirect presence
- Script-to-content ratio
- Favicon mismatch flag

**Visual (Screenshot):**
- Passed as raw image to CNN encoder
- No manual feature engineering needed

---

### Step 5 — Build Feature Extraction Pipeline
Scripts to build:
- `scripts/extract_url_features.py` → reads URL from metadata → outputs feature vector
- `scripts/extract_html_features.py` → parses `page.html` via BeautifulSoup → outputs feature vector
- `scripts/extract_visual_embeddings.py` → loads screenshot → passes through frozen pretrained CNN → outputs embedding

All scripts read from `manifest.csv` and write to `Dataset/features/`:
```
Dataset/features/
├── url_features.csv
├── html_features.csv
└── visual_embeddings.npy
```

---

### Step 6 — Train Baseline Models (Per Modality)
Before fusion, validate each modality independently:
- URL-only MLP → F1, AUC
- HTML-only MLP → F1, AUC
- Image-only CNN → F1, AUC

This tells you which modality contributes most and catches bugs early.

---

### Step 7 — Train Fusion Model
Concatenate embeddings from all three modalities → feed into MLP classifier with sigmoid output.

---

### Step 8 — Zero-Day Evaluation
Hold out a set of **never-seen phishing URLs** (recent, post-collection) to test generalization beyond the training distribution.

---

## 📦 Data Status

| Dataset     | Samples | Folder Naming             | Label | Status       |
|-------------|---------|---------------------------|-------|--------------|
| Phishing    | 860     | `0001_<hash>.domain`      | 1     | ✅ Available |
| Legitimate  | 300     | `www.domain.com`          | 0     | ✅ Available |
| Combined    | 1,160   | Unified via manifest.csv  | —     | ❌ Pending   |

---

## 🛠️ Unresolved Technical Decisions

| Item                          | Options / Notes                              | Priority |
|-------------------------------|----------------------------------------------|----------|
| Handle class imbalance        | Weighted loss vs SMOTE vs oversample legit   | High     |
| Embedding dimensions          | 64 or 128-dim per modality                   | Medium   |
| Zero-day evaluation protocol  | Holdout set of post-collection phishing URLs | High     |
| Inference backend (browser)   | TensorFlow.js vs ONNX Runtime Web            | Later    |
| Performance thresholds        | Latency < 1s, Memory < 100MB?                | Later    |
| Training framework            | PyTorch (recommended) vs sklearn             | High     |

---

## ✅ Phase-1 Must-Haves

- [ ] Python environment set up  
- [ ] `manifest.csv` built (without folder names as features)  
- [ ] EDA completed  
- [ ] Feature schema finalized  
- [ ] URL + HTML feature extraction scripts  
- [ ] Visual embedding extractor  
- [ ] Per-modality baseline evaluation  
- [ ] Fusion classifier trained  
- [ ] Explainability outputs (SHAP values, Grad-CAM heatmap)  
- [ ] Evaluation: Precision, Recall, F1, AUC  
- [ ] Model exported (ONNX or tfjs)  

## 🔮 Phase-2 Nice-to-Haves

- Federated learning for privacy-preserving updates  
- DNS / SSL certificate analysis  
- Active learning loop  
- Infrastructure fingerprinting (CDN, hosting)  

---

## 🔄 Runtime Flow (Browser Extension)

1. User visits a site  
2. Capture: URL, DOM tree, screenshot  
3. Extract engineered features (structured)  
4. Encode each modality → embeddings  
5. Fuse → classify → sigmoid probability  
6. Generate explanation (SHAP + Grad-CAM overlay)  
7. Show warning if probability > threshold (e.g., 0.85)

---

## 📌 Recommended Folder Structure (Going Forward)

```
FEDrA/
├── CLAUDE.md
├── Dataset/
│   ├── legit_dataset/
│   ├── phised_dataset/
│   ├── manifest.csv            ← to be created in Step 2
│   └── features/               ← to be created in Step 5
│       ├── url_features.csv
│       ├── html_features.csv
│       └── visual_embeddings.npy
├── notebooks/
│   └── eda.ipynb
├── scripts/
│   ├── build_manifest.py
│   ├── extract_url_features.py
│   ├── extract_html_features.py
│   └── extract_visual_embeddings.py
├── models/
└── extension/
```