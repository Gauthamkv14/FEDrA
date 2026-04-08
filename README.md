# FEDrA — Federated Detection of Ransomware & Phishing

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c)
![ONNX](https://img.shields.io/badge/ONNX-Standard-00599c)
![Manifest V3](https://img.shields.io/badge/Manifest-V3-success)
![License](https://img.shields.io/badge/License-MIT-yellow)

Real-time, client-side, explainable multimodal browser extension for zero-day phishing detection without blacklists.

---

## 📖 Table of Contents
- [Overview](#-overview)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Setup Instructions](#-setup-instructions)
- [How to Run](#-how-to-run)
- [Model Performance](#-model-performance)
- [Dataset](#-dataset)
- [Tech Stack](#-tech-stack)
- [Team](#-team)
- [License](#-license)

---

## 🔍 Overview
FEDrA (**FE**derated **D**etection of **Ra**nsomware & Phishing) is a Phase-1 project focused on building an explainable, multimodal browser extension for real-time phishing detection. 

Unlike traditional security solutions that rely on reactive blacklists, FEDrA analyzes the structural, content, and visual features of a website locally in the browser to detect **zero-day attacks**. By performing inference client-side, it ensures user privacy and reduces latency.

Key Features:
- **Zero-Day Detection:** No reliance on blacklists or known URL databases.
- **Client-Side Inference:** Runs locally; no server API calls during detection.
- **Explainability:** Provides human-readable reasons for detection using SHAP (for URL/HTML) and Grad-CAM (for visuals).

---

## 🏗️ Architecture
FEDrA uses a late-fusion pipeline that combines three distinct modalities to make a final prediction.

```text
      +------------+
      |  URL String | ----> [ Feature Extraction ] ----+
      +------------+                                   |
                                                       |
      +------------+                                   |
      |  HTML DOM  | ----> [ Feature Extraction ] ----+-----> [ Fusion MLP ] ----> [ Verdict ]
      +------------+                                   |        (Concat)
                                                       |
      +------------+                                   |
      | Screenshot | ----> [ Visual Embedding  ] ----+
      +------------+         (MobileNetV2)
```

The pipeline consists of:
1. **URL Modality:** MLP/Logistic Regression on structural features (length, TLD, entropy).
2. **HTML Modality:** MLP/Logistic Regression on DOM features (forms, inputs, external links).
3. **Visual Modality:** CNN encoder (MobileNetV2) for layout and visual consistency analysis.
4. **Fusion Layer:** A Feature-level concatenation followed by an MLP classifier.

---

## 📁 Project Structure
```text
FEDrA/
├── CLAUDE.md                ← Project context and ground rules
├── Dataset/                 ← Raw data and extracted features
│   ├── legit_dataset/       # Legitimate website snapshots
│   ├── phised_dataset/      # Phishing website snapshots
│   ├── features/            # Extracted feature CSVs/NPYs
│   └── manifest.csv         # Global sample index
├── models/                  ← Saved baseline and fusion models (.pkl)
├── scripts/                 ← Training, extraction, and inference scripts
├── extension/               ← Browser extension source (Manifest V3)
├── notebooks/               ← EDA and research notebooks
└── legitamate.py            ← Initial dataset collection script
```

---

## 🛠️ Setup Instructions

### 1. Create Conda Environment
```bash
conda create -n fedra python=3.10
conda activate fedra
```

### 2. Install Dependencies
```bash
pip install pandas numpy scikit-learn torch torchvision pillow \
            beautifulsoup4 shap tqdm jupyter joblib selenium
```

*Note: Selenium requires a compatible ChromeDriver installed on your system PATH.*

---

## 🚀 How to Run
To test the fusion detection pipeline on a specific URL:

```bash
python scripts/test_fusion.py --url "https://www.google.com"
```

**Example Output:**
```text
  VERDICT  [Mode: Fusion MLP]
  =============================================================
  Modality             Weight      Dims  Status
  -------------------------------------------------------
  URL                    0.4        26  live features
  HTML                   0.3        12  live HTML
  Visual                 0.3      1280  live screenshot
  -------------------------------------------------------
  Fusion phishing probability : 0.02%
  Prediction                  : ✅  LEGITIMATE
```

---

## 📊 Model Performance
Results from Step 6 (Baseline Training) of the pipeline:

| Modality | AUC Score | Accuracy | F1 Score |
|----------|-----------|----------|----------|
| **URL**  | **0.9964**| 0.9798   | 0.9845   |
| **HTML** | **0.8649**| 0.8485   | 0.8855   |
| **Image**| **0.8196**| 0.7727   | 0.8235   |

---

## 📂 Dataset
The project utilizes a combined dataset of **990 samples**:
- **340 Legitimate** (Legit)
- **650 Phishing** (Phish)

**Class Imbalance:** 66% Phishing / 34% Legit. This is addressed during training via class weighting (`class_weight='balanced'`) to ensure robust detection across both classes.

---

## 💻 Tech Stack
- **Languages:** Python 3.10, JavaScript
- **ML Frameworks:** PyTorch, Scikit-learn
- **Feature Extraction:** BeautifulSoup4, Selenium
- **Visual Encoding:** MobileNetV2 (via Torchvision)
- **Deployment:** Chrome Manifest V3, ONNX
- **Explainability:** SHAP, Grad-CAM

---

## 👥 Team
- **Gautham K V** (@Gauthamkv14)
- **Gauri S**(@gauris8)
- **A Soundara Lahari**(@sounds034)
- **Bharath Kumar B D**(@BharathKumarBD)
---

## 📜 License
This project is licensed under the **MIT License** 
