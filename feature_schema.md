# FEDrA — Feature Schema Specification

This document provides a comprehensive audit of the feature representations across all three modalities (URL, HTML, Visual) in the FEDrA pipeline.

---

## 1. URL FEATURE PIPELINE (CANONICAL IMPLEMENTATION)

### ✅ STATUS: RESOLVED — CANONICAL 22-DIMENSIONAL SCHEMA IMPLEMENTED
*The semantic mismatch between training and inference schemas has been fully resolved. A single centralized module (`scripts/url_features.py`) defines the canonical 22-dimensional lexical feature schema used identically across batch extraction, training, CLI inference, Flask backend, and zero-day benchmarking.*

---

### A. Batch Feature Extraction (`scripts/extract_url_features.py`)
Generates `Dataset/features/url_features.csv` from `Dataset/manifest.csv`.

**Columns (11 features + `sample_id`):**
1. `sample_id` (Index identifier)
2. `url_len` (Integer length of URL string)
3. `num_subdomains` (Integer count of subdomains via `tldextract`)
4. `has_ip` (Binary: 1 if IPv4 address in domain, else 0)
5. `is_https` (Binary: 1 if scheme is https, else 0)
6. `count_at` (Integer count of `@` characters)
7. `count_dash` (Integer count of `-` characters)
8. `count_double_slash` (Integer count of `//` occurrences excluding scheme)
9. `domain_entropy` (Float: Shannon entropy of domain + suffix token)
10. `tld_type` (String: Top-Level Domain string, e.g., `'com'`, `'org'`, `'xyz'`)
11. `num_params` (Integer count of query parameters)
12. `has_suspicious_params` (Binary: 1 if query keys match `login`, `redirect`, `verify`, `secure`, else 0)

---

### B. Training Preprocessing & Dimensionality (`scripts/train_baselines.py` & `scripts/train_fusion.py`)
During baseline and fusion model training:
```python
url_df = pd.read_csv("Dataset/features/url_features.csv")
url_df = pd.get_dummies(url_df, drop_first=True) # One-hot encodes 'tld_type'
X_url = url_df.drop(columns=["sample_id"]).values
```
- Because `tld_type` contained 79 unique suffix values across the 990 dataset samples, `pd.get_dummies()` expands `tld_type` into **79 binary dummy columns**.
- Resulting training matrix dimensionality: **89 features** (10 numerical features + 79 dummy TLD columns).
- Training scaler: `StandardScaler().fit(X_url_train)` (expects exactly 89 features).
- `models/url_baseline.pkl` (`LogisticRegression`) expects **89 features**.
- `models/fusion_model.pkl` (`MLPClassifier`) expects $89 + 12 + 1280 = \mathbf{1381\text{ features}}$.

---

### C. Live Inference Extraction (`scripts/test_fusion.py`, `scripts/api_server.py`, `scripts/zero_day_eval.py`, `scripts/test_single_url.py`)
Inference scripts do **not** run `pd.get_dummies()`. Instead, `extract_url_features(url, fetch_error_type)` constructs a **26-dimensional** vector:

- **Core 10 features (indices 0..9):**
  `[url_len, num_subdomains, is_ip, is_https, num_at, num_dash, num_double_slash, domain_entropy, num_params, has_susp_params]`
- **Extended 12 phishing-pattern features (indices 10..21):**
  `[brand_in_domain, brand_in_subdomain, has_typosquat, has_punycode, excessive_subdomains, suspicious_tld, free_hosting, is_shortener, excessive_hyphens, has_nonstandard_port, high_entropy, long_url]`
- **Fetch-failure signal features (indices 22..25):**
  `[is_unresolvable, is_ssl_error, is_refused, is_fetch_failed]`

Total live extracted vector width = **26 features**.

---

### D. Inference Padding & Alignment Mechanism
When live inference runs, the code attempts to adapt the 26-dim vector to the 89-dim scaler using `_align()`:
```python
def _align(X: np.ndarray, expected: int) -> np.ndarray:
    cur = X.shape[1]
    if cur < expected:
        return np.hstack([X, np.zeros((1, expected - cur))]) # 26 + 63 zeros = 89
    return X[:, :expected]
```

#### Why This Is a Critical Bug:
1. Columns 0..9 match the core training features.
2. Columns 10..21 (the 12 extended phishing heuristic features) and columns 22..25 (the 4 fetch failure features) are placed into positions 10..25.
3. In the fitted `StandardScaler` and `LogisticRegression` / `MLPClassifier`, columns 10..25 are actually the one-hot columns for specific training TLDs (e.g., `.de`, `.edu`, `.es`, `.fr`, etc.).
4. The models evaluate these 16 live security features using the weights learned for arbitrary country-code/educational TLDs, while positions 26..88 are zero-padded.

---

### E. Code Locations Responsible for URL Schema

| Location | Operation | Dimension | Schema Description |
|---|---|---|---|
| `scripts/extract_url_features.py:83-95` | Batch Extraction | 11 features | 10 numerical + 1 categorical `tld_type` |
| `scripts/train_baselines.py:94-97` | Baseline Training | 89 features | 10 numerical + 79 `tld_type` one-hot dummies |
| `scripts/train_fusion.py:76-78` | Fusion Training | 89 features | 10 numerical + 79 `tld_type` one-hot dummies |
| `scripts/test_single_url.py:247-335` | Single URL CLI | 26 features | 10 core + 12 extended + 4 fetch failure |
| `scripts/test_fusion.py:207-265` | Fusion CLI | 26 features | 10 core + 12 extended + 4 fetch failure |
| `scripts/zero_day_eval.py:236-275` | Zero-Day Eval | 26 features | 10 core + 12 extended + 4 fetch failure |
| `scripts/api_server.py:194-237` | Flask Server | 26 features | 10 core + 12 extended + 4 fetch failure |

---

## 2. HTML FEATURE PIPELINE (HIGH-LEVEL OVERVIEW)

### Status: `IMPLEMENTED & ALIGNED`

- **Extraction Script:** `scripts/extract_html_features.py`
- **Output File:** `Dataset/features/html_features.csv`
- **Parsing Library:** `BeautifulSoup(html_content, "html.parser")`
- **Fixed Dimension:** **12 features** (Identical in batch extraction, training, and live inference).

**Extracted Features:**
1. `num_forms` — Integer count of `<form>` elements.
2. `num_inputs` — Integer count of `<input>` elements.
3. `num_iframes` — Integer count of `<iframe>` elements.
4. `num_ext_links` — Integer count of `<a>` links pointing to an external domain.
5. `num_ext_scripts` — Integer count of `<script src="...">` loaded from an external domain.
6. `has_password_field` — Binary: 1 if `<input type="password">` present, else 0.
7. `has_meta_redirect` — Binary: 1 if `<meta http-equiv="refresh">` present, else 0.
8. `script_content_ratio` — Float: Length of inline/tag script characters / total raw HTML length.
9. `favicon_mismatch` — Binary: 1 if `<link rel="icon">` domain differs from page domain, else 0.
10. `has_auto_submit` — Binary: 1 if form exists and script contains `submit()`, else 0.
11. `input_submit_ratio` — Float: `num_inputs` / max(1, count of submit buttons/inputs).
12. `num_unique_ext_domains` — Integer count of distinct external FQDNs referenced.

---

## 3. VISUAL FEATURE PIPELINE (HIGH-LEVEL OVERVIEW)

### Status: `IMPLEMENTED & ALIGNED`

- **Extraction Script:** `scripts/extract_visual_embeddings.py`
- **Output File:** `Dataset/features/visual_embeddings.npy` (shape: `(990, 1280)`, dtype: `float32`)
- **Model:** PyTorch `torchvision.models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1).features`
- **Model State:** Frozen feature extractor (Pretrained on ImageNet-1k, no fine-tuning weights updated).
- **Fixed Dimension:** **1280 features** (Identical in batch extraction, training, and live inference).

**Preprocessing Pipeline:**
1. Screenshot loaded via PIL and converted to RGB.
2. `transforms.Resize(256)`
3. `transforms.CenterCrop(224)`
4. `transforms.ToTensor()`
5. `transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`
6. Forward pass through `.features` producing tensor `[1, 1280, 7, 7]`.
7. Global Average Pooling: `.mean([2, 3])` $\rightarrow$ `1280-dimensional` embedding vector.
8. If screenshot missing or corrupt, returns zero vector of shape `(1, 1280)`.

---

## 4. FUSION CONCATENATION SCHEMA

- **URL Vector:** 89 dimensions $\times$ Scaler $\times$ Weight `0.4`
- **HTML Vector:** 12 dimensions $\times$ Scaler $\times$ Weight `0.3`
- **Visual Vector:** 1280 dimensions $\times$ Scaler $\times$ Weight `0.3`
- **Total Fused Input Dimension:** $89 + 12 + 1280 = \mathbf{1381\text{ dimensions}}$
- **Classifier:** `sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(256, 128, 64), activation='relu', solver='adam')`
- **Output:** Sigmoid probability $\in [0.0, 1.0]$.

---

## URL Schema Forensics — Step 1

### A. Current Training Schema (Verified from Code & Artifacts)
- **Source Script:** `scripts/extract_url_features.py` + `scripts/train_baselines.py` / `scripts/train_fusion.py`
- **Feature Count (Raw Extraction):** 11 features + `sample_id`
- **Categorical Handling:** `pd.get_dummies(url_df, drop_first=True)`
- **Feature Count (Training Matrix):** **89 features**
- **Exact Column Ordering (Indices 0 to 88):**
  - Index 0: `url_len` (float/int)
  - Index 1: `num_subdomains` (float/int)
  - Index 2: `has_ip` (binary 0/1)
  - Index 3: `is_https` (binary 0/1)
  - Index 4: `count_at` (int count)
  - Index 5: `count_dash` (int count)
  - Index 6: `count_double_slash` (int count)
  - Index 7: `domain_entropy` (float)
  - Index 8: `num_params` (int count)
  - Index 9: `has_suspicious_params` (binary 0/1)
  - Indices 10–88 (79 dummy columns):
    `tld_type_ac.uk`, `tld_type_app`, `tld_type_ba`, `tld_type_bet`, `tld_type_ca`, `tld_type_cat`, `tld_type_cc`, `tld_type_cf`, `tld_type_cl`, `tld_type_cn`, `tld_type_co`, `tld_type_co.id`, `tld_type_co.in`, `tld_type_co.jp`, `tld_type_co.kr`, `tld_type_co.nz`, `tld_type_co.th`, `tld_type_co.uk`, `tld_type_co.za`, `tld_type_com`, `tld_type_com.ar`, `tld_type_com.au`, `tld_type_com.br`, `tld_type_com.cn`, `tld_type_com.co`, `tld_type_com.do`, `tld_type_com.eg`, `tld_type_com.hk`, `tld_type_com.mx`, `tld_type_com.my`, `tld_type_com.ng`, `tld_type_com.ph`, `tld_type_com.pk`, `tld_type_com.sa`, `tld_type_com.sg`, `tld_type_com.tr`, `tld_type_com.tw`, `tld_type_com.ua`, `tld_type_com.uy`, `tld_type_com.vn`, `tld_type_cz`, `tld_type_de`, `tld_type_dev`, `tld_type_edu`, `tld_type_es`, `tld_type_eu`, `tld_type_fr`, `tld_type_gov`, `tld_type_gr`, `tld_type_hk`, `tld_type_hu`, `tld_type_id`, `tld_type_in`, `tld_type_info`, `tld_type_io`, `tld_type_ir`, `tld_type_it`, `tld_type_jp`, `tld_type_kr`, `tld_type_link`, `tld_type_me`, `tld_type_ms`, `tld_type_mx`, `tld_type_my`, `tld_type_net`, `tld_type_net.au`, `tld_type_net.br`, `tld_type_net.in`, `tld_type_net.tr`, `tld_type_nl`, `tld_type_online`, `tld_type_org`, `tld_type_page`, `tld_type_pl`, `tld_type_ro`, `tld_type_ru`, `tld_type_se`, `tld_type_site`, `tld_type_top`, `tld_type_tr`, `tld_type_tv`, `tld_type_tw`, `tld_type_ua`, `tld_type_uk`, `tld_type_us`, `tld_type_vn`, `tld_type_wiki`, `tld_type_ws`, `tld_type_xyz`.
- **Scaling:** `StandardScaler` fitted on 89 columns.

---

### B. Current Live Inference Schema (Verified from Code)
- **Source Scripts:** `scripts/test_single_url.py`, `scripts/test_fusion.py`, `scripts/zero_day_eval.py`, `scripts/api_server.py`
- **Feature Count (Raw Extraction):** **26 features**
- **Exact Column Ordering (Indices 0 to 25):**
  - Core features (0–9):
    0. `url_len`
    1. `num_subdomains`
    2. `is_ip`
    3. `is_https`
    4. `num_at`
    5. `num_dash`
    6. `num_double_slash`
    7. `domain_entropy`
    8. `num_params`
    9. `has_susp_params`
  - Extended heuristic features (10–21):
    10. `brand_in_domain`
    11. `brand_in_subdomain`
    12. `has_typosquat`
    13. `has_punycode`
    14. `excessive_subdomains`
    15. `suspicious_tld`
    16. `free_hosting`
    17. `is_shortener`
    18. `excessive_hyphens`
    19. `has_nonstandard_port`
    20. `high_entropy`
    21. `long_url`
  - Fetch-failure features (22–25):
    22. `is_unresolvable`
    23. `is_ssl_error`
    24. `is_refused`
    25. `is_fetch_failed`
- **Padding Behavior:** `_align()` appends 63 zeros to make a 26 + 63 = 89-dimensional array before passing to the 89-dim `StandardScaler`.

---

### C. Feature-by-Feature Comparison Matrix

| # | Feature Name | In Batch Training? | In Live Inference? | Runtime / Network Dependency | In-Browser Local Compatible? | Recommendation for Canonical ML Schema | Rationale |
|---|---|---|---|---|---|---|---|
| 1 | `url_len` | Yes (Index 0) | Yes (Index 0) | None (Pure string parsing) | Yes | **RETAIN** | Fundamental lexical feature indicating URL obfuscation. |
| 2 | `num_subdomains` | Yes (Index 1) | Yes (Index 1) | None (TLD / domain parse) | Yes | **RETAIN** | Phishing frequently nests subdomains. |
| 3 | `has_ip` / `is_ip` | Yes (Index 2) | Yes (Index 2) | None (Regex IPv4 match) | Yes | **RETAIN** | Direct IP addressing is a strong phishing indicator. |
| 4 | `is_https` | Yes (Index 3) | Yes (Index 3) | None (Scheme check) | Yes | **RETAIN** | Clear protocol signal. |
| 5 | `count_at` / `num_at` | Yes (Index 4) | Yes (Index 4) | None (Char count `@`) | Yes | **RETAIN** | `@` in URL authority is a classic credential-stealing syntax. |
| 6 | `count_dash` / `num_dash` | Yes (Index 5) | Yes (Index 5) | None (Char count `-`) | Yes | **RETAIN** | Excessive hyphens indicate domain spoofing. |
| 7 | `count_double_slash` / `num_double_slash` | Yes (Index 6) | Yes (Index 6) | None (Substr count `//`) | Yes | **RETAIN** | Redirect indicator. |
| 8 | `domain_entropy` | Yes (Index 7) | Yes (Index 7) | None (Shannon entropy) | Yes | **RETAIN** | Algorithmic / DGA domain generation detection. |
| 9 | `tld_type` (79 one-hot columns) | Yes (Indices 10–88) | No | None | No (Open vocabulary) | **REMOVE (Replace with `is_suspicious_tld`)** | 79 dummy columns fail on open web (>1500 TLDs) and create index misalignment. |
| 10 | `num_params` | Yes (Index 8) | Yes (Index 8) | None (Query parsing) | Yes | **RETAIN** | Measures query complexity. |
| 11 | `has_suspicious_params` | Yes (Index 9) | Yes (Index 9) | None (Keyword check in query keys) | Yes | **RETAIN** | Detects `login`, `verify`, `secure`, `redirect` query arguments. |
| 12 | `brand_in_domain` | No | Yes (Index 10) | None (List matching on domain) | Yes | **ADD TO TRAINING** | Strong phishing signal (e.g. `paypal-update.xyz`). |
| 13 | `brand_in_subdomain` | No | Yes (Index 11) | None (List matching on subdomain) | Yes | **ADD TO TRAINING** | Strong subdomain abuse signal (e.g. `paypal.evil.com`). |
| 14 | `has_typosquat` | No | Yes (Index 12) | None (Char replacement check) | Yes | **ADD TO TRAINING** | Detects character substitutions (`0` for `o`, `1` for `l`). |
| 15 | `has_punycode` | No | Yes (Index 13) | None (Substring `xn--`) | Yes | **ADD TO TRAINING** | Detects homograph / IDN attacks. |
| 16 | `excessive_subdomains` | No | Yes (Index 14) | None (Threshold `num_subdomains > 3`) | Yes | **ADD TO TRAINING** | Binarized structural flag. |
| 17 | `suspicious_tld` / `is_suspicious_tld` | No | Yes (Index 15) | None (Set membership in 26 risky TLDs) | Yes | **ADD TO TRAINING** | High-generalization replacement for 79 one-hot TLD columns. |
| 18 | `free_hosting` | No | Yes (Index 16) | None (Set membership in 12 free hosters) | Yes | **ADD TO TRAINING** | Phishing heavily abuses free platforms (Netlify, Wix, Weebly). |
| 19 | `is_shortener` | No | Yes (Index 17) | None (Set membership in 9 URL shorteners) | Yes | **ADD TO TRAINING** | URL shorteners mask target destination. |
| 20 | `excessive_hyphens` | No | Yes (Index 18) | None (Threshold `domain.count("-") >= 3`) | Yes | **ADD TO TRAINING** | Binarized domain hyphen flag. |
| 21 | `has_nonstandard_port` | No | Yes (Index 19) | None (Port check $\ne 80, 443, 8080$) | Yes | **ADD TO TRAINING** | Phishing sites occasionally host on custom ports. |
| 22 | `high_entropy` | No | Yes (Index 20) | None (Threshold `domain_entropy > 3.8`) | Yes | **ADD TO TRAINING** | Binarized DGA / random string indicator. |
| 23 | `long_url` | No | Yes (Index 21) | None (Threshold `url_len > 100`) | Yes | **ADD TO TRAINING** | Binarized long URL indicator. |
| 24 | `is_unresolvable` | No | Yes (Index 22) | High (DNS / Network error) | Partial (Only post-fetch) | **EXCLUDE FROM ML FEATURE VECTOR** | Static training samples have no fetch errors (all 0). Handle in rule-based fallback logic. |
| 25 | `is_ssl_error` | No | Yes (Index 23) | High (TLS handshake failure) | Partial (Only post-fetch) | **EXCLUDE FROM ML FEATURE VECTOR** | Network transport signal; zero-variance in offline dataset. |
| 26 | `is_refused` | No | Yes (Index 24) | High (TCP connection error) | Partial (Only post-fetch) | **EXCLUDE FROM ML FEATURE VECTOR** | Network transport signal; zero-variance in offline dataset. |
| 27 | `is_fetch_failed` | No | Yes (Index 25) | High (General HTTP/TCP failure) | Partial (Only post-fetch) | **EXCLUDE FROM ML FEATURE VECTOR** | Transport signal, not a static lexical feature of the URL. |

---

### D. Root Cause Analysis
1. **Semantic Slot Aliasing:** In the trained `models/url_baseline.pkl` and `models/fusion_model.pkl`, feature index 10 corresponds to the one-hot column `tld_type_ac.uk`, index 11 to `tld_type_app`, index 12 to `tld_type_ba`, etc.
2. In live inference, index 10 is populated with `brand_in_domain`, index 11 with `brand_in_subdomain`, and index 12 with `has_typosquat`.
3. When `StandardScaler.transform()` runs during live inference, `brand_in_domain` is subtracted by the mean of `tld_type_ac.uk` ($\approx 0.001$) and divided by its standard deviation, then multiplied by the model's learned weight for `tld_type_ac.uk`.
4. The remaining 63 positions (indices 26 through 88) are filled with literal zeros, causing all other TLD weights learned during training to receive a zero input.
5. **Conclusion:** The live model is not evaluating the 12 extended features or 4 failure signals as intended; it is misinterpreting them through arbitrary TLD weights.

---

### E. Candidate Canonical URL Feature Schema (Proposed)

To ensure **100% deterministic consistency** across offline dataset extraction, training, CLI inference, Flask backend, and future client-side in-browser WebAssembly/JS inference:

- **Canonical Dimension:** **22 features** (Pure numerical/boolean lexical representations, zero open-vocabulary categorical variables).
- **Network / Transport Independence:** All 22 features are derived strictly from the URL string itself without requiring network requests, DNS lookups, or browser navigation states.
- **Handling Transport Errors:** Transport/network error states (`unresolvable`, `ssl`, `refused`, `timeout`) remain in application-level fallback logic (e.g., dead-site handler) rather than polluting the stationary ML feature distribution.

#### Ordered Canonical Feature List (Exactly 22 Dimensions):

```
Index  Feature Name            Type      Computation / Definition
-------------------------------------------------------------------------------------------------------------------
 0     url_len                 int       len(url)
 1     num_subdomains          int       Count of non-empty subdomains from parsed domain
 2     has_ip                  binary    1 if IPv4 address in domain regex r'^\d{1,3}(\.\d{1,3}){3}$', else 0
 3     is_https                binary    1 if url.lower().startswith("https"), else 0
 4     count_at                int       url.count('@')
 5     count_dash              int       url.count('-')
 6     count_double_slash      int       max(0, url.count('//') - 1) if "://" in url else url.count('//')
 7     domain_entropy          float     Shannon entropy on f"{domain}.{suffix}"
 8     num_params              int       len(urllib.parse.parse_qsl(parsed.query))
 9     has_suspicious_params   binary    1 if any key in query contains login/redirect/verify/secure, else 0
10     brand_in_domain         binary    1 if any brand keyword in domain and domain != brand, else 0
11     brand_in_subdomain      binary    1 if any brand keyword in subdomain, else 0
12     has_typosquat           binary    1 if domain matches 1-char substitution of brand keyword, else 0
13     has_punycode            binary    1 if "xn--" in hostname.lower(), else 0
14     excessive_subdomains    binary    1 if num_subdomains > 3, else 0
15     is_suspicious_tld       binary    1 if suffix.lower() in _SUSPICIOUS_TLDS, else 0
16     free_hosting            binary    1 if hostname matches free hosting list (000webhost, netlify, etc.), else 0
17     is_shortener            binary    1 if hostname matches known URL shorteners (bit.ly, tinyurl, etc.), else 0
18     excessive_hyphens       binary    1 if domain.count('-') >= 3, else 0
19     has_nonstandard_port    binary    1 if port is not None and port not in (80, 443, 8080), else 0
20     high_entropy            binary    1 if domain_entropy > 3.8, else 0
21     long_url                binary    1 if len(url) > 100, else 0
-------------------------------------------------------------------------------------------------------------------
```

#### Modality Dimensions Under Proposed Canonical Schema:
- **URL Vector:** 22 dimensions (Scaled with fitted `StandardScaler(22)` $\times$ Weight `0.4`)
- **HTML Vector:** 12 dimensions (Scaled with fitted `StandardScaler(12)` $\times$ Weight `0.3`)
- **Visual Vector:** 1280 dimensions (Scaled with fitted `StandardScaler(1280)` $\times$ Weight `0.3`)
- **Total Fused Input Dimension:** $22 + 12 + 1280 = \mathbf{1314\text{ dimensions}}$
- **Classifier:** `MLPClassifier(hidden_layer_sizes=(256, 128, 64), activation='relu')`

---

### F. Migration Impact (Files to be modified in Step 2)
When this canonical schema is implemented in subsequent steps, the following files will be updated:
1. `scripts/extract_url_features.py`: Implement the canonical 22-feature extractor and regenerate `Dataset/features/url_features.csv`.
2. `scripts/train_baselines.py`: Remove `pd.get_dummies()`; train `url_baseline.pkl` on 22 features.
3. `scripts/train_fusion.py`: Remove `pd.get_dummies()`; train `fusion_model.pkl` on $22 + 12 + 1280 = 1314$ features; store feature names in bundle.
4. `scripts/test_single_url.py`: Align `extract_url_features()` to return the exact 22 canonical features.
5. `scripts/test_fusion.py`: Align `extract_url_features()` to return the exact 22 canonical features without zero-padding.
6. `scripts/zero_day_eval.py`: Align `extract_url_features()` to return the exact 22 canonical features.
7. `scripts/api_server.py`: Align `extract_url_features()` to return the exact 22 canonical features.

---

### G. Validation Plan for Step 2 Implementation
After implementing the canonical schema:
1. **Dimensionality Assertions:**
   - Verify `url_features.csv` has shape `(990, 23)` (22 features + `sample_id`).
   - Verify `scaler.n_features_in_ == 22` in `url_baseline.pkl`.
   - Verify `model.n_features_in_ == 22` in `url_baseline.pkl`.
   - Verify `scalers['url'].n_features_in_ == 22` in `fusion_model.pkl`.
   - Verify `model.n_features_in_ == 1314` in `fusion_model.pkl`.
2. **Deterministic Extraction Test:**
   - Run live extraction and batch extraction on a test URL and assert identical vectors.
3. **Model Evaluation & Metric Update:**
   - Evaluate held-out 80/20 test split and record updated metrics in `baseline_metrics.json` and `fusion_metrics.json`.
4. **Live Inference Validation:**
   - Execute `test_fusion.py`, `zero_day_eval.py`, and `api_server.py` `/analyze` endpoint to verify zero zero-padding and error-free inference.

---

## 5. BROWSER-NATIVE JAVASCRIPT FEATURE EXTRACTION (STEPS 4 & 4.5)

### ✅ STATUS: IMPLEMENTED & 100% PARITY ACROSS VALIDATED TEST CORPUS
*Pure JavaScript implementations of the 22 canonical URL features (`extension/url_features.js`) and 12 canonical HTML features (`extension/html_features.js`) run directly in Chrome Extension MV3 with 100% mathematical and semantic parity across the validated test corpus compared with the Python reference implementation.*

### A. URL Feature Module (`extension/url_features.js`)
- **Dimensions:** Exactly 22 features in canonical index order `0..21`.
- **Environment:** Universal JavaScript (UMD wrapper compatible with Chrome Extension Content Script, Background Service Worker, Node.js, and browser Window).
- **Zero Dependencies:** Pure lexical parsing, regex string evaluation, and Shannon entropy computation. Zero external network requests, zero DNS lookups, zero WHOIS lookups.
- **Key Algorithmic Parity Elements:**
  - Multi-part ccTLD parsing matching Python `tldextract` behavior on public suffix tokens (`co.uk`, `com.au`, `ac.in`, etc.).
  - Query parameter parser matching Python `urllib.parse.parse_qsl` pair token and blank-value discard semantics.
  - Base-2 Shannon entropy calculation matching `scipy`/Python implementation.

### B. HTML Feature Module (`extension/html_features.js`)
- **Dimensions:** Exactly 12 features in canonical index order `0..11`.
- **Environment:** Operates on live DOM (`document` in Content Script) and raw HTML strings via `DOMParser`.
- **Zero Dependencies:** Pure DOM element queries and attribute inspections. Zero sub-resource fetching.
- **Key Algorithmic Parity Elements:**
  - Relative protocol `//` domain resolution matching Python `_get_domain()` fallback.
  - Exact submit button filtering matching BeautifulSoup (`<input type="submit|image">` and `<button type="submit">`).
  - Comment element stripping before script text calculation matching `BeautifulSoup.find_all(text=True)`.

### C. Parity Verification Results (`scratch/test_hardened_parity.py`)
- **URL Parity (Expanded Suite):** 33 diverse edge-case URLs $\times$ 22 features = **726 / 726 matches (0 mismatches, 100% parity across the validated test corpus)**.
- **HTML Parity (Expanded Suite):** 5 diverse DOM/HTML samples $\times$ 12 features = **60 / 60 matches (0 mismatches, 100% parity across the validated test corpus)**.
- **Overall Result:** **100% exact numerical agreement across the validated test corpus**.

