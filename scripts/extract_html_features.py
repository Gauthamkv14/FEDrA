"""
scripts/extract_html_features.py
================================
Extracts HTML-level features from Dataset/manifest.csv as per Step 5 schema.
Reads the `page.html` file using BeautifulSoup.
"""

import os
import urllib.parse
import pandas as pd
import tldextract
from bs4 import BeautifulSoup
from tqdm import tqdm

# ── Config ──────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST    = os.path.join(BASE_DIR, "Dataset", "manifest.csv")
OUT_DIR     = os.path.join(BASE_DIR, "Dataset", "features")
OUT_CSV     = os.path.join(OUT_DIR, "html_features.csv")

os.makedirs(OUT_DIR, exist_ok=True)

# ── Helpers ─────────────────────────────────────────────────────────────
def get_domain(url: str) -> str:
    """Extract root domain using tldextract."""
    if not url or not isinstance(url, str):
        return ""
    # Add fake scheme if relative to allow parsing
    if not url.startswith("http") and not url.startswith("//"):
        # Not a full absolute URL; likely same-origin or invalid relative path
        return ""
    
    extracted = tldextract.extract(url)
    return f"{extracted.domain}.{extracted.suffix}" if extracted.domain else ""

def extract_html_features(html_path: str, page_domain: str):
    """Parse HTML and extract schema features."""
    if not html_path or not os.path.isfile(html_path):
        # Missing file, return zeros/defaults
        return {
            'num_forms': 0, 'num_inputs': 0, 'num_iframes': 0,
            'num_ext_links': 0, 'num_ext_scripts': 0,
            'has_password_field': 0, 'has_meta_redirect': 0,
            'script_content_ratio': 0.0, 'favicon_mismatch': 0,
            'has_auto_submit': 0, 'input_submit_ratio': 0.0,
            'num_unique_ext_domains': 0
        }
        
    try:
        with open(html_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        soup = BeautifulSoup(content, "html.parser")
    except Exception as e:
        print(f"Error parsing {html_path}: {e}")
        return {
            'num_forms': 0, 'num_inputs': 0, 'num_iframes': 0,
            'num_ext_links': 0, 'num_ext_scripts': 0,
            'has_password_field': 0, 'has_meta_redirect': 0,
            'script_content_ratio': 0.0, 'favicon_mismatch': 0,
            'has_auto_submit': 0, 'input_submit_ratio': 0.0,
            'num_unique_ext_domains': 0
        }

    # 1. Basic counts
    forms = soup.find_all('form')
    inputs = soup.find_all('input')
    iframes = soup.find_all('iframe')
    num_forms = len(forms)
    num_inputs = len(inputs)
    num_iframes = len(iframes)

    # 2. External links & scripts, distinct domains
    external_domains = set()
    num_ext_links = 0
    num_ext_scripts = 0
    
    # 2a. Links
    for a in soup.find_all('a', href=True):
        link_domain = get_domain(a['href'])
        if link_domain and link_domain != page_domain:
            num_ext_links += 1
            external_domains.add(link_domain)
            
    # 2b. Scripts
    for sc in soup.find_all('script', src=True):
        sc_domain = get_domain(sc['src'])
        if sc_domain and sc_domain != page_domain:
            num_ext_scripts += 1
            external_domains.add(sc_domain)
            
    num_unique_ext_domains = len(external_domains)

    # 3. Password fields
    has_password_field = 1 if soup.find('input', type=lambda t: t and t.lower() == 'password') else 0

    # 4. Meta refresh
    meta_tags = soup.find_all('meta')
    has_meta_redirect = 1 if any(
        m.get('http-equiv', '').lower() == 'refresh' for m in meta_tags
    ) else 0

    # 5. Script-to-content ratio
    scripts = soup.find_all('script')
    script_text = "".join([s.get_text() for s in scripts if s.string])
    total_text = soup.get_text()
    
    script_len = len(script_text)
    total_len = len(content)  # using raw file length to prevent div by 0 on empty body
    script_content_ratio = script_len / max(1, total_len)

    # 6. Favicon mismatch
    favicon_mismatch = 0
    favicons = soup.find_all('link', rel=lambda r: r and 'icon' in r.lower())
    for fav in favicons:
        href = fav.get('href', '')
        fav_domain = get_domain(href)
        if fav_domain and fav_domain != page_domain:
            favicon_mismatch = 1
            break

    # 7. Auto-submission forms (heuristics: form with no submit button + JS presence)
    has_auto_submit = 0
    if len(forms) > 0:
        # Check if form tags contain 'submit()' calling script
        if "submit()" in script_text.lower():
            has_auto_submit = 1
            
    # 8. Ratio of input fields to submit buttons
    submits = soup.find_all(['input', 'button'], type=lambda t: t and t.lower() == 'submit')
    num_submits = len(submits)
    # Including image inputs which also act as submits
    num_submits += len(soup.find_all('input', type=lambda t: t and t.lower() == 'image'))
    
    input_submit_ratio = num_inputs / max(1, num_submits) if num_inputs > 0 else 0.0

    return {
        'num_forms': num_forms,
        'num_inputs': num_inputs,
        'num_iframes': num_iframes,
        'num_ext_links': num_ext_links,
        'num_ext_scripts': num_ext_scripts,
        'has_password_field': has_password_field,
        'has_meta_redirect': has_meta_redirect,
        'script_content_ratio': script_content_ratio,
        'favicon_mismatch': favicon_mismatch,
        'has_auto_submit': has_auto_submit,
        'input_submit_ratio': input_submit_ratio,
        'num_unique_ext_domains': num_unique_ext_domains
    }

# ── Main ─────────────────────────────────────────────────────────────────
def run():
    print(f"Reading {MANIFEST}...")
    df = pd.read_csv(MANIFEST)
    
    features_list = []
    
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Extracting HTML features"):
        url = row['url']
        page_domain = get_domain(url if isinstance(url, str) else "")
        html_path = row['html_path'] if pd.notna(row['html_path']) else ""
        
        feat = extract_html_features(html_path, page_domain)
        feat['sample_id'] = row['sample_id']
        features_list.append(feat)
        
    out_df = pd.DataFrame(features_list)
    cols = ['sample_id'] + [c for c in out_df.columns if c != 'sample_id']
    out_df = out_df[cols]
    
    out_df.to_csv(OUT_CSV, index=False)
    print(f"Saved {len(out_df)} records to {OUT_CSV}")

if __name__ == "__main__":
    run()
