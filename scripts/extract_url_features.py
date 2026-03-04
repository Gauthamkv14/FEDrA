"""
scripts/extract_url_features.py
================================
Extracts URL-level features from Dataset/manifest.csv as per Step 5 schema.
"""

import os
import re
import math
import urllib.parse
import pandas as pd
import tldextract
from tqdm import tqdm

# ── Config ──────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST    = os.path.join(BASE_DIR, "Dataset", "manifest.csv")
OUT_DIR     = os.path.join(BASE_DIR, "Dataset", "features")
OUT_CSV     = os.path.join(OUT_DIR, "url_features.csv")

os.makedirs(OUT_DIR, exist_ok=True)

SUSPICIOUS_WORDS = {'login', 'redirect', 'verify', 'secure'}

# ── Helpers ─────────────────────────────────────────────────────────────
def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    probabilities = [float(s.count(c)) / len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in probabilities)

def extract_features(url: str):
    if not isinstance(url, str) or not url:
        return {}
        
    # Ensure URL has scheme for accurate parsing if missing
    parsed_url = url if "://" in url else "http://" + url
    
    parsed = urllib.parse.urlparse(parsed_url)
    extracted = tldextract.extract(parsed_url)
    
    # 1. Length
    url_length = len(url)
    
    # 2. IP in domain
    domain = extracted.domain
    # Simple regex for IPv4
    is_ip = 1 if re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", domain) else 0
    
    # 3. Subdomains
    subdomain_str = extracted.subdomain
    num_subdomains = len([s for s in subdomain_str.split('.') if s]) if subdomain_str else 0
    
    # 4. HTTPS
    is_https = 1 if url.lower().startswith("https") else 0
    
    # 5. Char counts
    num_at = url.count('@')
    num_dash = url.count('-')
    num_double_slash = url.count('//') - 1 if "://" in url else url.count('//')
    if num_double_slash < 0: num_double_slash = 0 # sanity check
    
    # 6. Domain token entropy
    # Calculate on domain + suffix to represent the primary token
    domain_token = f"{extracted.domain}.{extracted.suffix}"
    domain_entropy = shannon_entropy(domain_token)
    
    # 7. TLD type
    tld_type = extracted.suffix
    
    # 8. Query parameters
    query_params = urllib.parse.parse_qsl(parsed.query)
    num_params = len(query_params)
    
    # 9. Suspicious parameters
    has_susp_params = 0
    query_keys_lower = [k.lower() for k, v in query_params]
    for key in query_keys_lower:
        if any(susp in key for susp in SUSPICIOUS_WORDS):
            has_susp_params = 1
            break
            
    return {
        'url_len': url_length,
        'num_subdomains': num_subdomains,
        'has_ip': is_ip,
        'is_https': is_https,
        'count_at': num_at,
        'count_dash': num_dash,
        'count_double_slash': num_double_slash,
        'domain_entropy': domain_entropy,
        'tld_type': tld_type,
        'num_params': num_params,
        'has_suspicious_params': has_susp_params
    }

# ── Main ─────────────────────────────────────────────────────────────────
def run():
    print(f"Reading {MANIFEST}...")
    df = pd.read_csv(MANIFEST)
    
    features_list = []
    
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Extracting URL features"):
        feat = extract_features(row['url'])
        feat['sample_id'] = row['sample_id']
        features_list.append(feat)
        
    out_df = pd.DataFrame(features_list)
    # Move sample_id to front
    cols = ['sample_id'] + [c for c in out_df.columns if c != 'sample_id']
    out_df = out_df[cols]
    
    out_df.to_csv(OUT_CSV, index=False)
    print(f"Saved {len(out_df)} records to {OUT_CSV}")

if __name__ == "__main__":
    run()
