"""
scripts/export_onnx.py
======================
FEDrA Step 5: Model Export to ONNX & Numerical Parity Validation.

Exports:
1. URL Baseline: models/onnx/url_baseline.onnx (Pipeline: StandardScaler(22) -> LogisticRegression)
2. HTML Baseline: models/onnx/html_baseline.onnx (Pipeline: StandardScaler(12) -> LogisticRegression)
3. Image Baseline: models/onnx/image_baseline.onnx (Pipeline: StandardScaler(1280) -> LogisticRegression)
4. Visual Feature Extractor: models/onnx/mobilenet_v2_visual.onnx (MobileNetV2 -> GAP -> 1280-dim embedding)
5. Fusion Classifier: models/onnx/fusion_model.onnx (MLPClassifier: 1314 -> 256 -> 128 -> 64 -> 2)
6. Modality Scalers & Weights Metadata: models/onnx/modality_scalers.json
7. Model Contracts & Architecture Metadata: models/onnx/model_contracts.json
8. Comprehensive Parity Validation Fixture: models/onnx/validation_baseline.json

Zero modification to underlying model weights or invariant dimensions:
- URL: 22
- HTML: 12
- Visual: 1280
- Fusion: 1314
"""

import os
import sys
import json
import base64
import joblib
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import onnx
import onnxruntime as ort
from skl2onnx import to_onnx
from sklearn.pipeline import Pipeline

# Ensure scripts directory is on sys.path
sys.path.insert(0, os.path.abspath("scripts"))
from url_features import extract_canonical_url_features_vector, CANONICAL_URL_FEATURE_NAMES, URL_FEATURE_DIM
from api_server import extract_html_features

ONNX_DIR = os.path.join("models", "onnx")
os.makedirs(ONNX_DIR, exist_ok=True)

# ── 1. PyTorch MobileNetV2 GAP Wrapper ─────────────────────────────────────────
class MobileNetV2GAP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1).features
        self.features.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, 224, 224]
        feats = self.features(x)
        # Global Average Pooling across spatial dimensions (H=7, W=7 -> 1, 1)
        return feats.mean([2, 3])


# ── 2. Export Functions ────────────────────────────────────────────────────────
def export_url_model():
    print("[1/5] Exporting URL baseline model...")
    bundle = joblib.load("models/url_baseline.pkl")
    scaler = bundle["scaler"]
    model = bundle["model"]
    pipeline = Pipeline([("scaler", scaler), ("model", model)])

    dummy_input = np.zeros((1, 22), dtype=np.float32)
    onx = to_onnx(
        pipeline,
        dummy_input,
        target_opset=17,
        options={type(model): {"zipmap": False}}
    )
    onnx_path = os.path.join(ONNX_DIR, "url_baseline.onnx")
    with open(onnx_path, "wb") as f:
        f.write(onx.SerializeToString())
    print(f"  [OK] Saved {onnx_path}")
    return onnx_path


def export_html_model():
    print("[2/5] Exporting HTML baseline model...")
    bundle = joblib.load("models/html_baseline.pkl")
    scaler = bundle["scaler"]
    model = bundle["model"]
    pipeline = Pipeline([("scaler", scaler), ("model", model)])

    dummy_input = np.zeros((1, 12), dtype=np.float32)
    onx = to_onnx(
        pipeline,
        dummy_input,
        target_opset=17,
        options={type(model): {"zipmap": False}}
    )
    onnx_path = os.path.join(ONNX_DIR, "html_baseline.onnx")
    with open(onnx_path, "wb") as f:
        f.write(onx.SerializeToString())
    print(f"  [OK] Saved {onnx_path}")
    return onnx_path


def export_image_baseline_model():
    print("[3/5] Exporting Image baseline classifier model...")
    bundle = joblib.load("models/image_baseline.pkl")
    scaler = bundle["scaler"]
    model = bundle["model"]
    pipeline = Pipeline([("scaler", scaler), ("model", model)])

    dummy_input = np.zeros((1, 1280), dtype=np.float32)
    onx = to_onnx(
        pipeline,
        dummy_input,
        target_opset=17,
        options={type(model): {"zipmap": False}}
    )
    onnx_path = os.path.join(ONNX_DIR, "image_baseline.onnx")
    with open(onnx_path, "wb") as f:
        f.write(onx.SerializeToString())
    print(f"  [OK] Saved {onnx_path}")
    return onnx_path


def export_mobilenet_v2():
    print("[4/5] Exporting MobileNetV2 visual feature extractor...")
    model = MobileNetV2GAP()
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)
    onnx_path = os.path.join(ONNX_DIR, "mobilenet_v2_visual.onnx")

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["image_input"],
        output_names=["visual_embedding"],
        dynamic_axes={"image_input": {0: "batch_size"}, "visual_embedding": {0: "batch_size"}},
        opset_version=17,
        dynamo=False
    )
    print(f"  [OK] Saved {onnx_path}")
    return onnx_path


def export_fusion_model():
    print("[5/5] Exporting Fusion MLP classifier...")
    bundle = joblib.load("models/fusion_model.pkl")
    model = bundle["model"]

    dummy_input = np.zeros((1, 1314), dtype=np.float32)
    onx = to_onnx(
        model,
        dummy_input,
        target_opset=17,
        options={type(model): {"zipmap": False}}
    )
    onnx_path = os.path.join(ONNX_DIR, "fusion_model.onnx")
    with open(onnx_path, "wb") as f:
        f.write(onx.SerializeToString())
    print(f"  [OK] Saved {onnx_path}")
    return onnx_path


def export_scalers_and_metadata():
    print("\n[Metadata] Exporting modality scalers and model contracts...")
    fusion_bundle = joblib.load("models/fusion_model.pkl")
    scalers = fusion_bundle["scalers"]
    weights = fusion_bundle["weights"]

    metadata = {
        "schema_version": "v1",
        "dimensions": {
            "url": 22,
            "html": 12,
            "visual": 1280,
            "fused": 1314
        },
        "modality_weights": {
            "url": float(weights["url"]),
            "html": float(weights["html"]),
            "visual": float(weights["visual"])
        },
        "scalers": {
            "url": {
                "mean": scalers["url"].mean_.tolist(),
                "scale": scalers["url"].scale_.tolist()
            },
            "html": {
                "mean": scalers["html"].mean_.tolist(),
                "scale": scalers["html"].scale_.tolist()
            },
            "visual": {
                "mean": scalers["visual"].mean_.tolist(),
                "scale": scalers["visual"].scale_.tolist()
            }
        },
        "url_feature_names": CANONICAL_URL_FEATURE_NAMES,
        "html_feature_names": [
            "num_forms", "num_inputs", "num_iframes", "num_ext_links", "num_ext_scripts",
            "has_password_field", "has_meta_redirect", "script_content_ratio",
            "favicon_mismatch", "has_auto_submit", "input_submit_ratio", "num_unique_ext_domains"
        ],
        "image_preprocessing": {
            "resize_dim": 256,
            "crop_dim": 224,
            "channels": 3,
            "color_mode": "RGB",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "input_layout": "NCHW"
        }
    }

    metadata_path = os.path.join(ONNX_DIR, "modality_scalers.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"  [OK] Saved {metadata_path}")

    contracts = {
        "url_baseline": {
            "file": "url_baseline.onnx",
            "input_name": "X",
            "input_shape": [1, 22],
            "input_dtype": "float32",
            "output_names": ["label", "probabilities"],
            "output_shapes": [[1], [1, 2]],
            "output_dtypes": ["int64", "float32"],
            "preprocessing": "Embedded StandardScaler within ONNX pipeline"
        },
        "html_baseline": {
            "file": "html_baseline.onnx",
            "input_name": "X",
            "input_shape": [1, 12],
            "input_dtype": "float32",
            "output_names": ["label", "probabilities"],
            "output_shapes": [[1], [1, 2]],
            "output_dtypes": ["int64", "float32"],
            "preprocessing": "Embedded StandardScaler within ONNX pipeline"
        },
        "image_baseline": {
            "file": "image_baseline.onnx",
            "input_name": "X",
            "input_shape": [1, 1280],
            "input_dtype": "float32",
            "output_names": ["label", "probabilities"],
            "output_shapes": [[1], [1, 2]],
            "output_dtypes": ["int64", "float32"],
            "preprocessing": "Embedded StandardScaler within ONNX pipeline"
        },
        "mobilenet_v2_visual": {
            "file": "mobilenet_v2_visual.onnx",
            "input_name": "image_input",
            "input_shape": [1, 3, 224, 224],
            "input_dtype": "float32",
            "output_names": ["visual_embedding"],
            "output_shapes": [[1, 1280]],
            "output_dtypes": ["float32"],
            "preprocessing": "Resize(256) -> CenterCrop(224) -> ToTensor() -> Normalize(ImageNet mean, std)"
        },
        "fusion_model": {
            "file": "fusion_model.onnx",
            "input_name": "X",
            "input_shape": [1, 1314],
            "input_dtype": "float32",
            "output_names": ["label", "probabilities"],
            "output_shapes": [[1], [1, 2]],
            "output_dtypes": ["int64", "float32"],
            "preprocessing": "np.hstack([scale(url)*0.4, scale(html)*0.3, scale(visual)*0.3])"
        }
    }

    contracts_path = os.path.join(ONNX_DIR, "model_contracts.json")
    with open(contracts_path, "w", encoding="utf-8") as f:
        json.dump(contracts, f, indent=2)
    print(f"  [OK] Saved {contracts_path}")


# ── 3. Parity Validation Fixture Generation ───────────────────────────────────
def validate_onnx_parity(sample_count: int = 50):
    print(f"\n[Validation] Running end-to-end Python vs ONNX parity on {sample_count} samples...")
    manifest = pd.read_csv("Dataset/manifest.csv")
    
    # Balanced stratified sample
    legit_df = manifest[manifest["label"] == 0].sample(n=sample_count // 2, random_state=42)
    phish_df = manifest[manifest["label"] == 1].sample(n=sample_count // 2, random_state=42)
    sample_df = pd.concat([legit_df, phish_df]).sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Load Python models
    url_bundle = joblib.load("models/url_baseline.pkl")
    html_bundle = joblib.load("models/html_baseline.pkl")
    image_bundle = joblib.load("models/image_baseline.pkl")
    fusion_bundle = joblib.load("models/fusion_model.pkl")

    mv2_net = MobileNetV2GAP()
    mv2_net.eval()
    img_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load ONNX sessions
    sess_url = ort.InferenceSession(os.path.join(ONNX_DIR, "url_baseline.onnx"))
    sess_html = ort.InferenceSession(os.path.join(ONNX_DIR, "html_baseline.onnx"))
    sess_image_cls = ort.InferenceSession(os.path.join(ONNX_DIR, "image_baseline.onnx"))
    sess_mv2 = ort.InferenceSession(os.path.join(ONNX_DIR, "mobilenet_v2_visual.onnx"))
    sess_fusion = ort.InferenceSession(os.path.join(ONNX_DIR, "fusion_model.onnx"))

    url_prob_diffs = []
    html_prob_diffs = []
    image_cls_prob_diffs = []
    visual_emb_diffs = []
    fusion_prob_diffs = []

    url_class_matches = 0
    html_class_matches = 0
    image_cls_class_matches = 0
    fusion_class_matches = 0

    fixtures = []

    for idx, row in sample_df.iterrows():
        sample_id = int(row["sample_id"])
        url = row["url"]
        label = int(row["label"])

        # 1. URL Modality
        u_raw = extract_canonical_url_features_vector(url).astype(np.float32) # (1, 22)
        # Python
        u_s = url_bundle["scaler"].transform(u_raw)
        u_py_pred = int(url_bundle["model"].predict(u_s)[0])
        u_py_prob = float(url_bundle["model"].predict_proba(u_s)[0][1])
        # ONNX
        u_ort_res = sess_url.run(None, {"X": u_raw})
        u_ort_pred = int(u_ort_res[0][0])
        u_ort_prob = float(u_ort_res[1][0][1])

        u_diff = abs(u_py_prob - u_ort_prob)
        url_prob_diffs.append(u_diff)
        if u_py_pred == u_ort_pred:
            url_class_matches += 1

        # 2. HTML Modality
        with open(row["html_path"], "r", encoding="utf-8", errors="ignore") as f:
            html_text = f.read()
        h_feat, _ = extract_html_features(html_text, url)
        h_raw = h_feat.astype(np.float32) # (1, 12)
        # Python
        h_s = html_bundle["scaler"].transform(h_raw)
        h_py_pred = int(html_bundle["model"].predict(h_s)[0])
        h_py_prob = float(html_bundle["model"].predict_proba(h_s)[0][1])
        # ONNX
        h_ort_res = sess_html.run(None, {"X": h_raw})
        h_ort_pred = int(h_ort_res[0][0])
        h_ort_prob = float(h_ort_res[1][0][1])

        h_diff = abs(h_py_prob - h_ort_prob)
        html_prob_diffs.append(h_diff)
        if h_py_pred == h_ort_pred:
            html_class_matches += 1

        # 3. Visual Embedding & Classifier Modality
        with Image.open(row["screenshot_path"]) as img_file:
            img = img_file.convert("RGB")
        img_tensor = img_transform(img).unsqueeze(0) # (1, 3, 224, 224)
        # Python
        with torch.no_grad():
            v_py_emb = mv2_net(img_tensor).numpy() # (1, 1280)
        v_s = image_bundle["scaler"].transform(v_py_emb)
        v_py_pred = int(image_bundle["model"].predict(v_s)[0])
        v_py_prob = float(image_bundle["model"].predict_proba(v_s)[0][1])
        # ONNX
        v_ort_emb = sess_mv2.run(None, {"image_input": img_tensor.numpy()})[0] # (1, 1280)
        v_ort_cls = sess_image_cls.run(None, {"X": v_ort_emb.astype(np.float32)})
        v_ort_pred = int(v_ort_cls[0][0])
        v_ort_prob = float(v_ort_cls[1][0][1])

        v_emb_diff = float(np.abs(v_py_emb - v_ort_emb).max())
        visual_emb_diffs.append(v_emb_diff)
        v_cls_diff = abs(v_py_prob - v_ort_prob)
        image_cls_prob_diffs.append(v_cls_diff)
        if v_py_pred == v_ort_pred:
            image_cls_class_matches += 1

        # 4. Fusion Model
        f_scalers = fusion_bundle["scalers"]
        f_weights = fusion_bundle["weights"]
        f_model = fusion_bundle["model"]

        # Python fused vector
        u_sw = f_scalers["url"].transform(u_raw) * f_weights["url"]
        h_sw = f_scalers["html"].transform(h_raw) * f_weights["html"]
        v_sw = f_scalers["visual"].transform(v_py_emb) * f_weights["visual"]
        f_py_input = np.hstack([u_sw, h_sw, v_sw]).astype(np.float32) # (1, 1314)
        f_py_pred = int(f_model.predict(f_py_input)[0])
        f_py_prob = float(f_model.predict_proba(f_py_input)[0][1])

        # ONNX fused inference
        f_ort_res = sess_fusion.run(None, {"X": f_py_input})
        f_ort_pred = int(f_ort_res[0][0])
        f_ort_prob = float(f_ort_res[1][0][1])

        f_diff = abs(f_py_prob - f_ort_prob)
        fusion_prob_diffs.append(f_diff)
        if f_py_pred == f_ort_pred:
            fusion_class_matches += 1

        fixtures.append({
            "sample_id": sample_id,
            "url": url,
            "true_label": label,
            "raw_url_features": u_raw[0].tolist(),
            "raw_html_features": h_raw[0].tolist(),
            "python_visual_embedding": v_py_emb[0].tolist(),
            "url_model": {
                "py_prob": u_py_prob, "ort_prob": u_ort_prob, "diff": u_diff, "class_match": u_py_pred == u_ort_pred,
                "py_pred": u_py_pred, "ort_pred": u_ort_pred
            },
            "html_model": {
                "py_prob": h_py_prob, "ort_prob": h_ort_prob, "diff": h_diff, "class_match": h_py_pred == h_ort_pred,
                "py_pred": h_py_pred, "ort_pred": h_ort_pred
            },
            "visual_embedding": {
                "max_abs_diff": v_emb_diff, "mean_abs_diff": float(np.abs(v_py_emb - v_ort_emb).mean())
            },
            "image_classifier": {
                "py_prob": v_py_prob, "ort_prob": v_ort_prob, "diff": v_cls_diff, "class_match": v_py_pred == v_ort_pred,
                "py_pred": v_py_pred, "ort_pred": v_ort_pred
            },
            "fusion_model": {
                "py_prob": f_py_prob, "ort_prob": f_ort_prob, "diff": f_diff, "class_match": f_py_pred == f_ort_pred,
                "py_pred": f_py_pred, "ort_pred": f_ort_pred
            }
        })

    summary = {
        "num_validation_samples": sample_count,
        "url_model": {
            "max_prob_diff": float(max(url_prob_diffs)),
            "mean_prob_diff": float(np.mean(url_prob_diffs)),
            "class_agreement_pct": float(url_class_matches / sample_count * 100.0)
        },
        "html_model": {
            "max_prob_diff": float(max(html_prob_diffs)),
            "mean_prob_diff": float(np.mean(html_prob_diffs)),
            "class_agreement_pct": float(html_class_matches / sample_count * 100.0)
        },
        "visual_embedding": {
            "max_elementwise_diff": float(max(visual_emb_diffs)),
            "mean_elementwise_diff": float(np.mean(visual_emb_diffs))
        },
        "image_classifier": {
            "max_prob_diff": float(max(image_cls_prob_diffs)),
            "mean_prob_diff": float(np.mean(image_cls_prob_diffs)),
            "class_agreement_pct": float(image_cls_class_matches / sample_count * 100.0)
        },
        "fusion_model": {
            "max_prob_diff": float(max(fusion_prob_diffs)),
            "mean_prob_diff": float(np.mean(fusion_prob_diffs)),
            "class_agreement_pct": float(fusion_class_matches / sample_count * 100.0)
        }
    }

    validation_artifact = {
        "summary": summary,
        "sample_fixtures": fixtures
    }

    validation_path = os.path.join(ONNX_DIR, "validation_baseline.json")
    with open(validation_path, "w", encoding="utf-8") as f:
        json.dump(validation_artifact, f, indent=2)
    print(f"  [OK] Saved validation baseline fixture: {validation_path}")

    print("\n" + "=" * 70)
    print("PARITY VALIDATION SUMMARY (Python vs ONNX Runtime):")
    print(f"  URL Baseline:      Max Diff: {summary['url_model']['max_prob_diff']:.2e} | Class Agreement: {summary['url_model']['class_agreement_pct']:.1f}%")
    print(f"  HTML Baseline:     Max Diff: {summary['html_model']['max_prob_diff']:.2e} | Class Agreement: {summary['html_model']['class_agreement_pct']:.1f}%")
    print(f"  Visual Embedding:  Max Diff: {summary['visual_embedding']['max_elementwise_diff']:.2e}")
    print(f"  Image Baseline:    Max Diff: {summary['image_classifier']['max_prob_diff']:.2e} | Class Agreement: {summary['image_classifier']['class_agreement_pct']:.1f}%")
    print(f"  Fusion Model:      Max Diff: {summary['fusion_model']['max_prob_diff']:.2e} | Class Agreement: {summary['fusion_model']['class_agreement_pct']:.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    print("=== FEDrA Step 5: Exporting ONNX Artifacts & Validating Parity ===")
    export_url_model()
    export_html_model()
    export_image_baseline_model()
    export_mobilenet_v2()
    export_fusion_model()
    export_scalers_and_metadata()
    validate_onnx_parity(sample_count=50)
