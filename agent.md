# FEDrA — Agent Guidelines & Operating Rules

This file establishes persistent instructions for AI agents and contributors working in this repository. All future development, refactoring, analysis, and maintenance must strictly adhere to these rules.

---

## 1. Ground Rules & Source of Truth

1. **The codebase and artifacts are the sole source of truth.**
   - Do NOT assume a feature exists because it is mentioned in documentation, specifications, or previous conversation logs.
   - Always verify existence, connections, and functionality directly in source code, configuration files, and model artifacts before asserting status.

2. **Strict Status Categorization:**
   Whenever assessing or reporting any component, explicitly classify it under one of these precise categories:
   - `IMPLEMENTED` — Code exists, is functional, and is verified connected.
   - `PARTIAL` — Meaningful implementation exists, but essential pieces are missing or incomplete.
   - `IMPLEMENTED-BUT-NOT-INTEGRATED` — Code is written and working standalone, but disconnected from the end-to-end runtime pipeline.
   - `MISSING` / `NOT IMPLEMENTED` — Described in documentation/specifications but completely absent in code.
   - `UNKNOWN` — Cannot be verified from repository evidence without additional execution or user input.

3. **Never Silently Resolve Contradictions:**
   - If documentation conflicts with implementation (e.g., claimed client-side inference vs. actual Flask server; claimed MLP baselines vs. actual Logistic Regression; claimed DNS-free operation vs. active `socket.getaddrinfo()` calls), document the contradiction explicitly in `architecture.md` and `memory.md`.
   - Do not adjust documentation to hide flaws, and do not make unapproved breaking assumptions.

4. **Task Execution Workflow (One Logical Task at a Time):**
   Follow this disciplined pattern for every task:
   1. Read the relevant sections of `agent.md`, `memory.md`, `architecture.md`, `progress.md`, and `feature_schema.md`.
   2. Verify current git branch is `development`.
   3. Inspect existing code before writing or editing.
   4. Make targeted, minimal changes. Do not perform broad refactors when fixing a specific issue.
   5. Test and validate the output against regression criteria.
   6. Record changes and validation metrics in `progress.md` and `memory.md`.
   7. Report what was done, verified results, and next steps clearly.

5. **Python Environment & Tooling:**
   - Always execute Python scripts within the active `fedra` conda environment (`conda activate fedra` or derive Python from the active environment path). Do not hardcode machine-specific interpreter paths.
   - Adhere to the `managing-python-dependencies` rules: never run global `pip install` or override the project's established dependency manager.

6. **Model Serialization & Architecture Invariants:**
   - All machine learning models, scalers, and pipelines must be saved using `joblib.dump()` and loaded using `joblib.load()`. Never use standard `pickle` directly.
   - Do not silently change agreed model dimensions or modality weights without explicit forensic justification and retraining.
   - Current canonical dimensions: URL = 22 dims, HTML = 12 dims, Visual = 1280 dims, Fused = 1314 dims.

7. **Dataset Integrity Rules:**
   - Treat `Dataset/` as read-only raw data. Never delete, rename, or overwrite raw dataset folders.
   - **Never use folder names as model features.** Legit folders contain domain names while phishing folders contain hashes; using folder names causes severe label leakage.
   - `sample_id` in manifests and feature CSVs must be a neutral integer index.

8. **Git Workflow & Branching:**
   - Active working branch: `development`. Never commit or push directly to `main`.
   - Merging to `main` is reserved for stable milestones via PR.
   - Commit message prefixes:
     - `feat:` new feature or pipeline capability
     - `fix:` bug fix or schema correction
     - `docs:` documentation updates
     - `data:` dataset or manifest changes
     - `model:` training, evaluation, or artifact updates

9. **Prohibited Unverified Claims:**
   Never state or imply that the following features are working until they are fully implemented, integrated, and verified in code:
   - Client-side in-browser inference (`onnxruntime-web` / TensorFlow.js)
   - SHAP explainability attributions
   - Grad-CAM visual heatmaps
   - Complete elimination of Flask backend
