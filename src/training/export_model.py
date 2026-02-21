"""Step 7: Export and package models for inference.

Collects all trained artifacts into models/ directory.

Usage:
    uv run python -m src.training.export_model
"""

import json
import shutil

from src.config import (
    CORPUS_PATH,
    ICD_FEATURES_PATH,
    MODELS_DIR,
    PROTOCOL_DATA_PATH,
    PROTOCOL_EMBEDDINGS_PATH,
    PROTOCOL_FEATURES_PATH,
    PROTOCOL_SUMMARIES_PATH,
    RANKER_PATH,
    RETRIEVER_DIR,
    TFIDF_PATH,
)


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Check all required files exist
    required = {
        "Retriever model": RETRIEVER_DIR / "config.json",
        "Protocol embeddings": PROTOCOL_EMBEDDINGS_PATH,
        "Protocol ID mapping": PROTOCOL_EMBEDDINGS_PATH.parent / "protocol_id_mapping.json",
        "ICD features": ICD_FEATURES_PATH,
        "Ranker model": RANKER_PATH,
        "TF-IDF vectorizer": TFIDF_PATH,
    }

    all_ok = True
    for name, path in required.items():
        if path.exists():
            print(f"  [OK] {name}: {path}")
        else:
            print(f"  [MISSING] {name}: {path}")
            all_ok = False

    if not all_ok:
        print("\nSome required files are missing! Run training steps first.")
        return

    # Build protocol_data.json: metadata + features for each protocol
    print("\nBuilding protocol_data.json...")
    protocol_data = {}

    # Load corpus for ICD codes
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            protocol_data[p["protocol_id"]] = {
                "protocol_id": p["protocol_id"],
                "source_file": p.get("source_file", ""),
                "title": p.get("title", ""),
                "icd_codes": p.get("icd_codes", []),
            }

    # Add features
    if PROTOCOL_FEATURES_PATH.exists():
        with open(PROTOCOL_FEATURES_PATH, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                pid = data["protocol_id"]
                if pid in protocol_data:
                    protocol_data[pid]["features"] = {
                        "disease_name": data.get("disease_name", ""),
                        "symptoms": data.get("symptoms", []),
                        "diagnostic_criteria": data.get("diagnostic_criteria", ""),
                        "body_system": data.get("body_system", ""),
                        "patient_category": data.get("patient_category", ""),
                        "icd_code_descriptions": data.get("icd_code_descriptions", []),
                    }

    # Add summaries
    if PROTOCOL_SUMMARIES_PATH.exists():
        with open(PROTOCOL_SUMMARIES_PATH, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                pid = data["protocol_id"]
                if pid in protocol_data:
                    protocol_data[pid]["summary"] = data.get("summary", "")

    with open(PROTOCOL_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(protocol_data, f, ensure_ascii=False, indent=None)

    n_with_features = sum(1 for p in protocol_data.values() if "features" in p)
    n_with_summaries = sum(1 for p in protocol_data.values() if "summary" in p)
    print(f"  Total protocols: {len(protocol_data)}")
    print(f"  With features: {n_with_features}")
    print(f"  With summaries: {n_with_summaries}")
    print(f"  Saved to: {PROTOCOL_DATA_PATH}")

    print("\n=== Export complete ===")
    print(f"Models directory: {MODELS_DIR}")

    # List all files in models/
    total_size = 0
    for path in sorted(MODELS_DIR.rglob("*")):
        if path.is_file():
            size_mb = path.stat().st_size / (1024 * 1024)
            total_size += size_mb
            print(f"  {path.relative_to(MODELS_DIR)}: {size_mb:.1f}MB")
    print(f"  Total: {total_size:.1f}MB")


if __name__ == "__main__":
    main()
