"""Step 7: Export and package models for inference.

Collects all trained artifacts into models/ directory.

Usage:
    uv run python -m src.training.export_model
"""

import json
import shutil

from loguru import logger

from src.config import settings, setup_logging


def main():
    setup_logging()
    settings.models_dir.mkdir(parents=True, exist_ok=True)

    # Check all required files exist
    required = {
        "Retriever model": settings.retriever_dir / "config.json",
        "Protocol embeddings": settings.protocol_embeddings_path,
        "Protocol ID mapping": settings.protocol_embeddings_path.parent / "protocol_id_mapping.json",
        "ICD features": settings.icd_features_path,
        "Ranker model": settings.ranker_path,
        "TF-IDF vectorizer": settings.tfidf_path,
    }

    all_ok = True
    for name, path in required.items():
        if path.exists():
            logger.info("  [OK] {}: {}", name, path)
        else:
            logger.warning("  [MISSING] {}: {}", name, path)
            all_ok = False

    if not all_ok:
        logger.error("Some required files are missing! Run training steps first.")
        return

    # Build protocol_data.json: metadata + features for each protocol
    logger.info("Building protocol_data.json...")
    protocol_data = {}

    # Load corpus for ICD codes
    with open(settings.corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            protocol_data[p["protocol_id"]] = {
                "protocol_id": p["protocol_id"],
                "source_file": p.get("source_file", ""),
                "title": p.get("title", ""),
                "icd_codes": p.get("icd_codes", []),
            }

    # Add features
    if settings.protocol_features_path.exists():
        with open(settings.protocol_features_path, "r", encoding="utf-8") as f:
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
    if settings.protocol_summaries_path.exists():
        with open(settings.protocol_summaries_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                pid = data["protocol_id"]
                if pid in protocol_data:
                    protocol_data[pid]["summary"] = data.get("summary", "")

    with open(settings.protocol_data_path, "w", encoding="utf-8") as f:
        json.dump(protocol_data, f, ensure_ascii=False, indent=None)

    n_with_features = sum(1 for p in protocol_data.values() if "features" in p)
    n_with_summaries = sum(1 for p in protocol_data.values() if "summary" in p)
    logger.info("  Total protocols: {}", len(protocol_data))
    logger.info("  With features: {}", n_with_features)
    logger.info("  With summaries: {}", n_with_summaries)
    logger.info("  Saved to: {}", settings.protocol_data_path)

    logger.info("=== Export complete ===")
    logger.info("Models directory: {}", settings.models_dir)

    # List all files in models/
    total_size = 0
    for path in sorted(settings.models_dir.rglob("*")):
        if path.is_file():
            size_mb = path.stat().st_size / (1024 * 1024)
            total_size += size_mb
            logger.info("  {}: {:.1f}MB", path.relative_to(settings.models_dir), size_mb)
    logger.info("  Total: {:.1f}MB", total_size)


if __name__ == "__main__":
    main()
