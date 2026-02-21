"""Build unified training set from synthetic data and test set queries.

Combines synthetic queries with real test queries (for validation split).

Usage:
    uv run python -m src.data_prep.build_training_set
"""

import json

from loguru import logger

from src.config import settings, setup_logging


def main():
    setup_logging()
    settings.processed_dir.mkdir(parents=True, exist_ok=True)

    # Load synthetic data
    synthetic = []
    if settings.synthetic_training_path.exists():
        with open(settings.synthetic_training_path, "r", encoding="utf-8") as f:
            for line in f:
                synthetic.append(json.loads(line))
    logger.info("Synthetic training examples: {}", len(synthetic))

    # Load test set queries (for validation)
    test_queries = []
    if settings.test_set_dir.exists():
        for fp in sorted(settings.test_set_dir.glob("*.json")):
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
                test_queries.append({
                    "query": data["query"],
                    "protocol_id": data["protocol_id"],
                    "target_icd_code": data["gt"],
                    "all_icd_codes": data["icd_codes"],
                    "is_test": True,
                })
    logger.info("Test set queries: {}", len(test_queries))

    # Write combined training set
    train_path = settings.processed_dir / "combined_training.jsonl"
    with open(train_path, "w", encoding="utf-8") as f:
        for item in synthetic:
            item["is_test"] = False
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Write validation set separately
    val_path = settings.processed_dir / "validation_set.jsonl"
    with open(val_path, "w", encoding="utf-8") as f:
        for item in test_queries:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info("Training set: {} ({} examples)", train_path, len(synthetic))
    logger.info("Validation set: {} ({} examples)", val_path, len(test_queries))


if __name__ == "__main__":
    main()
