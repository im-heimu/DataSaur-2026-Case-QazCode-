"""Build unified training set from synthetic data and test set queries.

Combines synthetic queries with real test queries (for validation split).

Usage:
    uv run python -m src.data_prep.build_training_set
"""

import json
from pathlib import Path

from src.config import (
    PROCESSED_DIR,
    SYNTHETIC_TRAINING_PATH,
    TEST_SET_DIR,
)


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Load synthetic data
    synthetic = []
    if SYNTHETIC_TRAINING_PATH.exists():
        with open(SYNTHETIC_TRAINING_PATH, "r", encoding="utf-8") as f:
            for line in f:
                synthetic.append(json.loads(line))
    print(f"Synthetic training examples: {len(synthetic)}")

    # Load test set queries (for validation)
    test_queries = []
    if TEST_SET_DIR.exists():
        for fp in sorted(TEST_SET_DIR.glob("*.json")):
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
                test_queries.append({
                    "query": data["query"],
                    "protocol_id": data["protocol_id"],
                    "target_icd_code": data["gt"],
                    "all_icd_codes": data["icd_codes"],
                    "is_test": True,
                })
    print(f"Test set queries: {len(test_queries)}")

    # Write combined training set
    train_path = PROCESSED_DIR / "combined_training.jsonl"
    with open(train_path, "w", encoding="utf-8") as f:
        for item in synthetic:
            item["is_test"] = False
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Write validation set separately
    val_path = PROCESSED_DIR / "validation_set.jsonl"
    with open(val_path, "w", encoding="utf-8") as f:
        for item in test_queries:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nTraining set: {train_path} ({len(synthetic)} examples)")
    print(f"Validation set: {val_path} ({len(test_queries)} examples)")


if __name__ == "__main__":
    main()
