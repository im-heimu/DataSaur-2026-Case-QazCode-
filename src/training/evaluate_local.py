"""Local evaluation of the full pipeline (retrieval + ranking).

Runs the inference engine directly (no HTTP) on the test set.

Usage:
    uv run python -m src.training.evaluate_local
"""

import json
import time

from tqdm import tqdm

from src.config import TEST_SET_DIR
from src.inference.engine import DiagnosisEngine


def main():
    print("Loading inference engine...")
    engine = DiagnosisEngine()

    # Load test queries
    test_queries = []
    for fp in sorted(TEST_SET_DIR.glob("*.json")):
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
            test_queries.append(data)

    print(f"Test queries: {len(test_queries)}")

    accuracy_hits = 0
    recall_hits = 0
    latencies = []

    for tq in tqdm(test_queries, desc="Evaluating"):
        query = tq["query"]
        gt = tq["gt"]
        valid_codes = set(tq["icd_codes"])

        start = time.perf_counter()
        results = engine.diagnose(query)
        elapsed = time.perf_counter() - start
        latencies.append(elapsed)

        top_codes = [r["icd10_code"] for r in results[:3]]

        # Accuracy@1
        if top_codes and top_codes[0] == gt:
            accuracy_hits += 1

        # Recall@3
        if any(c in valid_codes for c in top_codes):
            recall_hits += 1

    total = len(test_queries)
    print(f"\n=== Results ({total} test cases) ===")
    print(f"Accuracy@1: {accuracy_hits / total:.4f} ({accuracy_hits}/{total})")
    print(f"Recall@3:   {recall_hits / total:.4f} ({recall_hits}/{total})")
    print(f"Avg latency: {sum(latencies) / len(latencies):.3f}s")
    print(f"P95 latency: {sorted(latencies)[int(0.95 * len(latencies))]:.3f}s")


if __name__ == "__main__":
    main()
