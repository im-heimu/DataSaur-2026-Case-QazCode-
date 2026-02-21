"""Local evaluation of the full pipeline (retrieval + ranking).

Runs the inference engine directly (no HTTP) on the test set.

Usage:
    uv run python -m src.training.evaluate_local
"""

import json
import time

from loguru import logger
from tqdm import tqdm

from src.config import settings, setup_logging
from src.inference.engine import DiagnosisEngine


def main():
    setup_logging()
    logger.info("Loading inference engine...")
    engine = DiagnosisEngine()

    # Load test queries
    test_queries = []
    for fp in sorted(settings.test_set_dir.glob("*.json")):
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
            test_queries.append(data)

    logger.info("Test queries: {}", len(test_queries))

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
    logger.info("=== Results ({} test cases) ===", total)
    logger.info("Accuracy@1: {:.4f} ({}/{})", accuracy_hits / total, accuracy_hits, total)
    logger.info("Recall@3:   {:.4f} ({}/{})", recall_hits / total, recall_hits, total)
    logger.info("Avg latency: {:.3f}s", sum(latencies) / len(latencies))
    logger.info("P95 latency: {:.3f}s", sorted(latencies)[int(0.95 * len(latencies))])


if __name__ == "__main__":
    main()
