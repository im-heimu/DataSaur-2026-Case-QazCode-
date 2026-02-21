"""Local evaluation of the full pipeline (retrieval + ranking).

Runs the inference engine directly (no HTTP) on the test set.
Supports weight optimization mode to find best scoring weights.

Usage:
    uv run python -m src.training.evaluate_local
    uv run python -m src.training.evaluate_local --optimize
"""

import argparse
import json
import time

from loguru import logger
from tqdm import tqdm

from src.config import settings, setup_logging
from src.inference.engine import DiagnosisEngine


def evaluate(
    engine: DiagnosisEngine,
    test_queries: list[dict],
    w_code_embedding: float | None = None,
    w_code_tfidf: float | None = None,
    w_protocol_rank: float | None = None,
    show_progress: bool = True,
) -> dict:
    """Evaluate engine on test queries with optional custom weights.

    Returns dict with accuracy_at_1, recall_at_3, avg_latency, p95_latency.
    """
    accuracy_hits = 0
    recall_hits = 0
    latencies = []

    iterator = tqdm(test_queries, desc="Evaluating") if show_progress else test_queries

    for tq in iterator:
        query = tq["query"]
        gt = tq["gt"]
        valid_codes = set(tq["icd_codes"])

        start = time.perf_counter()
        results = engine.diagnose(
            query,
            w_code_embedding=w_code_embedding,
            w_code_tfidf=w_code_tfidf,
            w_protocol_rank=w_protocol_rank,
        )
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
    return {
        "accuracy_at_1": accuracy_hits / total if total else 0,
        "recall_at_3": recall_hits / total if total else 0,
        "accuracy_hits": accuracy_hits,
        "recall_hits": recall_hits,
        "total": total,
        "avg_latency": sum(latencies) / len(latencies) if latencies else 0,
        "p95_latency": sorted(latencies)[int(0.95 * len(latencies))] if latencies else 0,
    }


def optimize_weights(engine: DiagnosisEngine, test_queries: list[dict]):
    """Grid search over scoring weights to find best combination."""
    logger.info("=== Weight Optimization ===")

    best_accuracy = {"score": 0, "weights": {}}
    best_recall = {"score": 0, "weights": {}}
    best_combined = {"score": 0, "weights": {}}

    weight_grid = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    total_combos = len(weight_grid) ** 3
    combo_i = 0

    for w_rank in weight_grid:
        for w_emb in weight_grid:
            for w_tfidf in weight_grid:
                combo_i += 1
                metrics = evaluate(
                    engine, test_queries,
                    w_code_embedding=w_emb,
                    w_code_tfidf=w_tfidf,
                    w_protocol_rank=w_rank,
                    show_progress=False,
                )

                acc = metrics["accuracy_at_1"]
                rec = metrics["recall_at_3"]
                combined = acc + rec  # optimize both

                if acc > best_accuracy["score"]:
                    best_accuracy = {"score": acc, "weights": {"emb": w_emb, "tfidf": w_tfidf, "rank": w_rank}}
                if rec > best_recall["score"]:
                    best_recall = {"score": rec, "weights": {"emb": w_emb, "tfidf": w_tfidf, "rank": w_rank}}
                if combined > best_combined["score"]:
                    best_combined = {"score": combined, "weights": {"emb": w_emb, "tfidf": w_tfidf, "rank": w_rank}}

                if combo_i % 20 == 0:
                    logger.info(
                        "  [{}/{}] rank={:.1f} emb={:.1f} tfidf={:.1f} => acc={:.4f} rec={:.4f}",
                        combo_i, total_combos, w_rank, w_emb, w_tfidf, acc, rec
                    )

    logger.info("=== Optimization Results ===")
    logger.info("Best Accuracy@1: {:.4f} weights={}", best_accuracy["score"], best_accuracy["weights"])
    logger.info("Best Recall@3:   {:.4f} weights={}", best_recall["score"], best_recall["weights"])
    logger.info("Best Combined:   {:.4f} weights={}", best_combined["score"], best_combined["weights"])

    return best_combined


def main():
    setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("--optimize", action="store_true", help="Run weight optimization")
    args = parser.parse_args()

    logger.info("Loading inference engine...")
    engine = DiagnosisEngine()

    # Load test queries
    test_queries = []
    for fp in sorted(settings.test_set_dir.glob("*.json")):
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
            test_queries.append(data)

    logger.info("Test queries: {}", len(test_queries))

    if args.optimize:
        optimize_weights(engine, test_queries)
    else:
        metrics = evaluate(engine, test_queries)
        logger.info("=== Results ({} test cases) ===", metrics["total"])
        logger.info("Accuracy@1: {:.4f} ({}/{})", metrics["accuracy_at_1"], metrics["accuracy_hits"], metrics["total"])
        logger.info("Recall@3:   {:.4f} ({}/{})", metrics["recall_at_3"], metrics["recall_hits"], metrics["total"])
        logger.info("Avg latency: {:.3f}s", metrics["avg_latency"])
        logger.info("P95 latency: {:.3f}s", metrics["p95_latency"])


if __name__ == "__main__":
    main()
