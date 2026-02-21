"""Step 4: Fine-tune bi-encoder for protocol retrieval.

Fine-tunes multilingual-e5-base on (query, protocol_summary) pairs
using MultipleNegativesRankingLoss.

Usage:
    uv run python -m src.training.train_retriever
"""

import json
import random

import numpy as np
import torch
from loguru import logger
from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses,
    evaluation,
)
from torch.utils.data import DataLoader

from src.config import settings, setup_logging


def load_summaries() -> dict[str, str]:
    """Load protocol summaries keyed by protocol_id."""
    summaries = {}
    with open(settings.protocol_summaries_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            summaries[data["protocol_id"]] = data["summary"]
    return summaries


def load_synthetic() -> list[dict]:
    """Load synthetic training data."""
    data = []
    with open(settings.synthetic_training_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def load_test_queries() -> list[dict]:
    """Load test set queries for validation."""
    queries = []
    if settings.test_set_dir.exists():
        for fp in sorted(settings.test_set_dir.glob("*.json")):
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
                queries.append({
                    "query": data["query"],
                    "protocol_id": data["protocol_id"],
                })
    return queries


def build_training_examples(
    synthetic: list[dict], summaries: dict[str, str]
) -> list[InputExample]:
    """Build training pairs: (query, positive_summary)."""
    examples = []
    for item in synthetic:
        pid = item["protocol_id"]
        if pid not in summaries:
            continue
        query = item["query"]
        summary = summaries[pid]
        # e5 models expect "query: " and "passage: " prefixes
        examples.append(
            InputExample(texts=[f"query: {query}", f"passage: {summary}"])
        )
    random.shuffle(examples)
    return examples


def build_evaluator(
    test_queries: list[dict], summaries: dict[str, str]
) -> evaluation.InformationRetrievalEvaluator | None:
    """Build IR evaluator from test queries."""
    if not test_queries:
        return None

    queries_dict = {}
    corpus_dict = {}
    relevant_docs = {}

    # Build corpus from all summaries
    for pid, summary in summaries.items():
        corpus_dict[pid] = f"passage: {summary}"

    # Build queries and relevance
    for i, tq in enumerate(test_queries):
        qid = f"q_{i}"
        queries_dict[qid] = f"query: {tq['query']}"
        relevant_docs[qid] = {tq["protocol_id"]}

    return evaluation.InformationRetrievalEvaluator(
        queries=queries_dict,
        corpus=corpus_dict,
        relevant_docs=relevant_docs,
        name="test-retrieval",
        show_progress_bar=True,
        batch_size=32,
        main_score_function="cosine",
    )


def main():
    setup_logging()
    settings.retriever_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: {}", device)

    # Resume from checkpoint if available, otherwise load base model
    checkpoint = settings.retriever_dir / "config.json"
    if checkpoint.exists():
        logger.info("Resuming from checkpoint: {}", settings.retriever_dir)
        model = SentenceTransformer(str(settings.retriever_dir), device=device)
    else:
        logger.info("Loading base model: {}", settings.retriever_model_name)
        model = SentenceTransformer(settings.retriever_model_name, device=device)
    model.max_seq_length = settings.retriever_max_seq_length

    logger.info("Loading data...")
    summaries = load_summaries()
    logger.info("  Summaries: {}", len(summaries))

    synthetic = load_synthetic()
    logger.info("  Synthetic queries: {}", len(synthetic))

    test_queries = load_test_queries()
    logger.info("  Test queries: {}", len(test_queries))

    # Build training data
    train_examples = build_training_examples(synthetic, summaries)
    logger.info("  Training pairs: {}", len(train_examples))

    if not train_examples:
        logger.error("No training data! Run data prep steps first.")
        return

    train_dataloader = DataLoader(
        train_examples, shuffle=True, batch_size=settings.retriever_batch_size
    )
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # Build evaluator
    evaluator = build_evaluator(test_queries, summaries)

    # Train
    warmup_steps = int(len(train_dataloader) * settings.retriever_epochs * 0.1)
    logger.info("Training for {} epochs, {} steps/epoch", settings.retriever_epochs, len(train_dataloader))
    logger.info("Warmup steps: {}", warmup_steps)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=settings.retriever_epochs,
        evaluator=evaluator,
        evaluation_steps=len(train_dataloader) // 2 if evaluator else 0,
        warmup_steps=warmup_steps,
        output_path=str(settings.retriever_dir),
        optimizer_params={"lr": settings.retriever_lr},
        show_progress_bar=True,
        save_best_model=True if evaluator else False,
    )

    logger.info("Model saved to {}", settings.retriever_dir)

    # Pre-compute protocol embeddings
    logger.info("Pre-computing protocol embeddings...")
    model = SentenceTransformer(str(settings.retriever_dir), device=device)

    protocol_ids = sorted(summaries.keys())
    passages = [f"passage: {summaries[pid]}" for pid in protocol_ids]
    embeddings = model.encode(passages, show_progress_bar=True, batch_size=32)

    np.save(str(settings.protocol_embeddings_path), embeddings)

    # Save protocol ID mapping
    mapping_path = settings.protocol_embeddings_path.parent / "protocol_id_mapping.json"
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(protocol_ids, f)

    logger.info("Embeddings shape: {}", embeddings.shape)
    logger.info("Saved to {}", settings.protocol_embeddings_path)

    # Quick recall@10 evaluation
    if test_queries:
        logger.info("--- Retrieval Recall@10 on test set ---")
        query_texts = [f"query: {tq['query']}" for tq in test_queries]
        query_embs = model.encode(query_texts, show_progress_bar=True, batch_size=32)

        hits = 0
        for i, tq in enumerate(test_queries):
            sims = np.dot(embeddings, query_embs[i])
            top_indices = np.argsort(sims)[-10:][::-1]
            top_pids = [protocol_ids[idx] for idx in top_indices]
            if tq["protocol_id"] in top_pids:
                hits += 1

        recall_at_10 = hits / len(test_queries)
        logger.info("Recall@10: {:.4f} ({}/{})", recall_at_10, hits, len(test_queries))


if __name__ == "__main__":
    main()
