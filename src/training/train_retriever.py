"""Step 4: Save base bi-encoder and pre-compute protocol embeddings.

Uses multilingual-e5-base without fine-tuning (MNR loss degrades performance
on this task due to semantically overlapping medical protocols).
Retrieval is boosted by hybrid BM25+semantic search in the inference retriever.

Embeddings are computed from truncated protocol texts (diagnostic sections)
rather than short GPT-generated summaries, for better semantic matching.

Usage:
    uv run python -m src.training.train_retriever
"""

import json

import numpy as np
import torch
from loguru import logger
from sentence_transformers import SentenceTransformer

from src.config import settings, setup_logging


def load_protocol_passages() -> dict[str, str]:
    """Build rich passage texts for embedding by combining summaries + features.

    Combines GPT-generated summaries (patient-facing language) with
    extracted symptoms and disease names for best semantic matching
    against patient complaint queries.
    """
    # Load summaries
    summaries = {}
    if settings.protocol_summaries_path.exists():
        with open(settings.protocol_summaries_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                summaries[data["protocol_id"]] = data["summary"]

    # Load features (symptoms, disease name)
    features = {}
    if settings.protocol_features_path.exists():
        with open(settings.protocol_features_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                features[data["protocol_id"]] = data

    # Combine into passages
    passages = {}
    all_pids = set(summaries.keys())

    # Also include protocols without summaries from corpus
    if settings.corpus_path.exists():
        with open(settings.corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                all_pids.add(data["protocol_id"])

    for pid in all_pids:
        parts = []

        # Disease name from features
        feat = features.get(pid, {})
        if feat.get("disease_name"):
            parts.append(feat["disease_name"])

        # Symptoms list
        symptoms = feat.get("symptoms", [])
        if symptoms:
            parts.append("Симптомы: " + ", ".join(symptoms[:15]))

        # Summary (main content)
        if pid in summaries:
            parts.append(summaries[pid])

        if parts:
            # Cap at ~1500 chars to fit in 512 tokens
            passage = "\n".join(parts)[:1500]
            passages[pid] = passage

    return passages


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


def main():
    setup_logging()
    settings.retriever_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: {}", device)

    # Use base model without fine-tuning
    logger.info("Loading base model: {}", settings.retriever_model_name)
    model = SentenceTransformer(settings.retriever_model_name, device=device)
    model.max_seq_length = settings.retriever_max_seq_length

    logger.info("Loading data...")
    protocol_passages = load_protocol_passages()
    logger.info("  Protocol passages: {}", len(protocol_passages))

    test_queries = load_test_queries()
    logger.info("  Test queries: {}", len(test_queries))

    # Save base model as retriever
    model.save(str(settings.retriever_dir))
    logger.info("Base model saved to {}", settings.retriever_dir)

    # Pre-compute protocol embeddings
    logger.info("Pre-computing protocol embeddings...")
    protocol_ids = sorted(protocol_passages.keys())
    passages = [f"passage: {protocol_passages[pid]}" for pid in protocol_ids]
    embeddings = model.encode(passages, show_progress_bar=True, batch_size=32)

    np.save(str(settings.protocol_embeddings_path), embeddings)

    # Save protocol ID mapping
    mapping_path = settings.protocol_embeddings_path.parent / "protocol_id_mapping.json"
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(protocol_ids, f)

    logger.info("Embeddings shape: {}", embeddings.shape)
    logger.info("Saved to {}", settings.protocol_embeddings_path)

    # Quick recall@10 evaluation (semantic only, without BM25 hybrid)
    if test_queries:
        logger.info("--- Semantic-only Recall@10 on test set ---")
        query_texts = [f"query: {tq['query']}" for tq in test_queries]
        query_embs = model.encode(query_texts, show_progress_bar=True, batch_size=32)

        # Normalize
        emb_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        emb_normalized = embeddings / np.maximum(emb_norms, 1e-8)

        hits = 0
        for i, tq in enumerate(test_queries):
            q_norm = query_embs[i] / max(np.linalg.norm(query_embs[i]), 1e-8)
            sims = np.dot(emb_normalized, q_norm)
            top_indices = np.argsort(sims)[-10:][::-1]
            top_pids = [protocol_ids[idx] for idx in top_indices]
            if tq["protocol_id"] in top_pids:
                hits += 1

        recall_at_10 = hits / len(test_queries)
        logger.info("Semantic Recall@10: {:.4f} ({}/{})", recall_at_10, hits, len(test_queries))
        logger.info("(Hybrid BM25+semantic will be higher at inference time)")


if __name__ == "__main__":
    main()
