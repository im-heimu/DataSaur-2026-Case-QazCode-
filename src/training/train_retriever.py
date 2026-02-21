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


def truncate_protocol_text(text: str, max_chars: int = 1500) -> str:
    """Truncate protocol text to diagnostic sections only.

    Keeps ~1500 chars (fits in 512 tokens for multilingual-e5-base).
    """
    earliest_pos = len(text)
    for marker in settings.truncation_markers:
        pos = text.find(marker)
        if pos != -1 and pos < earliest_pos:
            earliest_pos = pos

    truncated = text[:earliest_pos].strip()
    if len(truncated) > max_chars:
        truncated = truncated[:max_chars]
    return truncated


def load_protocol_texts() -> dict[str, str]:
    """Load truncated protocol texts from corpus for embedding."""
    texts = {}
    with open(settings.corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            pid = data["protocol_id"]
            text = truncate_protocol_text(data.get("text", ""))
            if text:
                texts[pid] = text
    return texts


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
    protocol_texts = load_protocol_texts()
    logger.info("  Protocol texts: {}", len(protocol_texts))

    test_queries = load_test_queries()
    logger.info("  Test queries: {}", len(test_queries))

    # Save base model as retriever
    model.save(str(settings.retriever_dir))
    logger.info("Base model saved to {}", settings.retriever_dir)

    # Pre-compute protocol embeddings
    logger.info("Pre-computing protocol embeddings...")
    protocol_ids = sorted(protocol_texts.keys())
    passages = [f"passage: {protocol_texts[pid]}" for pid in protocol_ids]
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
