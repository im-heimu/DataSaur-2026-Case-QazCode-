"""Semantic retriever: fine-tuned bi-encoder for protocol retrieval."""

import json

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

from src.config import settings


class ProtocolRetriever:
    """Retrieves top-K protocols using fine-tuned semantic search."""

    def __init__(self):
        logger.info("  Loading retriever model...")
        self.model = SentenceTransformer(str(settings.retriever_dir))
        self.model.max_seq_length = 512

        logger.info("  Loading protocol embeddings...")
        self.embeddings = np.load(str(settings.protocol_embeddings_path))

        mapping_path = settings.protocol_embeddings_path.parent / "protocol_id_mapping.json"
        with open(mapping_path, "r", encoding="utf-8") as f:
            self.protocol_ids = json.load(f)

        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings_normalized = self.embeddings / np.maximum(norms, 1e-8)

        logger.info("  Retriever ready: {} protocols (semantic)", len(self.protocol_ids))

    def retrieve(
        self, query: str, top_k: int = settings.top_k_protocols
    ) -> list[tuple[str, float]]:
        """Retrieve top-K protocols using semantic search.

        Returns list of (protocol_id, cosine_similarity) tuples.
        """
        if not query:
            return []

        query_embedding = self.model.encode(
            f"query: {query}", show_progress_bar=False
        )
        query_norm = query_embedding / max(np.linalg.norm(query_embedding), 1e-8)
        scores = np.dot(self.embeddings_normalized, query_norm)

        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [(self.protocol_ids[idx], float(scores[idx])) for idx in top_indices]
