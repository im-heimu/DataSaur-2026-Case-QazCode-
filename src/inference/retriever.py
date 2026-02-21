"""Bi-encoder retrieval: find top-K protocols for a query."""

import json

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import (
    PROTOCOL_EMBEDDINGS_PATH,
    RETRIEVER_DIR,
    TOP_K_PROTOCOLS,
)


class ProtocolRetriever:
    """Retrieves top-K protocols using bi-encoder similarity."""

    def __init__(self):
        print("  Loading retriever model...")
        self.model = SentenceTransformer(str(RETRIEVER_DIR))
        self.model.max_seq_length = 512

        print("  Loading protocol embeddings...")
        self.embeddings = np.load(str(PROTOCOL_EMBEDDINGS_PATH))

        mapping_path = PROTOCOL_EMBEDDINGS_PATH.parent / "protocol_id_mapping.json"
        with open(mapping_path, "r", encoding="utf-8") as f:
            self.protocol_ids = json.load(f)

        # Normalize embeddings for faster cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings_normalized = self.embeddings / np.maximum(norms, 1e-8)

        print(f"  Retriever ready: {len(self.protocol_ids)} protocols")

    def retrieve(
        self, query: str, top_k: int = TOP_K_PROTOCOLS
    ) -> list[tuple[str, float]]:
        """Retrieve top-K protocols for a query.

        Returns list of (protocol_id, similarity_score) tuples.
        """
        query_embedding = self.model.encode(
            f"query: {query}", show_progress_bar=False
        )

        # Normalize query embedding
        query_norm = query_embedding / max(np.linalg.norm(query_embedding), 1e-8)

        # Cosine similarity
        similarities = np.dot(self.embeddings_normalized, query_norm)

        # Top-K
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            results.append((self.protocol_ids[idx], float(similarities[idx])))

        return results
