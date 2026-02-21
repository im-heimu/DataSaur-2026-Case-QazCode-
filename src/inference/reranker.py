"""Cross-encoder reranker for protocol retrieval.

Takes top-K candidates from bi-encoder and reranks them using
a cross-encoder model for higher accuracy.
"""

import numpy as np
from loguru import logger
from sentence_transformers import CrossEncoder

from src.config import settings


class ProtocolReranker:
    """Reranks retrieved protocols using a cross-encoder."""

    def __init__(self):
        logger.info("  Loading cross-encoder reranker: {}", settings.reranker_model_name)
        self.model = CrossEncoder(settings.reranker_model_name, max_length=512)
        logger.info("  Reranker ready")

    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, float]],
        passages: dict[str, str],
        top_k: int = settings.reranker_top_k_output,
    ) -> list[tuple[str, float]]:
        """Rerank candidates using cross-encoder.

        Args:
            query: the user query text
            candidates: list of (protocol_id, bi_encoder_score)
            passages: dict mapping protocol_id -> passage text
            top_k: number of results to return

        Returns:
            Reranked list of (protocol_id, cross_encoder_score)
        """
        if not candidates:
            return []

        # Build pairs for cross-encoder
        pairs = []
        valid_candidates = []
        for pid, score in candidates:
            passage = passages.get(pid, "")
            if passage:
                pairs.append((query, passage))
                valid_candidates.append(pid)

        if not pairs:
            return candidates[:top_k]

        # Score all pairs
        scores = self.model.predict(pairs, show_progress_bar=False)

        # Combine and sort
        scored = list(zip(valid_candidates, scores))
        scored.sort(key=lambda x: -x[1])

        return [(pid, float(score)) for pid, score in scored[:top_k]]
