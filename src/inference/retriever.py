"""Semantic retriever: fine-tuned bi-encoder for protocol retrieval."""

import json

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

from src.config import settings
from src.inference.reranker import ProtocolReranker


class ProtocolRetriever:
    """Retrieves top-K protocols using fine-tuned semantic search + cross-encoder reranking."""

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

        # Load passage texts for cross-encoder reranking
        logger.info("  Loading passage texts for reranker...")
        self.passages = self._load_passages()

        # Initialize cross-encoder reranker
        self.reranker = ProtocolReranker()

        logger.info("  Retriever ready: {} protocols (semantic + cross-encoder)", len(self.protocol_ids))

    def _load_passages(self) -> dict[str, str]:
        """Load protocol passage texts for cross-encoder input."""
        summaries = {}
        if settings.protocol_summaries_path.exists():
            with open(settings.protocol_summaries_path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    summaries[data["protocol_id"]] = data["summary"]

        features = {}
        if settings.protocol_features_path.exists():
            with open(settings.protocol_features_path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    features[data["protocol_id"]] = data

        passages = {}
        for pid in self.protocol_ids:
            parts = []
            feat = features.get(pid, {})
            if feat.get("disease_name"):
                parts.append(feat["disease_name"])
            symptoms = feat.get("symptoms", [])
            if symptoms:
                parts.append("Симптомы: " + ", ".join(symptoms))
            if feat.get("diagnostic_criteria"):
                dc = feat["diagnostic_criteria"]
                if isinstance(dc, list):
                    parts.append("Критерии: " + "; ".join(dc))
                elif isinstance(dc, str):
                    parts.append("Критерии: " + dc)
            icd_descs = feat.get("icd_code_descriptions", [])
            if icd_descs:
                dist_parts = []
                for cd in icd_descs:
                    name = cd.get("name", "")
                    dist = cd.get("distinguishing_features", "")
                    if name and dist:
                        dist_parts.append(f"{name}: {dist}")
                    elif name:
                        dist_parts.append(name)
                if dist_parts:
                    parts.append("Коды: " + "; ".join(dist_parts))
            if pid in summaries:
                parts.append(summaries[pid])
            if parts:
                passages[pid] = "\n".join(parts)[:2500]
        return passages

    def retrieve(
        self, query: str, top_k: int = settings.top_k_protocols
    ) -> list[tuple[str, float]]:
        """Retrieve top-K protocols using bi-encoder + cross-encoder reranking.

        Returns list of (protocol_id, score) tuples.
        """
        if not query:
            return []

        query_embedding = self.model.encode(
            f"query: {query}", show_progress_bar=False
        )
        query_norm = query_embedding / max(np.linalg.norm(query_embedding), 1e-8)
        scores = np.dot(self.embeddings_normalized, query_norm)

        # Step 1: Bi-encoder retrieves top-N candidates
        bi_top_k = settings.reranker_top_k_input
        top_indices = np.argsort(scores)[-bi_top_k:][::-1]
        bi_candidates = [(self.protocol_ids[idx], float(scores[idx])) for idx in top_indices]

        # Step 2: Cross-encoder reranks to top-K
        reranked = self.reranker.rerank(
            query, bi_candidates, self.passages, top_k=top_k
        )

        return reranked
