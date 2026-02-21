"""Hybrid retriever: BM25 + bi-encoder with Reciprocal Rank Fusion."""

import json
import re

import numpy as np
import pymorphy3
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

from src.config import settings


class ProtocolRetriever:
    """Retrieves top-K protocols using hybrid BM25 + semantic search."""

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

        # Load protocol texts for BM25 (full corpus texts = more keywords)
        logger.info("  Building BM25 index...")
        self.morph = pymorphy3.MorphAnalyzer()
        protocol_texts = self._load_protocol_texts()
        summaries = self._load_summaries()
        self.pid_to_idx = {pid: i for i, pid in enumerate(self.protocol_ids)}

        # Build lemmatized corpus for TF-IDF (BM25 approximation)
        # Combine full text + summary for maximum keyword coverage
        corpus = []
        for pid in self.protocol_ids:
            text = protocol_texts.get(pid, "")
            summary = summaries.get(pid, "")
            combined = f"{summary}\n{text}" if summary else text
            corpus.append(self._lemmatize(combined))

        self.tfidf = TfidfVectorizer(
            max_features=20000, ngram_range=(1, 2), sublinear_tf=True
        )
        self.tfidf_matrix = self.tfidf.fit_transform(corpus)

        logger.info("  Retriever ready: {} protocols (hybrid BM25+semantic)", len(self.protocol_ids))

    def _load_protocol_texts(self) -> dict[str, str]:
        """Load protocol texts from corpus for BM25 indexing."""
        texts = {}
        if settings.corpus_path.exists():
            with open(settings.corpus_path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    texts[data["protocol_id"]] = data.get("text", "")
        return texts

    def _load_summaries(self) -> dict[str, str]:
        """Load protocol summaries."""
        summaries = {}
        if settings.protocol_summaries_path.exists():
            with open(settings.protocol_summaries_path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    summaries[data["protocol_id"]] = data["summary"]
        return summaries

    def _lemmatize(self, text: str) -> str:
        """Lemmatize Russian text for BM25."""
        if not text:
            return ""
        words = re.findall(r"[а-яёА-ЯЁa-zA-Z0-9]+", text.lower())
        lemmas = []
        for word in words:
            parsed = self.morph.parse(word)
            if parsed:
                lemmas.append(parsed[0].normal_form)
            else:
                lemmas.append(word)
        return " ".join(lemmas)

    def retrieve(
        self, query: str, top_k: int = settings.top_k_protocols
    ) -> list[tuple[str, float]]:
        """Retrieve top-K protocols using hybrid BM25 + semantic search.

        Uses Reciprocal Rank Fusion (RRF) to combine rankings.
        Returns list of (protocol_id, similarity_score) tuples.
        """
        k_rrf = 60  # RRF constant

        # --- Semantic search ---
        query_embedding = self.model.encode(
            f"query: {query}", show_progress_bar=False
        )
        query_norm = query_embedding / max(np.linalg.norm(query_embedding), 1e-8)
        semantic_scores = np.dot(self.embeddings_normalized, query_norm)
        semantic_ranking = np.argsort(semantic_scores)[::-1]

        # --- BM25 (TF-IDF) search ---
        query_lemmatized = self._lemmatize(query)
        query_vec = self.tfidf.transform([query_lemmatized])
        bm25_scores = (self.tfidf_matrix @ query_vec.T).toarray().flatten()
        bm25_ranking = np.argsort(bm25_scores)[::-1]

        # --- Reciprocal Rank Fusion ---
        rrf_scores = {}
        n_consider = min(100, len(self.protocol_ids))

        for rank, idx in enumerate(semantic_ranking[:n_consider]):
            pid = self.protocol_ids[idx]
            rrf_scores[pid] = rrf_scores.get(pid, 0) + 1.0 / (k_rrf + rank + 1)

        for rank, idx in enumerate(bm25_ranking[:n_consider]):
            pid = self.protocol_ids[idx]
            rrf_scores[pid] = rrf_scores.get(pid, 0) + 1.0 / (k_rrf + rank + 1)

        # Sort by RRF score
        sorted_pids = sorted(rrf_scores.items(), key=lambda x: -x[1])

        results = []
        for pid, score in sorted_pids[:top_k]:
            results.append((pid, score))

        return results
