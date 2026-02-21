"""DiagnosisEngine: retrieval + multi-signal code ranking."""

import json
import pickle

import numpy as np
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity

from src.config import settings
from src.inference.retriever import ProtocolRetriever


class DiagnosisEngine:
    """Full inference pipeline: query -> top-N ICD-10 diagnoses.

    Uses bi-encoder + cross-encoder retrieval for protocols, then ranks
    ICD codes using a combination of embedding similarity, TF-IDF
    similarity, and protocol rank weighting.
    """

    def __init__(self):
        logger.info("Initializing DiagnosisEngine...")

        self.retriever = ProtocolRetriever()

        # Load protocol data (features, ICD descriptions)
        logger.info("  Loading protocol data...")
        with open(settings.protocol_data_path, "r", encoding="utf-8") as f:
            self.protocol_data = json.load(f)

        # Load ICD features for code descriptions
        logger.info("  Loading ICD features...")
        with open(settings.icd_features_path, "r", encoding="utf-8") as f:
            self.icd_descriptions = json.load(f)

        # Load TF-IDF vectorizer
        logger.info("  Loading TF-IDF vectorizer...")
        with open(str(settings.tfidf_path), "rb") as f:
            self.tfidf = pickle.load(f)

        # Pre-compute ICD code embeddings
        logger.info("  Pre-computing ICD code embeddings...")
        self.all_codes = sorted(self.icd_descriptions.keys())
        if self.all_codes:
            code_texts = [
                f"passage: {self.icd_descriptions[c]}" for c in self.all_codes
            ]
            self.code_embeddings = self.retriever.model.encode(
                code_texts, show_progress_bar=False, batch_size=64
            )
            # Normalize
            norms = np.linalg.norm(self.code_embeddings, axis=1, keepdims=True)
            self.code_embeddings = self.code_embeddings / np.maximum(norms, 1e-8)
            # Pre-compute TF-IDF vectors for codes
            self.code_tfidf = self.tfidf.transform(
                [self.icd_descriptions.get(c, c) for c in self.all_codes]
            )
        else:
            self.code_embeddings = np.array([])
            self.code_tfidf = None

        self.code_to_idx = {c: i for i, c in enumerate(self.all_codes)}

        # Scoring weights
        self.w_code_embedding = settings.w_code_embedding
        self.w_code_tfidf = settings.w_code_tfidf
        self.w_protocol_rank = settings.w_protocol_rank

        logger.info("DiagnosisEngine ready! (weights: emb={}, tfidf={}, rank={})",
                     self.w_code_embedding, self.w_code_tfidf, self.w_protocol_rank)

    def diagnose(
        self,
        symptoms: str,
        top_n: int = settings.top_n_diagnoses,
        w_code_embedding: float | None = None,
        w_code_tfidf: float | None = None,
        w_protocol_rank: float | None = None,
    ) -> list[dict]:
        """Run full diagnosis pipeline with multi-signal code ranking.

        Scores each (protocol, code) candidate using:
        1. Protocol rank weight (higher for top-ranked protocols)
        2. Embedding similarity (query <-> code description)
        3. TF-IDF similarity (query <-> code description)
        """
        if not symptoms or not symptoms.strip():
            return []

        w_emb = w_code_embedding if w_code_embedding is not None else self.w_code_embedding
        w_tfidf = w_code_tfidf if w_code_tfidf is not None else self.w_code_tfidf
        w_rank = w_protocol_rank if w_protocol_rank is not None else self.w_protocol_rank

        # Step 1: Retrieve top-K protocols (bi-encoder + cross-encoder)
        retrieved = self.retriever.retrieve(symptoms, top_k=settings.top_k_protocols)

        # Step 2: Encode query
        query_embedding = self.retriever.model.encode(
            f"query: {symptoms}", show_progress_bar=False
        )
        query_norm = query_embedding / max(np.linalg.norm(query_embedding), 1e-8)

        # Pre-compute query TF-IDF
        query_tfidf = self.tfidf.transform([symptoms])

        # Step 3: Collect all candidate (protocol, code) pairs and score them
        candidates = []
        seen_codes = set()

        for rank_idx, (pid, retrieval_score) in enumerate(retrieved):
            pd = self.protocol_data.get(pid)
            if not pd:
                continue

            icd_codes = pd.get("icd_codes", [])
            features = pd.get("features", {})
            icd_code_descriptions = features.get("icd_code_descriptions", [])

            # Protocol rank score (decays with rank)
            protocol_rank_score = 1.0 / (1.0 + rank_idx)

            for code in icd_codes:
                if code in seen_codes:
                    continue
                seen_codes.add(code)

                # Embedding similarity
                emb_sim = 0.0
                if code in self.code_to_idx:
                    cidx = self.code_to_idx[code]
                    emb_sim = float(np.dot(query_norm, self.code_embeddings[cidx]))

                # TF-IDF similarity
                tfidf_sim = 0.0
                if code in self.code_to_idx and self.code_tfidf is not None:
                    cidx = self.code_to_idx[code]
                    tfidf_sim = float(
                        cosine_similarity(query_tfidf, self.code_tfidf[cidx:cidx + 1])[0, 0]
                    )

                # Combined score
                combined_score = (
                    w_rank * protocol_rank_score +
                    w_emb * emb_sim +
                    w_tfidf * tfidf_sim
                )

                # Get code metadata
                code_name = code
                dist_features = ""
                for cd in icd_code_descriptions:
                    if cd.get("code") == code:
                        code_name = cd.get("name", code)
                        dist_features = cd.get("distinguishing_features", "")
                        break

                disease = features.get("disease_name", "")
                explanation = disease if disease else f"Код: {code}"
                if dist_features:
                    explanation += f". {dist_features}"

                candidates.append({
                    "code": code,
                    "code_name": code_name,
                    "explanation": explanation,
                    "score": combined_score,
                })

        # Sort by combined score
        candidates.sort(key=lambda x: -x["score"])

        # Build results
        results = []
        for cand in candidates[:top_n]:
            results.append({
                "rank": len(results) + 1,
                "diagnosis": cand["code_name"],
                "icd10_code": cand["code"],
                "explanation": cand["explanation"],
            })

        return results

    def precompute_candidates(self, symptoms: str) -> list[dict]:
        """Retrieve protocols and compute per-code signals without combining.

        Returns list of candidate dicts with raw signal scores for later
        weight optimization (avoids re-running retrieval + cross-encoder).
        """
        if not symptoms or not symptoms.strip():
            return []

        retrieved = self.retriever.retrieve(symptoms, top_k=settings.top_k_protocols)

        query_embedding = self.retriever.model.encode(
            f"query: {symptoms}", show_progress_bar=False
        )
        query_norm = query_embedding / max(np.linalg.norm(query_embedding), 1e-8)
        query_tfidf = self.tfidf.transform([symptoms])

        candidates = []
        seen_codes = set()

        for rank_idx, (pid, retrieval_score) in enumerate(retrieved):
            pd = self.protocol_data.get(pid)
            if not pd:
                continue

            icd_codes = pd.get("icd_codes", [])
            features = pd.get("features", {})
            icd_code_descriptions = features.get("icd_code_descriptions", [])
            protocol_rank_score = 1.0 / (1.0 + rank_idx)

            for code in icd_codes:
                if code in seen_codes:
                    continue
                seen_codes.add(code)

                emb_sim = 0.0
                if code in self.code_to_idx:
                    cidx = self.code_to_idx[code]
                    emb_sim = float(np.dot(query_norm, self.code_embeddings[cidx]))

                tfidf_sim = 0.0
                if code in self.code_to_idx and self.code_tfidf is not None:
                    cidx = self.code_to_idx[code]
                    tfidf_sim = float(
                        cosine_similarity(query_tfidf, self.code_tfidf[cidx:cidx + 1])[0, 0]
                    )

                code_name = code
                dist_features = ""
                for cd in icd_code_descriptions:
                    if cd.get("code") == code:
                        code_name = cd.get("name", code)
                        dist_features = cd.get("distinguishing_features", "")
                        break

                disease = features.get("disease_name", "")
                explanation = disease if disease else f"Код: {code}"
                if dist_features:
                    explanation += f". {dist_features}"

                candidates.append({
                    "code": code,
                    "code_name": code_name,
                    "explanation": explanation,
                    "emb_sim": emb_sim,
                    "tfidf_sim": tfidf_sim,
                    "rank_score": protocol_rank_score,
                })

        return candidates

    @staticmethod
    def score_candidates(
        candidates: list[dict],
        w_emb: float,
        w_tfidf: float,
        w_rank: float,
        top_n: int = 3,
    ) -> list[dict]:
        """Score pre-computed candidates with given weights (no model calls)."""
        for c in candidates:
            c["score"] = w_rank * c["rank_score"] + w_emb * c["emb_sim"] + w_tfidf * c["tfidf_sim"]

        ranked = sorted(candidates, key=lambda x: -x["score"])
        results = []
        for cand in ranked[:top_n]:
            results.append({
                "rank": len(results) + 1,
                "diagnosis": cand["code_name"],
                "icd10_code": cand["code"],
                "explanation": cand["explanation"],
            })
        return results
