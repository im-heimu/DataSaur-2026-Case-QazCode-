"""DiagnosisEngine: retrieval + protocol-first code ranking + optional LLM reranking."""

import json
import pickle

import numpy as np
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity

from src.config import settings
from src.inference.retriever import ProtocolRetriever
from src.inference.llm_ranker import llm_rerank


class DiagnosisEngine:
    """Full inference pipeline: query -> top-N ICD-10 diagnoses.

    Strategy: protocol-first ordering (codes from best-ranked protocol
    come first), with embedding + TF-IDF similarity as tiebreaker
    within the same protocol.
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

        # Scoring weights for within-protocol tiebreaking
        self.w_code_embedding = settings.w_code_embedding
        self.w_code_tfidf = settings.w_code_tfidf

        self.llm_enabled = settings.qazcode_enabled
        logger.info("DiagnosisEngine ready! (tiebreak weights: emb={}, tfidf={}, llm={})",
                     self.w_code_embedding, self.w_code_tfidf, self.llm_enabled)

    def _code_score(self, code: str, query_norm: np.ndarray, query_tfidf) -> float:
        """Compute combined code similarity score for within-protocol ranking."""
        emb_sim = 0.0
        tfidf_sim = 0.0

        if code in self.code_to_idx:
            cidx = self.code_to_idx[code]
            emb_sim = float(np.dot(query_norm, self.code_embeddings[cidx]))
            if self.code_tfidf is not None:
                tfidf_sim = float(
                    cosine_similarity(query_tfidf, self.code_tfidf[cidx:cidx + 1])[0, 0]
                )

        return self.w_code_embedding * emb_sim + self.w_code_tfidf * tfidf_sim

    def _collect_all_candidates(
        self,
        symptoms: str,
        retrieved: list[tuple[str, float]],
        query_norm: np.ndarray,
        query_tfidf,
        max_candidates: int = 10,
    ) -> list[dict]:
        """Collect top candidate codes from retrieved protocols (protocol-first)."""
        results = []
        seen_codes = set()

        for pid, retrieval_score in retrieved:
            pd = self.protocol_data.get(pid)
            if not pd:
                continue

            icd_codes = pd.get("icd_codes", [])
            features = pd.get("features", {})
            icd_code_descriptions = features.get("icd_code_descriptions", [])

            code_scores = []
            for code in icd_codes:
                if code in seen_codes:
                    continue
                score = self._code_score(code, query_norm, query_tfidf)
                code_scores.append((code, score))

            code_scores.sort(key=lambda x: -x[1])

            for code, score in code_scores:
                if code in seen_codes:
                    continue
                seen_codes.add(code)

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

                results.append({
                    "rank": len(results) + 1,
                    "diagnosis": code_name,
                    "icd10_code": code,
                    "explanation": explanation,
                })

                if len(results) >= max_candidates:
                    return results

        return results

    def diagnose(
        self,
        symptoms: str,
        top_n: int = settings.top_n_diagnoses,
        w_code_embedding: float | None = None,
        w_code_tfidf: float | None = None,
    ) -> list[dict]:
        """Run full diagnosis pipeline.

        Protocol-first strategy with optional LLM reranking:
        1. Retrieve top-K protocols
        2. Collect top-10 candidate codes (protocol-first + tiebreaker)
        3. If LLM enabled: ask LLM to pick best 3 from 10 candidates
        4. If LLM disabled or fails: return top-3 from step 2
        """
        if not symptoms or not symptoms.strip():
            return []

        # Allow weight override for optimization
        orig_w_emb, orig_w_tfidf = self.w_code_embedding, self.w_code_tfidf
        if w_code_embedding is not None:
            self.w_code_embedding = w_code_embedding
        if w_code_tfidf is not None:
            self.w_code_tfidf = w_code_tfidf

        # Step 1: Retrieve top-K protocols
        retrieved = self.retriever.retrieve(symptoms, top_k=settings.top_k_protocols)

        # Step 2: Encode query for code scoring
        query_embedding = self.retriever.model.encode(
            f"query: {symptoms}", show_progress_bar=False
        )
        query_norm = query_embedding / max(np.linalg.norm(query_embedding), 1e-8)
        query_tfidf = self.tfidf.transform([symptoms])

        # Step 3: Collect wider candidate pool for LLM (or just top_n if no LLM)
        n_candidates = 15 if self.llm_enabled else top_n
        candidates = self._collect_all_candidates(
            symptoms, retrieved, query_norm, query_tfidf, max_candidates=n_candidates
        )

        # Restore weights
        self.w_code_embedding, self.w_code_tfidf = orig_w_emb, orig_w_tfidf

        # Step 4: LLM reranking
        if self.llm_enabled and len(candidates) > top_n:
            llm_codes = llm_rerank(symptoms, candidates)
            if llm_codes:
                code_to_cand = {c["icd10_code"]: c for c in candidates}
                retriever_top = {c["icd10_code"] for c in candidates[:top_n]}

                # Merge strategy: LLM codes first, then fill from retriever order
                # This gives LLM full control over selection from wider pool
                merged = []
                seen = set()
                # LLM-selected codes first
                for code in llm_codes[:top_n]:
                    if code in code_to_cand and code not in seen:
                        seen.add(code)
                        cand = code_to_cand[code]
                        merged.append({
                            "rank": len(merged) + 1,
                            "diagnosis": cand["diagnosis"],
                            "icd10_code": cand["icd10_code"],
                            "explanation": cand["explanation"],
                        })
                # Fill remaining from retriever order
                for cand in candidates:
                    if len(merged) >= top_n:
                        break
                    if cand["icd10_code"] not in seen:
                        seen.add(cand["icd10_code"])
                        merged.append({
                            "rank": len(merged) + 1,
                            "diagnosis": cand["diagnosis"],
                            "icd10_code": cand["icd10_code"],
                            "explanation": cand["explanation"],
                        })
                return merged

        return candidates[:top_n]

    def precompute_candidates(self, symptoms: str) -> list[dict]:
        """Retrieve protocols and compute per-code signals for weight optimization.

        Returns list of candidate dicts with raw signal scores grouped by
        protocol rank (for protocol-first scoring).
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

            for code in icd_codes:
                if code in seen_codes:
                    continue
                seen_codes.add(code)

                emb_sim = 0.0
                tfidf_sim = 0.0
                if code in self.code_to_idx:
                    cidx = self.code_to_idx[code]
                    emb_sim = float(np.dot(query_norm, self.code_embeddings[cidx]))
                    if self.code_tfidf is not None:
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
                    "protocol_rank": rank_idx,
                })

        return candidates

    @staticmethod
    def score_candidates(
        candidates: list[dict],
        w_emb: float,
        w_tfidf: float,
        top_n: int = 3,
    ) -> list[dict]:
        """Score pre-computed candidates with protocol-first strategy.

        Candidates are grouped by protocol_rank. Within each protocol,
        codes are sorted by weighted emb+tfidf score.
        """
        if not candidates:
            return []

        # Group by protocol rank
        from collections import defaultdict
        by_rank = defaultdict(list)
        for c in candidates:
            by_rank[c["protocol_rank"]].append(c)

        results = []
        for rank in sorted(by_rank.keys()):
            group = by_rank[rank]
            # Sort within protocol by weighted score
            for c in group:
                c["_score"] = w_emb * c["emb_sim"] + w_tfidf * c["tfidf_sim"]
            group.sort(key=lambda x: -x["_score"])

            for cand in group:
                results.append({
                    "rank": len(results) + 1,
                    "diagnosis": cand["code_name"],
                    "icd10_code": cand["code"],
                    "explanation": cand["explanation"],
                })
                if len(results) >= top_n:
                    return results

        return results
