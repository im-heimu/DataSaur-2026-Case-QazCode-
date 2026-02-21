"""DiagnosisEngine: retrieval + embedding-based code ranking."""

import json

import numpy as np
from loguru import logger

from src.config import settings
from src.inference.retriever import ProtocolRetriever


class DiagnosisEngine:
    """Full inference pipeline: query -> top-N ICD-10 diagnoses.

    Uses bi-encoder retrieval for protocols, then ranks ICD codes by
    cosine similarity between query embedding and code description embedding.
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
        else:
            self.code_embeddings = np.array([])

        self.code_to_idx = {c: i for i, c in enumerate(self.all_codes)}

        logger.info("DiagnosisEngine ready!")

    def diagnose(self, symptoms: str, top_n: int = settings.top_n_diagnoses) -> list[dict]:
        """Run full diagnosis pipeline.

        Strategy: retrieve top-K protocols, then for each protocol rank its
        codes by embedding similarity. Return codes from the best protocols
        first, using code similarity only as tiebreaker within same protocol.
        """
        if not symptoms or not symptoms.strip():
            return []

        # Step 1: Retrieve top-K protocols
        retrieved = self.retriever.retrieve(symptoms, top_k=settings.top_k_protocols)

        # Step 2: Encode query
        query_embedding = self.retriever.model.encode(
            f"query: {symptoms}", show_progress_bar=False
        )
        query_norm = query_embedding / max(np.linalg.norm(query_embedding), 1e-8)

        # Step 3: For each protocol (in retrieval order), rank its codes
        # by embedding similarity and emit them. This ensures codes from
        # the top-1 protocol come first.
        results = []
        seen_codes = set()

        for pid, retrieval_score in retrieved:
            pd = self.protocol_data.get(pid)
            if not pd:
                continue

            icd_codes = pd.get("icd_codes", [])
            features = pd.get("features", {})
            icd_code_descriptions = features.get("icd_code_descriptions", [])

            # Score codes within this protocol
            code_scores = []
            for code in icd_codes:
                if code in seen_codes:
                    continue
                code_sim = 0.0
                if code in self.code_to_idx:
                    cidx = self.code_to_idx[code]
                    code_sim = float(np.dot(query_norm, self.code_embeddings[cidx]))
                code_scores.append((code, code_sim))

            # Sort by similarity within protocol
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

                if len(results) >= top_n:
                    return results

        return results
