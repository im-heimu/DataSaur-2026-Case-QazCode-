"""DiagnosisEngine: orchestrates retrieval, feature building, and ranking."""

import json

import numpy as np
from loguru import logger

from src.config import settings
from src.inference.feature_builder import FeatureBuilder
from src.inference.ranker import ICDRanker
from src.inference.retriever import ProtocolRetriever


class DiagnosisEngine:
    """Full inference pipeline: query -> top-N ICD-10 diagnoses."""

    def __init__(self):
        logger.info("Initializing DiagnosisEngine...")

        # Load components
        self.retriever = ProtocolRetriever()
        self.ranker = ICDRanker()
        self.feature_builder = FeatureBuilder(self.retriever.model)

        # Load protocol data
        logger.info("  Loading protocol data...")
        with open(settings.protocol_data_path, "r", encoding="utf-8") as f:
            self.protocol_data = json.load(f)

        # Compute code frequency for feature builder
        code_freq = {}
        for pd in self.protocol_data.values():
            for code in pd.get("icd_codes", []):
                code_freq[code] = code_freq.get(code, 0) + 1
        self.feature_builder.set_code_frequency(code_freq)

        logger.info("DiagnosisEngine ready!")

    def diagnose(self, symptoms: str, top_n: int = settings.top_n_diagnoses) -> list[dict]:
        """Run full diagnosis pipeline.

        Args:
            symptoms: free-text symptom description
            top_n: number of top diagnoses to return

        Returns:
            List of dicts with keys: rank, diagnosis, icd10_code, explanation
        """
        if not symptoms or not symptoms.strip():
            return []

        # Step 1: Retrieve top-K protocols
        retrieved = self.retriever.retrieve(symptoms, top_k=settings.top_k_protocols)

        # Step 2: Get query embedding for features
        query_embedding = self.retriever.model.encode(
            f"query: {symptoms}", show_progress_bar=False
        )

        # Step 3: Build candidates from all ICD codes in top-K protocols
        candidates = []
        seen_codes = set()
        for rank_idx, (pid, score) in enumerate(retrieved):
            pd = self.protocol_data.get(pid)
            if not pd:
                continue

            icd_codes = pd.get("icd_codes", [])
            features = pd.get("features", {})
            symptoms_list = features.get("symptoms", [])
            body_system = features.get("body_system", "")
            icd_code_descriptions = features.get("icd_code_descriptions", [])

            for code in icd_codes:
                if code in seen_codes:
                    continue
                seen_codes.add(code)
                candidates.append({
                    "code": code,
                    "protocol_id": pid,
                    "retrieval_score": score,
                    "protocol_rank": rank_idx,
                    "n_codes": len(icd_codes),
                    "symptoms": symptoms_list,
                    "body_system": body_system,
                    "icd_code_descriptions": icd_code_descriptions,
                    "disease_name": features.get("disease_name", ""),
                })

        if not candidates:
            return []

        # Step 4: Build features and rank
        feature_matrix = self.feature_builder.build_features(
            symptoms, query_embedding, candidates
        )
        scores = self.ranker.rank(feature_matrix)

        # Step 5: Sort by score and take top-N
        sorted_indices = np.argsort(scores)[::-1]

        results = []
        for rank, idx in enumerate(sorted_indices[:top_n], start=1):
            cand = candidates[idx]
            code = cand["code"]

            # Find diagnosis name from ICD code descriptions
            diagnosis_name = code
            for cd in cand.get("icd_code_descriptions", []):
                if cd.get("code") == code:
                    diagnosis_name = cd.get("name", code)
                    break

            # Build explanation
            disease = cand.get("disease_name", "")
            explanation = f"Протокол: {disease}" if disease else f"Код: {code}"

            results.append({
                "rank": rank,
                "diagnosis": diagnosis_name,
                "icd10_code": code,
                "explanation": explanation,
            })

        return results
