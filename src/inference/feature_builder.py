"""Feature builder for inference: computes features for (query, protocol, icd_code) triples."""

import json
import pickle
import re

import numpy as np
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity

from src.config import settings

import pymorphy3


class FeatureBuilder:
    """Builds feature vectors for LightGBM ranker at inference time."""

    def __init__(self, retriever_model):
        logger.info("  Loading TF-IDF vectorizer...")
        with open(str(settings.tfidf_path), "rb") as f:
            self.tfidf = pickle.load(f)

        logger.info("  Loading ICD features...")
        with open(str(settings.icd_features_path), "r", encoding="utf-8") as f:
            self.icd_descriptions = json.load(f)

        self.retriever_model = retriever_model
        self.morph = pymorphy3.MorphAnalyzer()

        # Pre-compute ICD description embeddings and TF-IDF vectors
        logger.info("  Pre-computing ICD embeddings...")
        self.all_codes = sorted(self.icd_descriptions.keys())
        if self.all_codes:
            code_texts = [
                f"passage: {self.icd_descriptions[c]}" for c in self.all_codes
            ]
            self.code_embeddings = retriever_model.encode(
                code_texts, show_progress_bar=False, batch_size=64
            )
            self.code_tfidf = self.tfidf.transform(
                [self.icd_descriptions.get(c, c) for c in self.all_codes]
            )
        else:
            self.code_embeddings = np.array([])
            self.code_tfidf = None

        self.code_to_idx = {c: i for i, c in enumerate(self.all_codes)}

        # Code frequency (pre-loaded from protocol_data)
        self.code_frequency = {}

        logger.info("  Feature builder ready")

    def set_code_frequency(self, freq: dict[str, int]):
        """Set code corpus frequency from protocol data."""
        self.code_frequency = freq

    def _lemmatize(self, text: str) -> set[str]:
        words = re.findall(r"[а-яёА-ЯЁa-zA-Z]+", text.lower())
        lemmas = set()
        for word in words:
            parsed = self.morph.parse(word)
            if parsed:
                lemmas.add(parsed[0].normal_form)
            else:
                lemmas.add(word)
        return lemmas

    def _symptom_overlap(self, query_lemmas: set[str], symptoms: list[str]) -> float:
        if not symptoms:
            return 0.0
        matches = 0
        for symptom in symptoms:
            symptom_lemmas = self._lemmatize(symptom)
            if not symptom_lemmas:
                continue
            overlap = len(symptom_lemmas & query_lemmas) / len(symptom_lemmas)
            if overlap >= 0.5:
                matches += 1
        return matches / len(symptoms)

    @staticmethod
    def _get_icd_chapter_idx(code: str) -> int:
        chapters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        if code and code[0].isalpha():
            ch = code[0].upper()
            idx = chapters.find(ch)
            return idx if idx >= 0 else 25
        return 25

    def build_features(
        self,
        query: str,
        query_embedding: np.ndarray,
        candidates: list[dict],
    ) -> np.ndarray:
        """Build feature matrix for candidate ICD codes.

        Each candidate is a dict with:
          - code: ICD-10 code
          - protocol_id: which protocol it came from
          - retrieval_score: cosine similarity from retriever
          - protocol_rank: position in retrieval results (0-indexed)
          - n_codes: total codes in the protocol
          - symptoms: list of symptoms from protocol features
          - body_system: body system string
          - icd_code_descriptions: list of code desc dicts from features
        """
        n = len(candidates)
        if n == 0:
            return np.zeros((0, 10), dtype=np.float32)

        # Pre-compute query TF-IDF
        query_tfidf = self.tfidf.transform([query])
        query_lemmas = self._lemmatize(query)

        features = np.zeros((n, 10), dtype=np.float32)

        for i, cand in enumerate(candidates):
            code = cand["code"]

            # 1. Retrieval score
            features[i, 0] = cand["retrieval_score"]

            # 2. TF-IDF similarity (query <-> ICD description)
            if code in self.code_to_idx:
                cidx = self.code_to_idx[code]
                features[i, 1] = float(
                    cosine_similarity(query_tfidf, self.code_tfidf[cidx : cidx + 1])[
                        0, 0
                    ]
                )

            # 3. Symptom overlap
            features[i, 2] = self._symptom_overlap(
                query_lemmas, cand.get("symptoms", [])
            )

            # 4. Query <-> ICD code embedding similarity
            if code in self.code_to_idx:
                cidx = self.code_to_idx[code]
                features[i, 3] = float(
                    np.dot(query_embedding, self.code_embeddings[cidx])
                )

            # 5. Protocol rank (normalized)
            features[i, 4] = 1.0 / (1.0 + cand.get("protocol_rank", 0))

            # 6. Number of codes (normalized)
            features[i, 5] = cand.get("n_codes", 1) / 100.0

            # 7. ICD chapter
            features[i, 6] = self._get_icd_chapter_idx(code)

            # 8. Body system match (simplified heuristic)
            features[i, 7] = 1.0 if cand.get("body_system") else 0.5

            # 9. Code corpus frequency
            features[i, 8] = self.code_frequency.get(code, 0)

            # 10. Distinguishing features similarity
            dist_feat = ""
            for cd in cand.get("icd_code_descriptions", []):
                if cd.get("code") == code:
                    dist_feat = cd.get("distinguishing_features", "")
                    break
            if dist_feat:
                dist_vec = self.tfidf.transform([dist_feat])
                features[i, 9] = float(
                    cosine_similarity(query_tfidf, dist_vec)[0, 0]
                )

        return features
