"""LightGBM ranker for ICD code ranking."""

import lightgbm as lgb
import numpy as np
from loguru import logger

from src.config import settings


class ICDRanker:
    """Ranks ICD code candidates using LightGBM."""

    def __init__(self):
        logger.info("  Loading LightGBM ranker...")
        self.model = lgb.Booster(model_file=str(settings.ranker_path))
        logger.info("  Ranker ready")

    def rank(self, features: np.ndarray) -> np.ndarray:
        """Predict relevance scores for candidate features.

        Args:
            features: shape (n_candidates, n_features)

        Returns:
            scores: shape (n_candidates,)
        """
        if features.shape[0] == 0:
            return np.array([])
        return self.model.predict(features)
