"""LightGBM ranker for ICD code ranking."""

import lightgbm as lgb
import numpy as np

from src.config import RANKER_PATH


class ICDRanker:
    """Ranks ICD code candidates using LightGBM."""

    def __init__(self):
        print("  Loading LightGBM ranker...")
        self.model = lgb.Booster(model_file=str(RANKER_PATH))
        print("  Ranker ready")

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
