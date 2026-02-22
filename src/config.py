"""Configuration for the medical diagnosis system.

All settings are loaded from environment variables and .env file.
Usage: ``from src.config import settings``
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from loguru import logger

import sys

_PROJECT_ROOT = Path(__file__).parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Paths (not from env, computed) ──────────────────────────
    project_root: Path = _PROJECT_ROOT

    # ── GPT-OSS API (offline data prep only) ────────────────────
    gpt_oss_url: str = "https://api.openai.com/v1"
    gpt_oss_key: str = Field(default="")
    gpt_oss_model: str = "gpt-4.1-mini"
    gpt_oss_concurrency: int = 100

    # ── QazCode LLM API (inference-time reasoning) ────────────
    qazcode_url: str = "https://hub.qazcode.ai"
    qazcode_key: str = Field(default="")
    qazcode_model: str = "oss-120b"
    qazcode_enabled: bool = False  # enable after maximizing non-LLM metrics

    # ── Retriever ───────────────────────────────────────────────
    retriever_model_name: str = "intfloat/multilingual-e5-base"
    retriever_max_seq_length: int = 512
    retriever_epochs: int = 2
    retriever_batch_size: int = 96
    retriever_lr: float = 2e-5
    retriever_max_per_protocol: int = 12
    retriever_hard_negatives_per_positive: int = 2

    # ── Ranker ──────────────────────────────────────────────────
    ranker_learning_rate: float = 0.05
    ranker_num_leaves: int = 31
    ranker_min_child_samples: int = 5
    ranker_n_estimators: int = 500
    ranker_reg_alpha: float = 0.1
    ranker_reg_lambda: float = 0.1

    # ── Cross-encoder reranker ────────────────────────────────────
    reranker_enabled: bool = False  # disabled: generic mmarco hurts medical retrieval
    reranker_model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    reranker_top_k_input: int = 50   # retrieve this many for reranking
    reranker_top_k_output: int = 10  # keep this many after reranking

    # ── Inference ───────────────────────────────────────────────
    top_k_protocols: int = 20
    top_n_diagnoses: int = 3

    # ── Code ranking weights (tiebreaker within protocol) ────
    w_code_embedding: float = 0.1
    w_code_tfidf: float = 0.1

    # ── Text processing ─────────────────────────────────────────
    truncation_markers: list[str] = [
        "ТАКТИКА ЛЕЧЕНИЯ",
        "МЕДИКАМЕНТОЗНОЕ ЛЕЧЕНИЕ",
        "ОРГАНИЗАЦИОННЫЕ АСПЕКТЫ",
        "ПОКАЗАНИЯ ДЛЯ ГОСПИТАЛИЗАЦИИ",
        "Тактика лечения",
        "Медикаментозное лечение",
        "Немедикаментозное лечение",
    ]

    # ── Computed paths ──────────────────────────────────────────
    @property
    def corpus_path(self) -> Path:
        return self.project_root / "corpus" / "protocols_corpus.jsonl"

    @property
    def test_set_dir(self) -> Path:
        return self.project_root / "data" / "test_set"

    @property
    def processed_dir(self) -> Path:
        return self.project_root / "data" / "processed"

    @property
    def models_dir(self) -> Path:
        return self.project_root / "models"

    # Model paths
    @property
    def retriever_dir(self) -> Path:
        return self.models_dir / "retriever"

    @property
    def protocol_embeddings_path(self) -> Path:
        return self.models_dir / "protocol_embeddings.npy"

    @property
    def protocol_data_path(self) -> Path:
        return self.models_dir / "protocol_data.json"

    @property
    def icd_features_path(self) -> Path:
        return self.models_dir / "icd_features.json"

    @property
    def ranker_path(self) -> Path:
        return self.models_dir / "ranker.lgb"

    @property
    def tfidf_path(self) -> Path:
        return self.models_dir / "tfidf_vectorizer.pkl"

    # Processed data paths
    @property
    def protocol_features_path(self) -> Path:
        return self.processed_dir / "protocol_features.jsonl"

    @property
    def synthetic_training_path(self) -> Path:
        return self.processed_dir / "synthetic_training.jsonl"

    @property
    def protocol_summaries_path(self) -> Path:
        return self.processed_dir / "protocol_summaries.jsonl"

    @property
    def training_features_path(self) -> Path:
        return self.processed_dir / "training_features.npz"

    @property
    def training_labels_path(self) -> Path:
        return self.processed_dir / "training_labels.npy"

    @property
    def training_groups_path(self) -> Path:
        return self.processed_dir / "training_groups.npy"

    # Ranker params dict for LightGBM
    @property
    def ranker_params(self) -> dict:
        return {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [1, 3],
            "learning_rate": self.ranker_learning_rate,
            "num_leaves": self.ranker_num_leaves,
            "min_child_samples": self.ranker_min_child_samples,
            "n_estimators": self.ranker_n_estimators,
            "reg_alpha": self.ranker_reg_alpha,
            "reg_lambda": self.ranker_reg_lambda,
            "verbose": -1,
        }


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()


def setup_logging(level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO") -> None:
    logger.remove()

    logger.add(
        sys.stdout,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        level=level,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    logger.add(
        f"logs/{level.lower()}.log",
        level=level,
        rotation="100 MB",
        retention="1 month",
        compression="zip",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
            "{name}:{function}:{line} | {message}"
        ),
    )

    logger.info(f"Logging configured with level: {level}")
