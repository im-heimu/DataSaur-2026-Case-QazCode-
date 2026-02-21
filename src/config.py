"""Configuration for the medical diagnosis system."""

import os
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
CORPUS_PATH = PROJECT_ROOT / "corpus" / "protocols_corpus.jsonl"
TEST_SET_DIR = PROJECT_ROOT / "data" / "test_set"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Model paths
MODELS_DIR = PROJECT_ROOT / "models"
RETRIEVER_DIR = MODELS_DIR / "retriever"
PROTOCOL_EMBEDDINGS_PATH = MODELS_DIR / "protocol_embeddings.npy"
PROTOCOL_DATA_PATH = MODELS_DIR / "protocol_data.json"
ICD_FEATURES_PATH = MODELS_DIR / "icd_features.json"
RANKER_PATH = MODELS_DIR / "ranker.lgb"
TFIDF_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"

# Processed data paths
PROTOCOL_FEATURES_PATH = PROCESSED_DIR / "protocol_features.jsonl"
SYNTHETIC_TRAINING_PATH = PROCESSED_DIR / "synthetic_training.jsonl"
PROTOCOL_SUMMARIES_PATH = PROCESSED_DIR / "protocol_summaries.jsonl"
TRAINING_FEATURES_PATH = PROCESSED_DIR / "training_features.npz"
TRAINING_LABELS_PATH = PROCESSED_DIR / "training_labels.npy"
TRAINING_GROUPS_PATH = PROCESSED_DIR / "training_groups.npy"

# GPT-OSS API config (offline data prep only)
GPT_OSS_URL = os.getenv("GPT_OSS_URL", "https://hub.qazcode.ai/v1")
GPT_OSS_KEY = os.environ["GPT_OSS_KEY"]
GPT_OSS_MODEL = os.getenv("GPT_OSS_MODEL", "oss-120b")
GPT_OSS_CONCURRENCY = 100

# Retriever config
RETRIEVER_MODEL_NAME = "intfloat/multilingual-e5-base"
RETRIEVER_MAX_SEQ_LENGTH = 512
RETRIEVER_EPOCHS = 3
RETRIEVER_BATCH_SIZE = 16
RETRIEVER_LR = 2e-5

# Ranker config
RANKER_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [1, 3],
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 5,
    "n_estimators": 500,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "verbose": -1,
}

# Inference config
TOP_K_PROTOCOLS = 10
TOP_N_DIAGNOSES = 3

# Text processing: markers for truncating protocol text
TRUNCATION_MARKERS = [
    "ТАКТИКА ЛЕЧЕНИЯ",
    "МЕДИКАМЕНТОЗНОЕ ЛЕЧЕНИЕ",
    "ОРГАНИЗАЦИОННЫЕ АСПЕКТЫ",
    "ПОКАЗАНИЯ ДЛЯ ГОСПИТАЛИЗАЦИИ",
    "Тактика лечения",
    "Медикаментозное лечение",
    "Немедикаментозное лечение",
]
