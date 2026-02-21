"""Step 6: Train LightGBM ranker for ICD code ranking.

Uses LambdaRank objective to rank ICD codes within a protocol.

Usage:
    uv run python -m src.training.train_ranker
"""

import json

import lightgbm as lgb
import numpy as np
from loguru import logger

from src.config import settings, setup_logging


def main():
    setup_logging()
    settings.ranker_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading training data...")
    features = np.load(str(settings.training_features_path))["features"]
    labels = np.load(str(settings.training_labels_path))
    groups = np.load(str(settings.training_groups_path))

    logger.info("Features: {}", features.shape)
    logger.info("Labels: {} ({:.0f} positives)", labels.shape, labels.sum())
    logger.info("Groups: {} ({} total rows)", groups.shape, groups.sum())

    # Split train/val based on is_test flag
    split_path = settings.processed_dir / "query_is_test.npy"
    is_test = np.load(str(split_path))

    # Build group-level mask: a group is test if any of its rows are test
    group_offsets = np.cumsum(groups)
    group_starts = np.concatenate([[0], group_offsets[:-1]])

    train_mask = np.zeros(len(labels), dtype=bool)
    val_mask = np.zeros(len(labels), dtype=bool)
    train_groups = []
    val_groups = []

    for i, (start, size) in enumerate(zip(group_starts, groups)):
        end = start + size
        if is_test[start]:  # Test group
            val_mask[start:end] = True
            val_groups.append(size)
        else:
            train_mask[start:end] = True
            train_groups.append(size)

    X_train = features[train_mask]
    y_train = labels[train_mask]
    X_val = features[val_mask]
    y_val = labels[val_mask]

    logger.info("Train: {} rows, {} groups", X_train.shape[0], len(train_groups))
    logger.info("Val:   {} rows, {} groups", X_val.shape[0], len(val_groups))

    feature_names = [
        "retrieval_score",
        "tfidf_similarity",
        "symptom_overlap",
        "query_code_embedding_sim",
        "protocol_rank",
        "n_codes_normalized",
        "icd_chapter",
        "body_system_match",
        "code_corpus_frequency",
        "distinguishing_features_sim",
    ]

    train_data = lgb.Dataset(
        X_train, label=y_train, group=train_groups, feature_name=feature_names
    )
    val_data = lgb.Dataset(
        X_val, label=y_val, group=val_groups, feature_name=feature_names,
        reference=train_data,
    )

    logger.info("Training LightGBM ranker...")
    callbacks = [lgb.log_evaluation(50)]

    booster = lgb.train(
        settings.ranker_params,
        train_data,
        valid_sets=[val_data],
        valid_names=["val"],
        num_boost_round=settings.ranker_params.get("n_estimators", 500),
        callbacks=callbacks,
    )

    # Save model
    booster.save_model(str(settings.ranker_path))
    logger.info("Model saved to {}", settings.ranker_path)

    # Feature importance
    importance = booster.feature_importance(importance_type="gain")
    logger.info("Feature importance (gain):")
    for name, imp in sorted(zip(feature_names, importance), key=lambda x: -x[1]):
        logger.info("  {}: {:.1f}", name, imp)

    # Quick validation accuracy
    if X_val.shape[0] > 0:
        val_preds = booster.predict(X_val)
        # Compute accuracy@1 per group
        offset = 0
        correct = 0
        total = 0
        for gsize in val_groups:
            group_preds = val_preds[offset : offset + gsize]
            group_labels = y_val[offset : offset + gsize]
            top_idx = np.argmax(group_preds)
            if group_labels[top_idx] == 1:
                correct += 1
            total += 1
            offset += gsize
        logger.info("Validation Accuracy@1: {:.4f} ({}/{})", correct / total, correct, total)


if __name__ == "__main__":
    main()
