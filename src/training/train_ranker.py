"""Step 6: Train LightGBM ranker for ICD code ranking.

Uses LambdaRank objective to rank ICD codes within a protocol.

Usage:
    uv run python -m src.training.train_ranker
"""

import json

import lightgbm as lgb
import numpy as np

from src.config import (
    TRAINING_FEATURES_PATH,
    TRAINING_LABELS_PATH,
    TRAINING_GROUPS_PATH,
    RANKER_PATH,
    RANKER_PARAMS,
    PROCESSED_DIR,
)


def main():
    RANKER_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Loading training data...")
    features = np.load(str(TRAINING_FEATURES_PATH))["features"]
    labels = np.load(str(TRAINING_LABELS_PATH))
    groups = np.load(str(TRAINING_GROUPS_PATH))

    print(f"Features: {features.shape}")
    print(f"Labels: {labels.shape} ({labels.sum():.0f} positives)")
    print(f"Groups: {groups.shape} ({groups.sum()} total rows)")

    # Split train/val based on is_test flag
    split_path = PROCESSED_DIR / "query_is_test.npy"
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

    print(f"\nTrain: {X_train.shape[0]} rows, {len(train_groups)} groups")
    print(f"Val:   {X_val.shape[0]} rows, {len(val_groups)} groups")

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

    print("\nTraining LightGBM ranker...")
    callbacks = [lgb.log_evaluation(50)]

    booster = lgb.train(
        RANKER_PARAMS,
        train_data,
        valid_sets=[val_data],
        valid_names=["val"],
        num_boost_round=RANKER_PARAMS.get("n_estimators", 500),
        callbacks=callbacks,
    )

    # Save model
    booster.save_model(str(RANKER_PATH))
    print(f"\nModel saved to {RANKER_PATH}")

    # Feature importance
    importance = booster.feature_importance(importance_type="gain")
    print("\nFeature importance (gain):")
    for name, imp in sorted(zip(feature_names, importance), key=lambda x: -x[1]):
        print(f"  {name}: {imp:.1f}")

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
        print(f"\nValidation Accuracy@1: {correct / total:.4f} ({correct}/{total})")


if __name__ == "__main__":
    main()
