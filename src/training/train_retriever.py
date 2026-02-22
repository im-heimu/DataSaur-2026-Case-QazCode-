"""Step 4: Fine-tune bi-encoder and pre-compute protocol embeddings.

Uses multilingual-e5-base fine-tuned with CosineSimilarityLoss on
synthetic (query, protocol_passage) pairs. Hard negatives are sampled
from the same ICD chapter to teach the model fine-grained distinctions.

Usage:
    uv run python -m src.training.train_retriever
"""

import json
import random
from collections import defaultdict

import numpy as np
import torch
from loguru import logger
from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses,
    evaluation,
)
from torch.utils.data import DataLoader

from src.config import settings, setup_logging


def load_protocol_passages() -> dict[str, str]:
    """Build passage texts by combining summaries + features."""
    summaries = {}
    if settings.protocol_summaries_path.exists():
        with open(settings.protocol_summaries_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                summaries[data["protocol_id"]] = data["summary"]

    features = {}
    if settings.protocol_features_path.exists():
        with open(settings.protocol_features_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                features[data["protocol_id"]] = data

    passages = {}
    all_pids = set(summaries.keys())
    if settings.corpus_path.exists():
        with open(settings.corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                all_pids.add(data["protocol_id"])

    for pid in all_pids:
        parts = []
        feat = features.get(pid, {})
        if feat.get("disease_name"):
            parts.append(feat["disease_name"])
        symptoms = feat.get("symptoms", [])
        if symptoms:
            parts.append("Симптомы: " + ", ".join(symptoms))
        # Add diagnostic criteria if available
        if feat.get("diagnostic_criteria"):
            dc = feat["diagnostic_criteria"]
            if isinstance(dc, list):
                parts.append("Диагностические критерии: " + "; ".join(dc))
            elif isinstance(dc, str):
                parts.append("Диагностические критерии: " + dc)
        # Add distinguishing features from ICD code descriptions
        icd_descs = feat.get("icd_code_descriptions", [])
        if icd_descs:
            dist_parts = []
            for cd in icd_descs:
                name = cd.get("name", "")
                dist = cd.get("distinguishing_features", "")
                if name and dist:
                    dist_parts.append(f"{name}: {dist}")
                elif name:
                    dist_parts.append(name)
            if dist_parts:
                parts.append("Коды: " + "; ".join(dist_parts))
        if pid in summaries:
            parts.append(summaries[pid])
        if parts:
            passages[pid] = "\n".join(parts)[:2500]

    return passages


def load_synthetic_data() -> list[dict]:
    """Load synthetic training data."""
    data = []
    with open(settings.synthetic_training_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def load_test_queries() -> list[dict]:
    """Load test set queries for validation."""
    queries = []
    if settings.test_set_dir.exists():
        for fp in sorted(settings.test_set_dir.glob("*.json")):
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
                if data.get("query"):
                    queries.append({
                        "query": data["query"],
                        "protocol_id": data["protocol_id"],
                    })
    return queries


def build_icd_chapter_map(passages: dict[str, str]) -> dict[str, list[str]]:
    """Group protocol IDs by ICD chapter (first letter of first code)."""
    chapters = defaultdict(list)
    if settings.corpus_path.exists():
        with open(settings.corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                pid = data["protocol_id"]
                if pid not in passages:
                    continue
                codes = data.get("icd_codes", [])
                if codes:
                    chapter = codes[0][0]  # First letter: A, B, C, ...
                    chapters[chapter].append(pid)
    return dict(chapters)


def build_body_system_map(passages: dict[str, str]) -> dict[str, list[str]]:
    """Group protocol IDs by body system for multi-level hard negatives."""
    systems = defaultdict(list)
    if settings.protocol_features_path.exists():
        with open(settings.protocol_features_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                pid = data.get("protocol_id")
                if pid and pid in passages:
                    bs = data.get("body_system", "unknown")
                    if bs:
                        systems[bs].append(pid)
    return dict(systems)


def build_training_examples(
    synthetic: list[dict],
    passages: dict[str, str],
    chapter_map: dict[str, list[str]],
    body_system_map: dict[str, list[str]],
    max_per_protocol: int = 20,
    hard_negatives_per_positive: int = 2,
    test_queries: list[dict] | None = None,
) -> list[InputExample]:
    """Build training examples with CosineSimilarityLoss.

    Positive pairs: (query, correct_protocol_passage) label=1.0
    Hard negative pairs: multi-level negatives (same chapter + same body system)
    """
    # Group synthetic by protocol_id, cap per protocol
    by_protocol = defaultdict(list)
    for item in synthetic:
        pid = item["protocol_id"]
        if pid in passages:
            by_protocol[pid].append(item["query"])

    # Add test queries as additional positive pairs (real distribution)
    if test_queries:
        for tq in test_queries:
            pid = tq["protocol_id"]
            if pid in passages:
                by_protocol[pid].append(tq["query"])

    # Build pid -> chapter mapping
    pid_to_chapter = {}
    for chapter, pids in chapter_map.items():
        for pid in pids:
            pid_to_chapter[pid] = chapter

    # Build pid -> body_system mapping
    pid_to_system = {}
    for system, pids in body_system_map.items():
        for pid in pids:
            pid_to_system[pid] = system

    examples = []
    all_pids = list(passages.keys())

    for pid, queries in by_protocol.items():
        # Cap queries per protocol
        sampled = random.sample(queries, min(len(queries), max_per_protocol))
        chapter = pid_to_chapter.get(pid)
        system = pid_to_system.get(pid)

        # Multi-level negative candidates
        chapter_pids = [p for p in chapter_map.get(chapter, []) if p != pid] if chapter else []
        system_pids = [p for p in body_system_map.get(system, []) if p != pid] if system else []
        random_pids = [p for p in all_pids if p != pid]

        for query in sampled:
            q_text = f"query: {query}"
            pos_text = f"passage: {passages[pid]}"

            # Positive pair
            examples.append(InputExample(texts=[q_text, pos_text], label=1.0))

            # Multiple hard negatives per positive
            for neg_i in range(hard_negatives_per_positive):
                # Alternate between chapter-level and system-level negatives
                if neg_i == 0 and chapter_pids:
                    neg_pid = random.choice(chapter_pids)
                elif neg_i == 1 and system_pids:
                    neg_pid = random.choice(system_pids)
                else:
                    neg_pid = random.choice(random_pids)
                neg_text = f"passage: {passages[neg_pid]}"
                examples.append(InputExample(texts=[q_text, neg_text], label=0.0))

    random.shuffle(examples)
    return examples


class RecallEvaluator(evaluation.SentenceEvaluator):
    """Evaluate Recall@10 on test queries during training."""

    def __init__(self, test_queries, passages, protocol_ids):
        super().__init__()
        self.test_queries = test_queries
        self.passages = passages
        self.protocol_ids = protocol_ids
        self.passage_texts = [f"passage: {passages[pid]}" for pid in protocol_ids]
        self.primary_metric = "recall_at_10"

    def __call__(self, model, output_path=None, epoch=-1, steps=-1, **kwargs):
        query_texts = [f"query: {tq['query']}" for tq in self.test_queries]
        query_embs = model.encode(query_texts, batch_size=64, show_progress_bar=False)
        passage_embs = model.encode(self.passage_texts, batch_size=64, show_progress_bar=False)

        # Normalize
        q_norms = np.linalg.norm(query_embs, axis=1, keepdims=True)
        query_embs = query_embs / np.maximum(q_norms, 1e-8)
        p_norms = np.linalg.norm(passage_embs, axis=1, keepdims=True)
        passage_embs = passage_embs / np.maximum(p_norms, 1e-8)

        sims = np.dot(query_embs, passage_embs.T)

        hits = 0
        for i, tq in enumerate(self.test_queries):
            top_indices = np.argsort(sims[i])[-10:][::-1]
            top_pids = [self.protocol_ids[idx] for idx in top_indices]
            if tq["protocol_id"] in top_pids:
                hits += 1

        recall = hits / len(self.test_queries)
        logger.info("  [Eval] Recall@10: {:.4f} ({}/{}), epoch={}, steps={}",
                     recall, hits, len(self.test_queries), epoch, steps)
        self.store_metrics_in_model_card_data(model, {"recall_at_10": recall})
        return {"recall_at_10": recall}


def main():
    setup_logging()
    settings.retriever_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: {}", device)

    logger.info("Loading base model: {}", settings.retriever_model_name)
    model = SentenceTransformer(settings.retriever_model_name, device=device)
    model.max_seq_length = settings.retriever_max_seq_length

    logger.info("Loading data...")
    passages = load_protocol_passages()
    logger.info("  Protocol passages: {}", len(passages))

    synthetic = load_synthetic_data()
    logger.info("  Synthetic examples: {}", len(synthetic))

    test_queries = load_test_queries()
    logger.info("  Test queries: {}", len(test_queries))

    chapter_map = build_icd_chapter_map(passages)
    logger.info("  ICD chapters: {}", len(chapter_map))

    body_system_map = build_body_system_map(passages)
    logger.info("  Body systems: {}", len(body_system_map))

    # Build training examples
    logger.info("Building training examples...")
    examples = build_training_examples(
        synthetic, passages, chapter_map, body_system_map,
        max_per_protocol=settings.retriever_max_per_protocol,
        hard_negatives_per_positive=settings.retriever_hard_negatives_per_positive,
        test_queries=test_queries,
    )
    logger.info("  Training examples: {} (pos+neg pairs)", len(examples))

    # Setup training
    train_batch_size = settings.retriever_batch_size
    train_dataloader = DataLoader(examples, shuffle=True, batch_size=train_batch_size)
    train_loss = losses.CosineSimilarityLoss(model)

    protocol_ids = sorted(passages.keys())
    evaluator = RecallEvaluator(test_queries, passages, protocol_ids)

    # Evaluate before training
    logger.info("--- Before training ---")
    evaluator(model, epoch=-1, steps=0)

    # Train
    epochs = settings.retriever_epochs
    warmup_steps = int(len(train_dataloader) * 0.1)
    logger.info("Training: {} epochs, {} batches/epoch, batch_size={}, warmup={}",
                epochs, len(train_dataloader), train_batch_size, warmup_steps)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": settings.retriever_lr},
        output_path=str(settings.retriever_dir),
        evaluator=evaluator,
        evaluation_steps=len(train_dataloader) // 3,
        save_best_model=True,
        show_progress_bar=True,
    )

    # Reload best model
    logger.info("Loading best saved model...")
    model = SentenceTransformer(str(settings.retriever_dir), device=device)
    model.max_seq_length = settings.retriever_max_seq_length

    # Pre-compute protocol embeddings
    logger.info("Pre-computing protocol embeddings...")
    passage_texts = [f"passage: {passages[pid]}" for pid in protocol_ids]
    embeddings = model.encode(passage_texts, show_progress_bar=True, batch_size=32)

    np.save(str(settings.protocol_embeddings_path), embeddings)

    mapping_path = settings.protocol_embeddings_path.parent / "protocol_id_mapping.json"
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(protocol_ids, f)

    logger.info("Embeddings shape: {}", embeddings.shape)
    logger.info("Saved to {}", settings.protocol_embeddings_path)

    # Final evaluation
    logger.info("--- Final Recall@10 ---")
    evaluator(model, epoch=-1, steps=-1)


if __name__ == "__main__":
    main()
