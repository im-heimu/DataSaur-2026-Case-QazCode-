"""Step 5: Feature engineering for the LightGBM ranker.

For each (query, protocol, icd_code) triple, computes features for ranking.

Usage:
    uv run python -m src.training.build_features
"""

import json
import pickle

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from src.config import settings, setup_logging
from src.training.text_utils import lemmatize_text, compute_symptom_overlap


def load_all_data():
    """Load all required data files."""
    # Protocol features
    pf_map = {}
    if settings.protocol_features_path.exists():
        with open(settings.protocol_features_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                pf_map[data["protocol_id"]] = data

    # Summaries
    summaries = {}
    with open(settings.protocol_summaries_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            summaries[data["protocol_id"]] = data["summary"]

    # Synthetic training data
    synthetic = []
    with open(settings.synthetic_training_path, "r", encoding="utf-8") as f:
        for line in f:
            synthetic.append(json.loads(line))

    # Test queries
    test_queries = []
    if settings.test_set_dir.exists():
        for fp in sorted(settings.test_set_dir.glob("*.json")):
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
                test_queries.append(data)

    return pf_map, summaries, synthetic, test_queries


def get_icd_chapter(code: str) -> str:
    """Extract ICD chapter letter from code."""
    if code and code[0].isalpha():
        return code[0].upper()
    return "X"


def compute_code_frequency(pf_map: dict) -> dict[str, int]:
    """Compute how many protocols each code appears in."""
    freq = {}
    for pf in pf_map.values():
        for code in pf.get("icd_codes", []):
            freq[code] = freq.get(code, 0) + 1
    return freq


def build_icd_descriptions(pf_map: dict) -> dict[str, str]:
    """Build text descriptions for each ICD code across all protocols."""
    desc_map = {}
    for pf in pf_map.values():
        code_descs = pf.get("icd_code_descriptions", [])
        for cd in code_descs:
            code = cd.get("code", "")
            if not code:
                continue
            name = cd.get("name", "")
            features = cd.get("distinguishing_features", "")
            text = f"{name}. {features}"
            if code not in desc_map or len(text) > len(desc_map[code]):
                desc_map[code] = text
    return desc_map


def main():
    setup_logging()
    settings.processed_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading data...")
    pf_map, summaries, synthetic, test_queries = load_all_data()
    logger.info("  Protocol features: {}", len(pf_map))
    logger.info("  Summaries: {}", len(summaries))
    logger.info("  Synthetic queries: {}", len(synthetic))
    logger.info("  Test queries: {}", len(test_queries))

    # Load retriever model and embeddings
    logger.info("Loading retriever model...")
    model = SentenceTransformer(str(settings.retriever_dir))
    protocol_embeddings = np.load(str(settings.protocol_embeddings_path))
    mapping_path = settings.protocol_embeddings_path.parent / "protocol_id_mapping.json"
    with open(mapping_path, "r", encoding="utf-8") as f:
        protocol_ids = json.load(f)
    pid_to_idx = {pid: i for i, pid in enumerate(protocol_ids)}

    # Build ICD descriptions and TF-IDF
    logger.info("Building TF-IDF vectorizer...")
    icd_descriptions = build_icd_descriptions(pf_map)
    code_frequency = compute_code_frequency(pf_map)

    # Fit TF-IDF on all descriptions + summaries
    all_texts = list(icd_descriptions.values()) + list(summaries.values())
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    tfidf.fit(all_texts)

    # Save artifacts
    with open(str(settings.tfidf_path), "wb") as f:
        pickle.dump(tfidf, f)

    with open(str(settings.icd_features_path), "w", encoding="utf-8") as f:
        json.dump(icd_descriptions, f, ensure_ascii=False)

    # Pre-compute ICD description embeddings
    logger.info("Computing ICD code embeddings...")
    all_codes = sorted(icd_descriptions.keys())
    code_texts = [f"passage: {icd_descriptions[c]}" for c in all_codes]
    code_embeddings = model.encode(code_texts, show_progress_bar=True, batch_size=64)
    code_to_idx = {c: i for i, c in enumerate(all_codes)}

    # Pre-compute ICD TF-IDF vectors
    code_tfidf = tfidf.transform([icd_descriptions.get(c, c) for c in all_codes])

    # Build training groups: each group is (query, protocol) with all codes as candidates
    logger.info("Building feature matrix...")

    # Combine synthetic + test queries (filter out None queries)
    all_queries = []
    for item in synthetic:
        if not item.get("query"):
            continue
        all_queries.append({
            "query": item["query"],
            "protocol_id": item["protocol_id"],
            "target_icd_code": item["target_icd_code"],
            "all_icd_codes": item["all_icd_codes"],
            "is_test": False,
        })
    for tq in test_queries:
        if not tq.get("query"):
            continue
        all_queries.append({
            "query": tq["query"],
            "protocol_id": tq["protocol_id"],
            "target_icd_code": tq["gt"],
            "all_icd_codes": tq["icd_codes"],
            "is_test": True,
        })

    # Encode all queries
    logger.info("Encoding {} queries...", len(all_queries))
    query_texts = [f"query: {q['query']}" for q in all_queries]

    # Batch encode
    batch_size = 256
    query_embeddings = model.encode(query_texts, show_progress_bar=True, batch_size=batch_size)

    # TF-IDF for queries
    query_tfidf = tfidf.transform([q["query"] for q in all_queries])

    # Build features
    features_list = []
    labels_list = []
    groups_list = []

    icd_chapters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    chapter_to_idx = {c: i for i, c in enumerate(icd_chapters)}

    # Pre-compute symptom overlaps per (query_text, protocol_id) to avoid
    # repeated pymorphy3 lemmatization — the main bottleneck.
    logger.info("Pre-computing symptom overlaps...")
    symptom_overlap_cache = {}
    unique_pairs = {}
    for i, q in enumerate(all_queries):
        pid = q["protocol_id"]
        if pid not in pid_to_idx:
            continue
        key = (i, pid)
        if key not in unique_pairs:
            unique_pairs[key] = (q["query"], pf_map.get(pid, {}).get("symptoms", []))

    for (i, pid), (query_text, symptoms) in tqdm(unique_pairs.items(), desc="Symptom overlaps"):
        symptom_overlap_cache[(i, pid)] = compute_symptom_overlap(query_text, symptoms)

    for i, q in enumerate(tqdm(all_queries, desc="Building features")):
        pid = q["protocol_id"]
        icd_codes = q["all_icd_codes"]
        target = q["target_icd_code"]

        if pid not in pid_to_idx:
            continue

        # Protocol retrieval score
        proto_idx = pid_to_idx[pid]
        retrieval_score = float(np.dot(protocol_embeddings[proto_idx], query_embeddings[i]))

        pf = pf_map.get(pid, {})
        symptoms = pf.get("symptoms", [])
        body_system = pf.get("body_system", "")
        n_codes = len(icd_codes)

        group_size = 0
        for code in icd_codes:
            # Label: 1 if target, 0 otherwise
            label = 1 if code == target else 0

            # Feature vector
            feat = []

            # 1. Retrieval score (query <-> protocol)
            feat.append(retrieval_score)

            # 2. TF-IDF similarity (query <-> ICD description)
            if code in code_to_idx:
                cidx = code_to_idx[code]
                tfidf_sim = float(cosine_similarity(query_tfidf[i:i+1], code_tfidf[cidx:cidx+1])[0, 0])
            else:
                tfidf_sim = 0.0
            feat.append(tfidf_sim)

            # 3. Symptom overlap (pre-computed)
            overlap = symptom_overlap_cache.get((i, pid), 0.0)
            feat.append(overlap)

            # 4. Query <-> ICD code embedding similarity
            if code in code_to_idx:
                cidx = code_to_idx[code]
                emb_sim = float(np.dot(query_embeddings[i], code_embeddings[cidx]))
            else:
                emb_sim = 0.0
            feat.append(emb_sim)

            # 5. Protocol rank (1.0 for exact match, normalized)
            feat.append(1.0)  # During training, protocol is always the correct one

            # 6. Number of codes in protocol (normalized)
            feat.append(n_codes / 100.0)

            # 7. ICD chapter (numeric)
            chapter = get_icd_chapter(code)
            feat.append(chapter_to_idx.get(chapter, 25))

            # 8. Body system match (simplified: always 1 for same protocol)
            feat.append(1.0)

            # 9. Code corpus frequency
            feat.append(code_frequency.get(code, 0))

            # 10. Distinguishing features similarity
            code_descs = pf.get("icd_code_descriptions", [])
            dist_feat = ""
            for cd in code_descs:
                if cd.get("code") == code:
                    dist_feat = cd.get("distinguishing_features", "")
                    break
            if dist_feat:
                dist_vec = tfidf.transform([dist_feat])
                dist_sim = float(cosine_similarity(query_tfidf[i:i+1], dist_vec)[0, 0])
            else:
                dist_sim = 0.0
            feat.append(dist_sim)

            features_list.append(feat)
            labels_list.append(label)
            group_size += 1

        if group_size > 0:
            groups_list.append(group_size)

    features = np.array(features_list, dtype=np.float32)
    labels = np.array(labels_list, dtype=np.float32)
    groups = np.array(groups_list, dtype=np.int32)

    logger.info("Features shape: {}", features.shape)
    logger.info("Labels: {:.0f} positives / {} total", labels.sum(), len(labels))
    logger.info("Groups: {}", len(groups))

    np.savez(str(settings.training_features_path), features=features)
    np.save(str(settings.training_labels_path), labels)
    np.save(str(settings.training_groups_path), groups)

    # Also save train/val split info
    split_path = settings.processed_dir / "query_is_test.npy"
    is_test_flags = []
    for q in all_queries:
        if q["protocol_id"] not in pid_to_idx:
            continue
        n_codes = len(q["all_icd_codes"])
        is_test_flags.extend([q["is_test"]] * n_codes)
    np.save(str(split_path), np.array(is_test_flags, dtype=bool))

    logger.info("Saved to:")
    logger.info("  {}", settings.training_features_path)
    logger.info("  {}", settings.training_labels_path)
    logger.info("  {}", settings.training_groups_path)


if __name__ == "__main__":
    main()
