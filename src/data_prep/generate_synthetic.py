"""Step 2: Generate synthetic training data using GPT-OSS.

For each (protocol, icd_code) pair, generates patient complaint variations.
Test set protocols get more variations (10-15) than others (3-5).

Usage:
    uv run python -m src.data_prep.generate_synthetic
"""

import asyncio
import json

from loguru import logger
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

from src.config import settings, setup_logging


SYSTEM_PROMPT = """Ты — пациент, который описывает свои жалобы врачу. Пиши естественным разговорным русским языком от первого лица."""

USER_PROMPT_TEMPLATE = """Сгенерируй {n} вариантов жалоб пациента для следующего диагноза. Каждая жалоба — это текст от 400 до 1000 символов, написанный от первого лица.

Диагноз: {disease_name}
МКБ-10 код: {code} — {code_name}
Ключевые симптомы: {symptoms}
Отличительные признаки этого кода: {distinguishing_features}

Требования:
- Разный возраст, пол и стиль изложения
- Бытовые подробности (когда началось, как развивалось)
- Естественная речь, не медицинские термины
- Каждый вариант уникален по сценарию

Верни ТОЛЬКО JSON массив строк, без markdown-разметки. Пример: ["жалоба 1", "жалоба 2"]"""


def load_test_protocol_ids() -> set[str]:
    """Load protocol IDs from the test set."""
    ids = set()
    if settings.test_set_dir.exists():
        for f in settings.test_set_dir.glob("*.json"):
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
                ids.add(data["protocol_id"])
    return ids


def load_protocol_features() -> list[dict]:
    """Load extracted protocol features."""
    features = []
    with open(settings.protocol_features_path, "r", encoding="utf-8") as f:
        for line in f:
            features.append(json.loads(line))
    return features


async def generate_queries(
    client: AsyncOpenAI,
    disease_name: str,
    code: str,
    code_name: str,
    symptoms: list[str],
    distinguishing_features: str,
    n: int,
    max_retries: int = 3,
) -> list[str] | None:
    """Generate synthetic patient queries for a (protocol, icd_code) pair."""
    prompt = USER_PROMPT_TEMPLATE.format(
        n=n,
        disease_name=disease_name,
        code=code,
        code_name=code_name,
        symptoms=", ".join(symptoms[:10]),
        distinguishing_features=distinguishing_features,
    )

    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=settings.gpt_oss_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.9,
                max_tokens=4096,
            )
            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()

            queries = json.loads(content)
            if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
                return queries
            return None

        except json.JSONDecodeError:
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** (attempt + 1))
                continue
            return None
        except Exception as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** (attempt + 1))
                continue
            logger.error("API error: {}", e)
            return None


async def main():
    setup_logging()
    settings.synthetic_training_path.parent.mkdir(parents=True, exist_ok=True)

    # Load already generated pairs for resumability
    processed_pairs = set()
    if settings.synthetic_training_path.exists():
        with open(settings.synthetic_training_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                processed_pairs.add(
                    (data["protocol_id"], data["target_icd_code"])
                )
        logger.info("Resuming: {} pairs already processed", len(processed_pairs))

    test_protocol_ids = load_test_protocol_ids()
    logger.info("Test set protocol IDs: {}", len(test_protocol_ids))

    protocol_features = load_protocol_features()
    logger.info("Protocols with features: {}", len(protocol_features))

    # Build work items
    all_pairs = []
    for pf in protocol_features:
        pid = pf["protocol_id"]
        is_test = pid in test_protocol_ids
        code_descriptions = pf.get("icd_code_descriptions", [])
        icd_codes = pf.get("icd_codes", [])

        code_info = {}
        for cd in code_descriptions:
            code_info[cd.get("code", "")] = cd

        for code in icd_codes:
            if (pid, code) in processed_pairs:
                continue
            all_pairs.append((pf, code, code_info.get(code, {}), is_test))

    logger.info("Remaining (protocol, code) pairs: {}", len(all_pairs))

    if not all_pairs:
        logger.info("All pairs already processed!")
        return

    client = AsyncOpenAI(base_url=settings.gpt_oss_url, api_key=settings.gpt_oss_key)
    semaphore = asyncio.Semaphore(settings.gpt_oss_concurrency)

    total_generated = 0
    total_pairs = 0
    lock = asyncio.Lock()

    out_f = open(settings.synthetic_training_path, "a", encoding="utf-8")

    async def process_one(pf: dict, code: str, ci: dict, is_test: bool):
        nonlocal total_generated, total_pairs
        pid = pf["protocol_id"]
        n = 12 if is_test else 4
        disease_name = pf.get("disease_name", "Unknown")
        symptoms = pf.get("symptoms", [])
        code_name = ci.get("name", code)
        distinguishing = ci.get("distinguishing_features", "")

        async with semaphore:
            queries = await generate_queries(
                client, disease_name, code, code_name, symptoms, distinguishing, n
            )

        async with lock:
            if queries:
                for query in queries:
                    record = {
                        "query": query,
                        "protocol_id": pid,
                        "target_icd_code": code,
                        "all_icd_codes": pf.get("icd_codes", []),
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_generated += len(queries)
            total_pairs += 1
            out_f.flush()

    tasks = [process_one(pf, code, ci, is_test) for pf, code, ci, is_test in all_pairs]
    await tqdm_asyncio.gather(*tasks, desc="Generating synthetic data")

    out_f.close()
    await client.close()

    logger.info("Done! Pairs processed: {}, Queries generated: {}", total_pairs, total_generated)
    logger.info("Output: {}", settings.synthetic_training_path)


if __name__ == "__main__":
    asyncio.run(main())
