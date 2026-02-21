"""Step 3: Generate protocol summaries for retrieval using GPT-OSS.

Creates concise 200-300 word summaries focused on symptoms and diagnostic criteria.

Usage:
    uv run python -m src.data_prep.generate_summaries
"""

import asyncio
import json

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

from src.config import (
    CORPUS_PATH,
    GPT_OSS_CONCURRENCY,
    GPT_OSS_KEY,
    GPT_OSS_MODEL,
    GPT_OSS_URL,
    PROTOCOL_SUMMARIES_PATH,
    TRUNCATION_MARKERS,
)


SYSTEM_PROMPT = """Ты — медицинский эксперт. Создай краткую сводку клинического протокола, фокусируясь на жалобах пациентов и диагностических критериях. Пиши на русском языке."""

USER_PROMPT_TEMPLATE = """Создай краткую сводку (200-300 слов) клинического протокола. Сфокусируйся на:
1. Название заболевания
2. Основные жалобы пациентов (симптомы, с которыми приходят)
3. Ключевые диагностические критерии
4. Какие МКБ-10 коды покрывает протокол

МКБ-10 коды: {icd_codes}

Текст протокола:
{text}

Пиши сплошным текстом, без заголовков и списков. Начни с названия заболевания."""


def truncate_text(text: str, max_chars: int = 6000) -> str:
    """Truncate to diagnostic sections."""
    earliest_pos = len(text)
    for marker in TRUNCATION_MARKERS:
        pos = text.find(marker)
        if pos != -1 and pos < earliest_pos:
            earliest_pos = pos
    truncated = text[:earliest_pos].strip()
    if len(truncated) > max_chars:
        truncated = truncated[:max_chars]
    return truncated


async def generate_summary(
    client: AsyncOpenAI, protocol: dict, max_retries: int = 3
) -> dict | None:
    """Generate a summary for a single protocol."""
    text = truncate_text(protocol["text"])
    icd_codes = protocol.get("icd_codes", [])
    prompt = USER_PROMPT_TEMPLATE.format(
        icd_codes=json.dumps(icd_codes, ensure_ascii=False),
        text=text,
    )

    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=GPT_OSS_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=1024,
            )
            summary = response.choices[0].message.content.strip()
            return {
                "protocol_id": protocol["protocol_id"],
                "summary": summary,
                "icd_codes": icd_codes,
                "source_file": protocol.get("source_file", ""),
            }
        except Exception as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** (attempt + 1))
            else:
                print(f"  Failed for {protocol['protocol_id']}: {e}")
                return None


async def main():
    PROTOCOL_SUMMARIES_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Load already processed
    processed_ids = set()
    if PROTOCOL_SUMMARIES_PATH.exists():
        with open(PROTOCOL_SUMMARIES_PATH, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                processed_ids.add(data["protocol_id"])
        print(f"Resuming: {len(processed_ids)} already processed")

    # Load all protocols
    protocols = []
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            protocols.append(json.loads(line))
    print(f"Total protocols: {len(protocols)}")

    remaining = [p for p in protocols if p["protocol_id"] not in processed_ids]
    print(f"Remaining to process: {len(remaining)}")

    if not remaining:
        print("All done!")
        return

    client = AsyncOpenAI(base_url=GPT_OSS_URL, api_key=GPT_OSS_KEY)
    semaphore = asyncio.Semaphore(GPT_OSS_CONCURRENCY)

    success = 0
    lock = asyncio.Lock()

    out_f = open(PROTOCOL_SUMMARIES_PATH, "a", encoding="utf-8")

    async def process_one(protocol: dict):
        nonlocal success
        async with semaphore:
            record = await generate_summary(client, protocol)
        async with lock:
            if record:
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
                success += 1

    tasks = [process_one(p) for p in remaining]
    await tqdm_asyncio.gather(*tasks, desc="Generating summaries")

    out_f.close()
    await client.close()

    print(f"\nDone! Summaries generated: {success}")
    print(f"Output: {PROTOCOL_SUMMARIES_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
