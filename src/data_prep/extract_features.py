"""Step 1: Extract features from clinical protocols using GPT-OSS.

For each protocol with ICD codes, sends text to GPT-OSS to extract:
- disease name, symptoms, diagnostic criteria, body system, patient category
- per-ICD-code descriptions with distinguishing features

Usage:
    uv run python -m src.data_prep.extract_features
"""

import asyncio
import json
import re

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

from src.config import (
    CORPUS_PATH,
    GPT_OSS_CONCURRENCY,
    GPT_OSS_KEY,
    GPT_OSS_MODEL,
    GPT_OSS_URL,
    PROTOCOL_FEATURES_PATH,
    TRUNCATION_MARKERS,
)


def truncate_protocol_text(text: str, max_chars: int = 8000) -> str:
    """Truncate protocol text to diagnostic sections only."""
    earliest_pos = len(text)
    for marker in TRUNCATION_MARKERS:
        pos = text.find(marker)
        if pos != -1 and pos < earliest_pos:
            earliest_pos = pos

    truncated = text[:earliest_pos].strip()
    if len(truncated) > max_chars:
        truncated = truncated[:max_chars]
    return truncated


def load_protocols() -> list[dict]:
    """Load protocols that have ICD codes."""
    protocols = []
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            if p.get("icd_codes") and len(p["icd_codes"]) > 0:
                protocols.append(p)
    return protocols


SYSTEM_PROMPT = """Ты — медицинский эксперт-аналитик. Проанализируй клинический протокол и извлеки информацию строго в JSON формате. Отвечай ТОЛЬКО валидным JSON без markdown-разметки."""

USER_PROMPT_TEMPLATE = """Проанализируй клинический протокол и извлеки следующую информацию в JSON:

1. "disease_name": Название заболевания (строка)
2. "symptoms": Список ключевых симптомов и жалоб пациентов [массив строк, 5-15 пунктов]
3. "diagnostic_criteria": Основные критерии диагностики (строка, 2-3 предложения)
4. "body_system": Система органов (одна из: нервная, сердечно-сосудистая, дыхательная, пищеварительная, мочеполовая, эндокринная, костно-мышечная, кожа, кровь, инфекции, психиатрия, травмы, другое)
5. "patient_category": Категория пациентов (дети/взрослые/все)
6. "icd_code_descriptions": Для КАЖДОГО МКБ-10 кода из списка ниже:
   - "code": код МКБ-10
   - "name": название диагноза
   - "distinguishing_features": чем этот код отличается от других кодов протокола (1-2 предложения)

МКБ-10 коды протокола: {icd_codes}

Текст протокола:
{text}

Верни ТОЛЬКО валидный JSON объект."""


def strip_markdown(content: str) -> str:
    """Strip markdown code block wrappers if present."""
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1] if "\n" in content else content[3:]
    if content.endswith("```"):
        content = content[:-3]
    return content.strip()


def repair_truncated_json(text: str) -> str:
    """Attempt to repair JSON truncated by max_tokens.

    Strategy: walk backwards to find the last complete JSON element,
    then close all open brackets.
    """
    # Step 1: Find last complete object in icd_code_descriptions array
    # Look for last "}," or "}" that closes an array element
    last_complete = -1
    brace_depth = 0
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            brace_depth += 1
        elif ch == "}":
            brace_depth -= 1
            if brace_depth >= 0:
                last_complete = i

    # Step 2: Truncate to last complete closing brace
    if last_complete > 0:
        text = text[: last_complete + 1]

    # Step 3: Remove any trailing comma
    text = text.rstrip()
    if text.endswith(","):
        text = text[:-1]

    # Step 4: Count unclosed brackets and close them
    opens = 0
    open_sq = 0
    in_str = False
    esc = False
    for ch in text:
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            opens += 1
        elif ch == "}":
            opens -= 1
        elif ch == "[":
            open_sq += 1
        elif ch == "]":
            open_sq -= 1

    text += "]" * open_sq + "}" * opens
    return text


def parse_json_response(content: str) -> dict | None:
    """Parse JSON from GPT response, with fallback repair for truncated output."""
    content = strip_markdown(content)

    # Try direct parse
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Try repair
    try:
        repaired = repair_truncated_json(content)
        return json.loads(repaired)
    except json.JSONDecodeError:
        return None


async def extract_features_for_protocol(
    client: AsyncOpenAI, protocol: dict, max_retries: int = 3
) -> dict | None:
    """Extract features from a single protocol using GPT-OSS."""
    text = truncate_protocol_text(protocol["text"])
    icd_codes = protocol["icd_codes"]

    # For protocols with many codes, limit to avoid output truncation
    max_tokens = 8192 if len(icd_codes) > 20 else 4096

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
                temperature=0.1,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content.strip()
            features = parse_json_response(content)

            if features is None:
                raise json.JSONDecodeError("Failed to parse", content[:100], 0)

            features["protocol_id"] = protocol["protocol_id"]
            features["icd_codes"] = icd_codes
            return features

        except json.JSONDecodeError:
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** (attempt + 1))
                continue
            print(
                f"  Failed to parse JSON for {protocol['protocol_id']} after {max_retries} attempts"
            )
            return None
        except Exception as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** (attempt + 1))
                continue
            print(f"  API error for {protocol['protocol_id']}: {e}")
            return None


async def main():
    PROTOCOL_FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Load already processed protocols for resumability
    processed_ids = set()
    if PROTOCOL_FEATURES_PATH.exists():
        with open(PROTOCOL_FEATURES_PATH, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                processed_ids.add(data["protocol_id"])
        print(f"Resuming: {len(processed_ids)} already processed")

    protocols = load_protocols()
    print(f"Total protocols with ICD codes: {len(protocols)}")

    remaining = [p for p in protocols if p["protocol_id"] not in processed_ids]
    print(f"Remaining to process: {len(remaining)}")

    if not remaining:
        print("All protocols already processed!")
        return

    client = AsyncOpenAI(base_url=GPT_OSS_URL, api_key=GPT_OSS_KEY)
    semaphore = asyncio.Semaphore(GPT_OSS_CONCURRENCY)

    success_count = 0
    fail_count = 0
    lock = asyncio.Lock()

    out_f = open(PROTOCOL_FEATURES_PATH, "a", encoding="utf-8")

    async def process_one(protocol: dict):
        nonlocal success_count, fail_count
        async with semaphore:
            features = await extract_features_for_protocol(client, protocol)
        async with lock:
            if features:
                out_f.write(json.dumps(features, ensure_ascii=False) + "\n")
                out_f.flush()
                success_count += 1
            else:
                fail_count += 1

    tasks = [process_one(p) for p in remaining]
    await tqdm_asyncio.gather(*tasks, desc="Extracting features")

    out_f.close()
    await client.close()

    print(f"\nDone! Success: {success_count}, Failed: {fail_count}")
    print(f"Output: {PROTOCOL_FEATURES_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
