"""LLM-based reranker using QazCode oss-120b for clinical reasoning.

Takes retriever candidates (protocols + ICD codes) and asks LLM to
select the most appropriate diagnosis using clinical reasoning.
"""

import re

from loguru import logger
from openai import OpenAI

from src.config import settings


SYSTEM_PROMPT = """Ты — опытный врач-клиницист. Тебе дано описание пациента и список возможных диагнозов с кодами МКБ-10.

Твоя задача: выбрать РОВНО 3 наиболее вероятных диагноза из предложенного списка.

Правила:
- Используй ТОЛЬКО коды из предложенного списка
- Первый код — самый вероятный основной диагноз
- Обрати внимание на ВСЕ симптомы пациента, не только главную жалобу
- Диагнозы в начале списка уже отсортированы по релевантности — отклоняйся от этого порядка только при наличии веских клинических оснований

В самом конце ответа напиши итог СТРОГО в формате:
РЕЗУЛЬТАТ: КОД1, КОД2, КОД3"""


def build_candidates_text(candidates: list[dict]) -> str:
    """Format candidate codes for LLM prompt."""
    lines = []
    for i, c in enumerate(candidates, 1):
        code = c.get("icd10_code") or c.get("code", "")
        name = c.get("diagnosis") or c.get("code_name", code)
        explanation = c.get("explanation", "")
        lines.append(f"{i}. {code} — {name}")
        if explanation:
            lines.append(f"   {explanation}")
    return "\n".join(lines)


def _extract_codes(text: str, valid_codes: set[str]) -> list[str] | None:
    """Extract ICD codes from LLM response text using multiple strategies."""
    if not text:
        return None

    # Strategy 1: Look for "РЕЗУЛЬТАТ: ..." line
    result_match = re.search(r'РЕЗУЛЬТАТ\s*:\s*(.+)', text, re.IGNORECASE)
    if result_match:
        line = result_match.group(1)
        found = [c for c in valid_codes if c in line]
        if found:
            positions = [(line.find(c), c) for c in found]
            positions.sort()
            return [c for _, c in positions][:3]

    # Strategy 2: Find all valid ICD codes mentioned in text, take last 3
    all_found = []
    for m in re.finditer(r'[A-Z]\d{2}(?:\.\d{1,2})?', text):
        code = m.group(0)
        if code in valid_codes and code not in all_found:
            all_found.append(code)
    if all_found:
        return all_found[-3:] if len(all_found) > 3 else all_found

    return None


# Lazy-initialized client
_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=f"{settings.qazcode_url}/v1",
            api_key=settings.qazcode_key,
        )
    return _client


def llm_rerank(
    symptoms: str,
    candidates: list[dict],
    timeout: float = 30.0,
) -> list[str] | None:
    """Ask QazCode LLM to select best 3 codes from candidates."""
    if not candidates:
        return None

    candidates_text = build_candidates_text(candidates)
    valid_codes = {
        c.get("icd10_code") or c.get("code", "") for c in candidates
    }

    user_prompt = f"""Описание пациента:
{symptoms}

Возможные диагнозы:
{candidates_text}

Выбери 3 наиболее вероятных. В конце напиши: РЕЗУЛЬТАТ: КОД1, КОД2, КОД3"""

    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=settings.qazcode_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=512,
            temperature=0.0,
            timeout=timeout,
        )

        message = response.choices[0].message

        # Collect ALL text from response — openai SDK exposes reasoning_content
        texts = []
        if message.content:
            texts.append(message.content)
        # reasoning_content for reasoning models (o1, oss-120b, etc.)
        if hasattr(message, "reasoning_content") and message.reasoning_content:
            texts.append(message.reasoning_content)
        # Also check provider_specific_fields if present
        if hasattr(message, "provider_specific_fields") and message.provider_specific_fields:
            psf = message.provider_specific_fields
            for key in ("reasoning", "reasoning_content"):
                val = psf.get(key) if isinstance(psf, dict) else getattr(psf, key, None)
                if val and val not in texts:
                    texts.append(val)

        # Try each text source
        for text in texts:
            codes = _extract_codes(text, valid_codes)
            if codes:
                logger.info("LLM extracted {} codes from {} chars", len(codes), len(text))
                return codes

        if texts:
            logger.warning("LLM: no codes found in {} chars of response", sum(len(t) for t in texts))
            # Log snippet for debugging
            for i, t in enumerate(texts):
                logger.debug("LLM text[{}] (first 200): {}", i, t[:200])
        else:
            logger.warning("LLM returned completely empty response")
        return None

    except Exception as e:
        logger.warning("LLM rerank failed: {}", e)
        return None
