"""LLM-based reranker using QazCode oss-120b for clinical reasoning.

Takes retriever candidates (protocols + ICD codes) and asks LLM to
select the most appropriate diagnosis using clinical reasoning.
"""

import json
import re

import httpx
from loguru import logger

from src.config import settings


SYSTEM_PROMPT = """Ты — опытный врач-клиницист. Тебе дано описание пациента и список возможных диагнозов с кодами МКБ-10.

Твоя задача: выбрать РОВНО 3 наиболее вероятных диагноза из предложенного списка и расположить их в порядке убывания вероятности.

ВАЖНО:
- Используй ТОЛЬКО коды из предложенного списка
- Отвечай СТРОГО в формате JSON без markdown
- Первый код — наиболее вероятный диагноз
- Учитывай симптомы, анамнез, возраст, пол при выборе

Формат ответа:
{"codes": ["КОД1", "КОД2", "КОД3"]}"""


def build_candidates_text(candidates: list[dict]) -> str:
    """Format candidate codes for LLM prompt."""
    lines = []
    for i, c in enumerate(candidates, 1):
        code = c.get("icd10_code") or c.get("code", "")
        name = c.get("diagnosis") or c.get("code_name", code)
        explanation = c.get("explanation", "")
        lines.append(f"{i}. {code} — {name}")
        if explanation:
            lines.append(f"   Описание: {explanation}")
    return "\n".join(lines)


def _extract_codes_from_text(text: str, valid_codes: set[str]) -> list[str] | None:
    """Try to extract ICD codes from any text (JSON, reasoning, etc.)."""
    # Try 1: Parse as JSON directly
    try:
        parsed = json.loads(text)
        codes = parsed.get("codes", [])
        if codes:
            filtered = [c for c in codes if c in valid_codes]
            if filtered:
                return filtered[:3]
    except (json.JSONDecodeError, AttributeError):
        pass

    # Try 2: Find JSON object with "codes" in text
    json_match = re.search(r'\{[^{}]*"codes"\s*:\s*\[[^\]]*\][^{}]*\}', text)
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
            codes = parsed.get("codes", [])
            filtered = [c for c in codes if c in valid_codes]
            if filtered:
                return filtered[:3]
        except json.JSONDecodeError:
            pass

    # Try 3: Find quoted ICD codes directly in text
    found = []
    for code in valid_codes:
        # Match code in quotes or after numbering
        if re.search(rf'["\']?{re.escape(code)}["\']?', text):
            found.append(code)
    if found:
        # Try to preserve order from text
        positions = []
        for code in found:
            pos = text.find(code)
            positions.append((pos, code))
        positions.sort()
        return [code for _, code in positions][:3]

    return None


def llm_rerank(
    symptoms: str,
    candidates: list[dict],
    timeout: float = 30.0,
) -> list[str] | None:
    """Ask QazCode LLM to select best 3 codes from candidates.

    Returns:
        List of 3 ICD codes in order of likelihood, or None on failure.
    """
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

Выбери 3 наиболее вероятных диагноза из списка выше. Ответь ТОЛЬКО в формате JSON: {{"codes": ["КОД1", "КОД2", "КОД3"]}}"""

    try:
        response = httpx.post(
            f"{settings.qazcode_url}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.qazcode_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": settings.qazcode_model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": 200,
                "temperature": 0.0,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()

        message = data["choices"][0]["message"]

        # Collect all text from response (content + reasoning)
        all_texts = []
        content = message.get("content")
        if content:
            all_texts.append(content)

        reasoning = message.get("reasoning_content")
        if reasoning:
            all_texts.append(reasoning)

        psf = message.get("provider_specific_fields", {})
        if psf:
            for key in ("reasoning", "reasoning_content"):
                val = psf.get(key)
                if val and val not in all_texts:
                    all_texts.append(val)

        if not all_texts:
            logger.warning("LLM returned no content at all")
            return None

        # Try to extract codes from each text, preferring content over reasoning
        for text in all_texts:
            codes = _extract_codes_from_text(text, valid_codes)
            if codes:
                return codes

        # Log first 200 chars for debugging
        combined = " | ".join(t[:100] for t in all_texts)
        logger.warning("LLM: could not extract codes from: {}", combined[:200])
        return None

    except (httpx.HTTPError, KeyError, IndexError) as e:
        logger.warning("LLM rerank failed: {}", e)
        return None
