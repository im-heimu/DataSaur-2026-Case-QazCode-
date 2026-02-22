"""LLM-based reranker using QazCode oss-120b for clinical reasoning.

Takes retriever candidates (protocols + ICD codes) and asks LLM to
select the most appropriate diagnosis using clinical reasoning.
"""

import json

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


def llm_rerank(
    symptoms: str,
    candidates: list[dict],
    timeout: float = 15.0,
) -> list[str] | None:
    """Ask QazCode LLM to select best 3 codes from candidates.

    Args:
        symptoms: patient symptom text
        candidates: list of candidate dicts with icd10_code, diagnosis, explanation
        timeout: request timeout in seconds

    Returns:
        List of 3 ICD codes in order of likelihood, or None on failure.
    """
    if not candidates:
        return None

    candidates_text = build_candidates_text(candidates)

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
                "max_tokens": 100,
                "temperature": 0.0,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()

        message = data["choices"][0]["message"]
        content = message.get("content") or ""
        # Reasoning models may put answer in reasoning_content
        if not content.strip():
            content = (
                message.get("reasoning_content")
                or message.get("provider_specific_fields", {}).get("reasoning_content")
                or ""
            )
        content = content.strip()

        if not content:
            logger.warning("LLM returned empty content")
            return None

        # Parse JSON from response (handle markdown wrapping)
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        # Try to extract JSON from mixed text
        if not content.startswith("{"):
            import re
            json_match = re.search(r'\{[^}]*"codes"\s*:\s*\[[^\]]*\][^}]*\}', content)
            if json_match:
                content = json_match.group(0)

        parsed = json.loads(content)
        codes = parsed.get("codes", [])

        if codes and isinstance(codes, list):
            # Validate codes are from candidates
            valid_codes = {
                c.get("icd10_code") or c.get("code", "") for c in candidates
            }
            filtered = [c for c in codes if c in valid_codes]
            if filtered:
                return filtered[:3]

        logger.warning("LLM returned invalid codes: {}", codes)
        return None

    except (httpx.HTTPError, json.JSONDecodeError, KeyError, IndexError) as e:
        logger.warning("LLM rerank failed: {}", e)
        return None
