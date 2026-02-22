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
- Первый код — наиболее вероятный диагноз
- Учитывай симптомы, анамнез, возраст, пол при выборе"""

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "diagnosis_response",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "codes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "3 ICD-10 codes in order of likelihood",
                },
            },
            "required": ["codes"],
            "additionalProperties": False,
        },
    },
}


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

Выбери 3 наиболее вероятных диагноза из списка выше."""

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
                "response_format": RESPONSE_FORMAT,
                "max_tokens": 200,
                "temperature": 0.0,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()

        message = data["choices"][0]["message"]
        content = message.get("content") or ""

        if not content.strip():
            logger.warning("LLM returned empty content")
            return None

        parsed = json.loads(content)
        codes = parsed.get("codes", [])

        if codes and isinstance(codes, list):
            filtered = [c for c in codes if c in valid_codes]
            if filtered:
                return filtered[:3]

        logger.warning("LLM returned no valid codes from: {}", codes)
        return None

    except json.JSONDecodeError as e:
        # Fallback: try to find codes in raw content
        logger.warning("LLM JSON parse failed: {}, trying regex fallback", e)
        try:
            text = data["choices"][0]["message"].get("content") or ""
            found = [c for c in valid_codes if c in text]
            if found:
                positions = [(text.find(c), c) for c in found]
                positions.sort()
                return [c for _, c in positions][:3]
        except Exception:
            pass
        return None

    except (httpx.HTTPError, KeyError, IndexError) as e:
        logger.warning("LLM rerank failed: {}", e)
        return None
