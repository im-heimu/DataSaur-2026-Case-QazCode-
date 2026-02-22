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

Выбери РОВНО 3 наиболее вероятных диагноза из предложенного списка.
Используй ТОЛЬКО коды из списка.

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
            # Preserve order from line
            positions = [(line.find(c), c) for c in found]
            positions.sort()
            return [c for _, c in positions][:3]

    # Strategy 2: Find JSON with "codes" key
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

    # Strategy 3: Find all valid ICD codes mentioned in text, take last 3
    # (LLM typically concludes with final answer)
    all_found = []
    for m in re.finditer(r'[A-Z]\d{2}(?:\.\d{1,2})?', text):
        code = m.group(0)
        if code in valid_codes and code not in all_found:
            all_found.append(code)
    if all_found:
        # Take last 3 unique codes (conclusion usually at end)
        return all_found[-3:] if len(all_found) > 3 else all_found

    return None


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
                "max_tokens": 512,
                "temperature": 0.0,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()

        message = data["choices"][0]["message"]

        # Collect ALL text from response
        texts = []
        for key in ("content", "reasoning_content"):
            val = message.get(key)
            if val:
                texts.append(val)
        psf = message.get("provider_specific_fields", {})
        if psf:
            for key in ("reasoning", "reasoning_content"):
                val = psf.get(key)
                if val and val not in texts:
                    texts.append(val)

        # Try each text source
        for text in texts:
            codes = _extract_codes(text, valid_codes)
            if codes:
                return codes

        if texts:
            logger.warning("LLM: no codes found in {} chars of response", sum(len(t) for t in texts))
        else:
            logger.warning("LLM returned completely empty response")
        return None

    except (httpx.HTTPError, KeyError, IndexError) as e:
        logger.warning("LLM rerank failed: {}", e)
        return None
