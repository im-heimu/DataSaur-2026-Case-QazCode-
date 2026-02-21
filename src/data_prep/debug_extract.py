"""Debug: see raw GPT-OSS response for a failing protocol."""

import asyncio
import json

from openai import AsyncOpenAI

from src.config import GPT_OSS_KEY, GPT_OSS_MODEL, GPT_OSS_URL, PROTOCOL_FEATURES_PATH, CORPUS_PATH
from src.data_prep.extract_features import (
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    truncate_protocol_text,
)


async def main():
    # Find a failing protocol
    processed_ids = set()
    if PROTOCOL_FEATURES_PATH.exists():
        with open(PROTOCOL_FEATURES_PATH, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                processed_ids.add(data["protocol_id"])

    # Load one failing protocol
    target = None
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            if p.get("icd_codes") and p["protocol_id"] not in processed_ids:
                target = p
                break

    if not target:
        print("No failing protocols found")
        return

    print(f"Protocol: {target['protocol_id']}")
    print(f"ICD codes: {target['icd_codes']}")
    print(f"Text length: {len(target['text'])}")

    text = truncate_protocol_text(target["text"])
    prompt = USER_PROMPT_TEMPLATE.format(
        icd_codes=json.dumps(target["icd_codes"], ensure_ascii=False),
        text=text,
    )

    client = AsyncOpenAI(base_url=GPT_OSS_URL, api_key=GPT_OSS_KEY)
    response = await client.chat.completions.create(
        model=GPT_OSS_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=4096,
    )
    content = response.choices[0].message.content
    print(f"\n=== RAW RESPONSE ({len(content)} chars) ===")
    print(content[:3000])
    print("=== END ===")
    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
