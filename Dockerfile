FROM python:3.12-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY models/ ./models/
COPY data/processed/protocol_features.jsonl ./data/processed/protocol_features.jsonl
COPY data/processed/protocol_summaries.jsonl ./data/processed/protocol_summaries.jsonl
COPY src/ ./src/

ENV PYTHONUNBUFFERED=1

EXPOSE 8080

CMD ["uv", "run", "uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8080"]
