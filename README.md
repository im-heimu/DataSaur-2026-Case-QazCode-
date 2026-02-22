# DataSaur — Medical Diagnosis (Symptoms → ICD-10)

Система диагностики: свободный текст симптомов → топ-3 диагноза с кодами МКБ-10.
Основана на 1137 клинических протоколах Казахстана.

## Архитектура

1. **Retriever** — fine-tuned `multilingual-e5-base`, semantic search по протоколам
2. **Code ranker** — protocol-first ordering + embedding/TF-IDF tiebreaker внутри протокола
3. **LLM reranker** (опционально) — QazCode `oss-120b` переранжирует топ-15 кандидатов

## Быстрый старт

### Требования
- Python 3.12
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- ~1.5GB на модели (`models/`)

### Установка

```bash
git clone <repo-url> && cd hack-nu
uv sync
```

> **Модели не включены в репозиторий** (~800MB). Скачай архив `models.tar.gz` отдельно и распакуй:
> ```bash
> tar xzf models.tar.gz
> # В корне проекта должна появиться папка models/
> ```

### Запуск сервера

```bash
uv run uvicorn src.server:app --host 0.0.0.0 --port 8080
```

Эндпоинт: `POST /diagnose`

```json
{"symptoms": "боль в груди, одышка, повышение температуры"}
```

Ответ:

```json
{
  "diagnoses": [
    {"rank": 1, "diagnosis": "...", "icd10_code": "J18.9", "explanation": "..."},
    {"rank": 2, "diagnosis": "...", "icd10_code": "J15.9", "explanation": "..."},
    {"rank": 3, "diagnosis": "...", "icd10_code": "J44.1", "explanation": "..."}
  ]
}
```

### Docker

```bash
docker build -t diagnosis .
docker run -p 8080:8080 diagnosis
```

Контейнер включает все модели и данные, внешних вызовов нет.

Для LLM-реранкинга (опционально):

```bash
docker run -p 8080:8080 -e QAZCODE_ENABLED=true -e QAZCODE_KEY=<key> diagnosis
```

## Бенчмарк

```bash
# Запустить сервер (в одном терминале):
uv run uvicorn src.server:app --host 127.0.0.1 --port 8080

# Eval (в другом терминале):
uv run python evaluate.py -e http://127.0.0.1:8080/diagnose -d ./data/test_set -n test
```

Результаты → `data/evals/test_metrics.json`

Локальный eval без HTTP (быстрее):

```bash
uv run python -m src.training.evaluate_local            # без LLM
uv run python -m src.training.evaluate_local --llm       # с LLM
uv run python -m src.training.evaluate_local --optimize   # подбор весов
```

### Метрики (test_set, 221 кейс)

| Метрика | Значение |
|---------|----------|
| Accuracy@1 | ~10% |
| Recall@3 | ~43% |
| Avg latency | 0.020s |

## Пайплайн обучения (воспроизведение)

Если нужно переобучить с нуля:

```bash
# 1. Извлечь фичи из протоколов (нужен GPT-OSS ключ в .env)
uv run python -m src.data_prep.extract_features
uv run python -m src.data_prep.generate_summaries
uv run python -m src.data_prep.generate_synthetic

# 2. Обучить retriever (~25 мин на GPU)
uv run python -m src.training.train_retriever

# 3. Экспорт моделей для inference
uv run python -m src.training.export_model

# 4. Оптимизация весов
uv run python -m src.training.evaluate_local --optimize
# → обновить w_code_embedding / w_code_tfidf в src/config.py
```

## Структура

```
models/              # retriever, embeddings, TF-IDF, protocol data (~800MB)
data/test_set/       # 221 тестовый кейс (query + gt + icd_codes)
data/processed/      # извлечённые фичи, синтетика, саммари
corpus/              # исходные клинические протоколы
src/
  server.py          # FastAPI сервер (POST /diagnose)
  config.py          # все настройки и пути
  inference/
    engine.py        # DiagnosisEngine — основной пайплайн
    retriever.py     # semantic retriever
    llm_ranker.py    # LLM reranker (QazCode oss-120b)
  training/
    train_retriever.py    # обучение bi-encoder
    evaluate_local.py     # локальный eval + оптимизация весов
    export_model.py       # экспорт для inference
  data_prep/         # подготовка данных (extract, summarize, synthetic)
evaluate.py          # HTTP-based eval скрипт
Dockerfile           # production image
```

## Конфигурация

Через env-переменные или `.env` файл:

| Переменная | По умолчанию | Описание |
|-----------|-------------|----------|
| `QAZCODE_ENABLED` | `false` | Включить LLM реранкинг |
| `QAZCODE_KEY` | — | API ключ QazCode |
| `QAZCODE_URL` | `https://hub.qazcode.ai` | URL API |
