"""FastAPI inference server for medical diagnosis.

Loads trained models at startup, serves POST /diagnose endpoint.

Usage:
    uv run uvicorn src.server:app --host 0.0.0.0 --port 8080
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger

from src.config import setup_logging
from src.inference.engine import DiagnosisEngine
from src.models import Diagnosis, DiagnoseRequest, DiagnoseResponse

engine: DiagnosisEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    setup_logging()
    logger.info("Loading diagnosis engine...")
    engine = DiagnosisEngine()
    logger.info("Server ready!")
    yield
    engine = None


app = FastAPI(title="Medical Diagnosis API", lifespan=lifespan)


@app.post("/diagnose", response_model=DiagnoseResponse)
async def handle_diagnose(request: DiagnoseRequest) -> DiagnoseResponse:
    """Diagnose based on symptoms text."""
    symptoms = request.symptoms or ""

    results = engine.diagnose(symptoms)

    diagnoses = [
        Diagnosis(
            rank=r["rank"],
            diagnosis=r["diagnosis"],
            icd10_code=r["icd10_code"],
            explanation=r["explanation"],
        )
        for r in results
    ]

    # Ensure at least 3 results (pad with empty if needed)
    while len(diagnoses) < 3:
        diagnoses.append(
            Diagnosis(
                rank=len(diagnoses) + 1,
                diagnosis="Unknown",
                icd10_code="Z99",
                explanation="Insufficient data for diagnosis",
            )
        )

    return DiagnoseResponse(diagnoses=diagnoses)
