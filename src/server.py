"""FastAPI inference server for medical diagnosis.

Loads trained models at startup, serves POST /diagnose endpoint.

Usage:
    uv run uvicorn src.server:app --host 0.0.0.0 --port 8080
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from fastapi.concurrency import run_in_threadpool
from src.inference.engine import DiagnosisEngine
from src.models import Diagnosis, DiagnoseRequest, DiagnoseResponse

engine: DiagnosisEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    print("Loading diagnosis engine...")
    engine = DiagnosisEngine()
    print("Server ready!")
    yield
    engine = None


app = FastAPI(title="Medical Diagnosis API", lifespan=lifespan)


@app.post("/diagnose", response_model=DiagnoseResponse)
async def handle_diagnose(request: DiagnoseRequest) -> DiagnoseResponse:
    """Diagnose based on symptoms text."""
    symptoms = request.symptoms or ""

    if engine is None:
        return DiagnoseResponse(diagnoses=[])

    # Offload CPU-bound inference to a thread pool to avoid blocking the event loop
    results = await run_in_threadpool(engine.diagnose, symptoms)

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
