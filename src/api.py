"""
api.py - AegisNet REST API (FastAPI).

Exposes the AegisNetPredictor as a production REST service.

Run:
    uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
"""

import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

from src.predict import AegisNetPredictor, ANOMALY_THRESHOLD

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("aegisnet.api")


# ── Lazy model holder ─────────────────────────────────────────────────────────
# We store the predictor instance on app state so it is created once
# at startup and reused across every request (avoids repeated disk I/O).
predictor: AegisNetPredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup; release on shutdown."""
    global predictor
    logger.info("Loading AegisNet model ...")
    try:
        predictor = AegisNetPredictor()
        logger.info("Model ready. API is live.")
    except Exception as exc:
        logger.error(f"Failed to load model: {exc}")
        predictor = None   # API will return 503 on prediction requests
    yield
    logger.info("Shutting down AegisNet API.")
    predictor = None


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="AegisNet Anomaly Detection API",
    description=(
        "Real-time network anomaly detection using a trained Autoencoder. "
        "POST flow features to /predict and receive an anomaly score."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    """Input payload for the /predict endpoint."""
    features: list[float]

    @field_validator("features")
    @classmethod
    def features_must_not_be_empty(cls, v: list[float]) -> list[float]:
        if len(v) == 0:
            raise ValueError("'features' must not be empty.")
        return v


class PredictResponse(BaseModel):
    """Output payload returned by /predict."""
    anomaly_score: float
    is_anomaly:    bool


class HealthResponse(BaseModel):
    status:      str
    model_ready: bool


# ── Middleware: request timing ─────────────────────────────────────────────────

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Attach X-Process-Time header (milliseconds) to every response."""
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time"] = f"{elapsed_ms:.2f}ms"
    return response


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check():
    """
    Health check endpoint.

    Returns HTTP 200 when the service is running.
    model_ready indicates whether the ML model loaded successfully.
    """
    return HealthResponse(
        status="ok",
        model_ready=predictor is not None,
    )


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict(payload: PredictRequest):
    """
    Run anomaly detection on a single network flow.

    **Body**:
    ```json
    { "features": [float, float, ...] }
    ```

    **Response**:
    ```json
    {
      "anomaly_score": 0.123456,
      "is_anomaly": false
    }
    ```

    - `anomaly_score`: MSE reconstruction error (higher = more anomalous).
    - `is_anomaly`: True if score exceeds the configured threshold.
    """
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Check server logs for startup errors.",
        )

    try:
        result = predictor.predict(payload.features)
    except ValueError as exc:
        # Input validation failure (wrong dim, NaN, Inf)
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("Unexpected inference error")
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")

    logger.info(
        f"score={result['anomaly_score']:.6f}  "
        f"anomaly={result['is_anomaly']}  "
        f"threshold={result['threshold']}"
    )

    return PredictResponse(
        anomaly_score=result["anomaly_score"],
        is_anomaly=result["is_anomaly"],
    )


# ── Global exception handler ─────────────────────────────────────────────────

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled exception on {request.url}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. See server logs."},
    )
