"""
main.py — Servidor FastAPI asíncrono de producción.

Arquitectura del pipeline por request:
  HTTP POST → Validación Pydantic → Ridge (ThreadPool) → LLM (async) → Response

El middleware X-Process-Time-Ms captura latencia end-to-end en milisegundos,
permitiendo monitorización de SLAs sin herramientas externas en fase MVP.
"""

from __future__ import annotations

import logging
import time

import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from agent import generate_biometric_report
from analytics import BiohackingAnalyticsEngine
from schemas import BiometricReportResponse, WeeklyPredictionInput

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Aplicación FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Deep-Biohacking Tracker API",
    description=(
        "Motor predictivo híbrido: Ridge Regression local (coste $0) + "
        "interpretación semántica LLM (microcéntimos por request)."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS permisivo para fase MVP. En producción, restringir a dominios conocidos.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Singleton del motor analítico
# ---------------------------------------------------------------------------

# Se instancia una vez en el módulo: no hay estado mutable entre requests,
# por lo que es thread-safe para uso concurrente con uvicorn workers.
analytics_engine = BiohackingAnalyticsEngine()


# ---------------------------------------------------------------------------
# Middleware — Latencia en cabecera X-Process-Time-Ms
# ---------------------------------------------------------------------------

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """
    Middleware que inyecta el tiempo de procesamiento en milisegundos
    en la cabecera X-Process-Time-Ms de cada respuesta HTTP.

    Útil para detectar regresiones de rendimiento en CI/CD y para que
    el bot de Telegram muestre la latencia real del sistema al usuario.
    """
    start_ns = time.perf_counter_ns()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
    return response


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Infra"])
async def health_check() -> dict:
    """Endpoint de salud para liveness probes de Docker/Kubernetes."""
    return {"status": "ok", "service": "deep-biohacking-tracker"}


@app.post(
    "/api/v1/predict-performance",
    response_model=BiometricReportResponse,
    status_code=status.HTTP_200_OK,
    summary="Genera reporte biométrico predictivo",
    tags=["Predicción"],
)
async def predict_performance(
    payload: WeeklyPredictionInput,
) -> BiometricReportResponse:
    """
    Pipeline completo de predicción de rendimiento en un solo endpoint:

    1. **Validación** — Pydantic v2 rechaza payloads malformados automáticamente.
    2. **Analytics** — Ridge Regression en ThreadPool (no bloquea event loop).
    3. **LLM Reasoning** — El agente interpreta coeficientes semánticamente.
    4. **Response** — BiometricReportResponse validado y serializado.

    Coste estimado por request: ~$0.00003 USD (solo tokens del paso 3).
    """
    logger.info(
        "Iniciando predicción | user_id=%s | logs=%d",
        payload.user_id,
        len(payload.logs),
    )

    try:
        # — PASO 1: Capa ML local (coste $0, ejecuta en ThreadPool) —
        analytics_output = await analytics_engine.compute_insights(payload.logs)
        logger.info(
            "Analytics completado | user_id=%s | predicted_rpe=%.2f",
            payload.user_id,
            analytics_output["predicted_rpe_next_week"],
        )

        # — PASO 2: Capa de razonamiento LLM (coste de microcéntimos) —
        report = await generate_biometric_report(analytics_output)
        logger.info(
            "Reporte generado | user_id=%s | rendimiento=%d%%",
            payload.user_id,
            report.rendimiento_estimado_porcentaje,
        )

        return report

    except ValueError as exc:
        # Errores de validación interna (ej: datos insuficientes para Ridge)
        logger.warning("Error de validación: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        # Captura genérica: loguea el error completo y devuelve 500 limpio.
        logger.exception("Error inesperado en predict_performance: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error interno del servidor. Consulta los logs del sistema.",
        ) from exc


# ---------------------------------------------------------------------------
# Entrypoint de desarrollo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # En producción, usar: uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,          # Hot-reload sólo para desarrollo local
        log_level="info",
    )
