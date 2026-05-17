# Dockerfile — Optimizado sin pérdida de sub-dependencias en el runtime

# ── STAGE 1: BUILDER (Compilación aislada de Wheels pesados) ──────────────────
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements.txt .

RUN pip wheel --no-cache-dir --no-deps --wheel-dir /build/wheels scikit-learn==1.5.2 numpy==1.26.4


# ── STAGE 2: RUNNER (Imagen limpia final de producción) ───────────────────────
FROM python:3.12-slim AS runtime

RUN groupadd -r appgroup && useradd -r -g appgroup appuser

WORKDIR /app


COPY --from=builder /build/wheels /wheels
COPY requirements.txt .

RUN pip install --no-cache-dir /wheels/* && \
    pip install --no-cache-dir -r requirements.txt

COPY schemas.py analytics.py agent.py main.py ./

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4", "--loop", "uvloop", "--log-level", "info"]