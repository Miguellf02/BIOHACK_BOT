# =============================================================================
# Dockerfile — Multi-stage build para Deep-Biohacking Tracker
#
# Estrategia de optimización:
#   Stage 1 (builder): Instala dependencias con todas las herramientas de
#     compilación disponibles (gcc, build-essential para scikit-learn/numpy).
#   Stage 2 (runtime): Imagen mínima python:3.12-slim sin compiladores.
#     Sólo copia los paquetes ya compilados del stage anterior.
#
# Resultado: imagen final ~60% más ligera que un build single-stage.
# =============================================================================

# ─────────────────────────────────────────────
# Stage 1: Builder — Compilación de dependencias
# ─────────────────────────────────────────────
FROM python:3.12-slim AS builder

# Instalar herramientas de compilación necesarias para scikit-learn y numpy.
# Se instalan SÓLO en este stage; no contaminarán la imagen final.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copiar requirements primero para aprovechar la caché de capas de Docker.
# Si el código cambia pero requirements.txt no, esta capa no se reconstruye.
COPY requirements.txt .

# Instalar paquetes en un directorio local para copiarlos al stage final.
# --no-cache-dir: evita almacenar la caché de pip en la imagen.
# --prefix=/install: instala en un path aislado que copiaremos al runtime.
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ─────────────────────────────────────────────
# Stage 2: Runtime — Imagen de producción mínima
# ─────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# Usuario no-root para seguridad en producción.
# Muchos escáneres de vulnerabilidades (Trivy, Snyk) marcan como HIGH
# los contenedores que corren como root.
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

WORKDIR /app

# Copiar SÓLO los paquetes compilados del stage builder.
# Los compiladores (gcc, g++) quedan excluidos automáticamente.
COPY --from=builder /install /usr/local

# Copiar código fuente de la aplicación.
COPY schemas.py analytics.py agent.py main.py ./

# Cambiar a usuario no-root antes de exponer el puerto.
USER appuser

EXPOSE 8000

# Healthcheck nativo de Docker: llama al endpoint /health cada 30s.
# Si falla 3 veces consecutivas, el contenedor se marca como unhealthy.
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Modo producción: 4 workers Uvicorn, sin hot-reload.
# --workers 4: paralelismo para manejar múltiples requests concurrentes.
# --loop uvloop: event loop más rápido que asyncio estándar (C extension).
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "4", "--loop", "uvloop", "--log-level", "info"]
