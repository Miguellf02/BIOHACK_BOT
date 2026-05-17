"""
bot_telegram.py — Interfaz de usuario: Bot de Telegram asíncrono.

Flujo del comando /predecir:
  1. Genera 14 DailyLog simulados (demo realista con varianza controlada).
  2. Realiza POST asíncrono a la API FastAPI con httpx.AsyncClient.
  3. Extrae la cabecera X-Process-Time-Ms para mostrar latencia real.
  4. Formatea el reporte BiometricReportResponse en Markdown enriquecido.
  5. Envía la respuesta al usuario en Telegram.

Nota de arquitectura: python-telegram-bot v20+ es nativo async, por lo que
se integra perfectamente con httpx.AsyncClient sin necesidad de run_until_complete.
"""

from __future__ import annotations

import logging
import os
import random
from datetime import date, timedelta
from typing import List

import httpx
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes

from schemas import DailyLog, TrainingType, WeeklyPredictionInput

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
API_BASE_URL: str = os.getenv("API_BASE_URL", "http://api:8000")
PREDICT_ENDPOINT: str = f"{API_BASE_URL}/api/v1/predict-performance"

if not TELEGRAM_BOT_TOKEN:
    raise EnvironmentError(
        "Variable de entorno TELEGRAM_BOT_TOKEN no configurada. "
        "Añádela al fichero .env o al docker-compose.yml."
    )


# ---------------------------------------------------------------------------
# Generador de logs simulados para demo
# ---------------------------------------------------------------------------

def _generate_demo_logs(user_id: str) -> WeeklyPredictionInput:
    """
    Genera 14 días de DailyLog con varianza realista para demostración.

    El patrón simula un ciclo de carga típico: alta intensidad días 1-5,
    fatiga acumulada visible días 6-10, recuperación parcial días 11-14.
    Esto garantiza que Ridge Regression detecte coeficientes significativos.
    """
    training_rotation: List[TrainingType] = [
        TrainingType.FUERZA,
        TrainingType.BOXEO,
        TrainingType.FUERZA,
        TrainingType.BOXEO,
        TrainingType.FUERZA,
        TrainingType.DESCANSO,
        TrainingType.DESCANSO,
    ]

    logs: List[DailyLog] = []
    today = date.today()

    for i in range(14):
        day_offset = 13 - i  # Día 0 = hace 13 días; día 13 = ayer
        log_date = today - timedelta(days=day_offset)

        # Fatiga creciente en la primera semana, recuperación en la segunda
        fatigue_factor = 1.0 + (i / 14.0) * 0.3 if i < 7 else 1.0 - ((i - 7) / 14.0) * 0.2

        tipo = training_rotation[i % 7]
        volumen = 0.0 if tipo == TrainingType.DESCANSO else round(
            random.uniform(3000, 6000) * fatigue_factor, 1
        )
        rpe = 5 if tipo == TrainingType.DESCANSO else int(
            min(10, max(1, round(6 + fatigue_factor * 1.5 + random.uniform(-1, 1))))
        )

        logs.append(
            DailyLog(
                fecha=log_date,
                horas_sueno=round(random.uniform(5.5, 8.5) / fatigue_factor, 1),
                calidad_sueno=max(1, min(10, int(7 - fatigue_factor + random.uniform(-1, 1)))),
                proteinas_g=round(random.uniform(140, 220), 1),
                carbohidratos_g=round(random.uniform(200, 400), 1),
                grasas_g=round(random.uniform(50, 90), 1),
                tipo_entrenamiento=tipo,
                volumen_total_kg=volumen,
                rpe=rpe,
            )
        )

    return WeeklyPredictionInput(user_id=user_id, logs=logs)


# ---------------------------------------------------------------------------
# Formateador de reporte Markdown
# ---------------------------------------------------------------------------

def _format_report_markdown(report: dict, latency_ms: str) -> str:
    """
    Convierte el BiometricReportResponse en un mensaje Telegram con Markdown.

    Usa MarkdownV2 de Telegram con escapes para caracteres especiales.
    El rendimiento se visualiza con una barra de progreso emoji.
    """
    rendimiento: int = report["rendimiento_estimado_porcentaje"]
    riesgo: str = report["factor_riesgo_critico"]
    recomendacion: str = report["recomendacion_ajuste_split"]
    razones: List[str] = report["razones_tecnicas"]

    # Barra de progreso visual (10 bloques)
    filled = round(rendimiento / 10)
    bar = "🟩" * filled + "⬛" * (10 - filled)

    # Emoji de alerta según nivel de rendimiento
    status_emoji = "🔴" if rendimiento < 40 else "🟡" if rendimiento < 70 else "🟢"

    razones_formatted = "\n".join(f"  • {r}" for r in razones)

    return (
        f"🧬 *DEEP\\-BIOHACKING TRACKER* — Reporte Semanal\n"
        f"{'─' * 34}\n\n"
        f"{status_emoji} *Rendimiento estimado:* `{rendimiento}%`\n"
        f"{bar}\n\n"
        f"⚠️ *Factor de riesgo crítico:*\n"
        f"_{_escape_md(riesgo)}_\n\n"
        f"🎯 *Ajuste de split recomendado:*\n"
        f"_{_escape_md(recomendacion)}_\n\n"
        f"📊 *Razones técnicas \\(coeficientes Ridge\\):*\n"
        f"{_escape_md(razones_formatted)}\n\n"
        f"{'─' * 34}\n"
        f"⚡ _Latencia del sistema: `{_escape_md(latency_ms)} ms`_\n"
        f"🤖 _Motor: Ridge Regression \\+ GPT\\-4o\\-mini_"
    )


def _escape_md(text: str) -> str:
    """
    Escapa caracteres especiales de MarkdownV2 de Telegram.
    Caracteres que requieren escape: _ * [ ] ( ) ~ ` > # + - = | { } . !
    """
    special_chars = r"_*[]()~`>#+-=|{}.!"
    return "".join(f"\\{c}" if c in special_chars else c for c in text)


# ---------------------------------------------------------------------------
# Handler del comando /predecir
# ---------------------------------------------------------------------------

async def predecir_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handler asíncrono del comando /predecir.

    Proceso:
      1. Genera logs de demo (sin bloqueo, operación CPU-ligera).
      2. POST asíncrono a FastAPI con httpx.AsyncClient (no bloquea el bot).
      3. Extrae latencia de la cabecera X-Process-Time-Ms.
      4. Formatea y envía el reporte.
    """
    user = update.effective_user
    user_id = str(user.id) if user else "unknown"

    await update.message.reply_text(
        "⏳ Analizando tus biométricos de los últimos 14 días...\n"
        "El motor Ridge Regression está calculando tus coeficientes de fatiga.",
        parse_mode=None,
    )

    logger.info("Comando /predecir recibido | user_id=%s", user_id)

    try:
        # Generar payload de demo
        prediction_input = _generate_demo_logs(user_id)

        # httpx.AsyncClient: cliente HTTP async de alto rendimiento.
        # timeout=60s para contemplar la latencia del LLM en carga alta.
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                PREDICT_ENDPOINT,
                json=prediction_input.model_dump(mode="json"),
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()

        # Extrae latencia desde cabecera custom del middleware de FastAPI
        latency_ms: str = response.headers.get("X-Process-Time-Ms", "N/A")
        report_data: dict = response.json()

        message = _format_report_markdown(report_data, latency_ms)

        await update.message.reply_text(
            message,
            parse_mode=ParseMode.MARKDOWN_V2,
        )
        logger.info("Reporte enviado | user_id=%s | latency=%s ms", user_id, latency_ms)

    except httpx.HTTPStatusError as exc:
        logger.error("Error HTTP de la API: %s", exc)
        await update.message.reply_text(
            f"❌ Error al conectar con el servidor de análisis (HTTP {exc.response.status_code}).\n"
            "Verifica que el servicio FastAPI está corriendo."
        )
    except httpx.RequestError as exc:
        logger.error("Error de conexión: %s", exc)
        await update.message.reply_text(
            "❌ No se pudo conectar con el servidor de análisis.\n"
            f"URL configurada: `{PREDICT_ENDPOINT}`"
        )
    except Exception as exc:
        logger.exception("Error inesperado en /predecir: %s", exc)
        await update.message.reply_text(
            "❌ Error inesperado. Consulta los logs del sistema."
        )


async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Mensaje de bienvenida con instrucciones de uso."""
    await update.message.reply_text(
        "🧬 *Deep\\-Biohacking Tracker*\n\n"
        "Soy tu motor predictivo de rendimiento atlético\\.\n\n"
        "📊 Usa */predecir* para generar tu reporte biométrico semanal\\.\n"
        "El sistema analiza 14 días de datos y predice tu rendimiento "
        "usando Ridge Regression \\+ IA semántica\\.",
        parse_mode=ParseMode.MARKDOWN_V2,
    )


# ---------------------------------------------------------------------------
# Entrypoint del bot
# ---------------------------------------------------------------------------

def main() -> None:
    """Inicializa y arranca el bot de Telegram en modo polling."""
    logger.info("Iniciando Deep-Biohacking Tracker Bot...")

    application = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .build()
    )

    application.add_handler(CommandHandler("start", start_handler))
    application.add_handler(CommandHandler("predecir", predecir_handler))

    logger.info("Bot escuchando comandos. Ctrl+C para detener.")
    # run_polling() gestiona el event loop internamente con python-telegram-bot v20+
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
