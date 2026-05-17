"""
bot_telegram.py — Diario Biomorfológico Continuo con Almacenamiento Local.

Flujo de datos:
  1. /start explica el flujo: Subir un CSV inicial de >= 14 días.
  2. Al subir un CSV de cualquier tamaño, el bot fusiona los datos con su historial local (sin duplicar fechas).
  3. El comando /hoy permite picar el día actual de forma rápida sin subir archivos.
  4. Cada vez que se actualizan datos, el bot extrae automáticamente los ÚLTIMOS 14 días cronológicos 
     y los envía al backend FastAPI para ejecutar la regresión Ridge y la IA semántica.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
from datetime import date
from typing import List, Dict, Any

import httpx
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

from schemas import DailyLog, TrainingType, WeeklyPredictionInput

# Configuración e Infraestructura de Almacenamiento
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
API_BASE_URL: str = os.getenv("API_BASE_URL", "http://api:8000")
PREDICT_ENDPOINT: str = f"{API_BASE_URL}/api/v1/predict-performance"

# Carpeta local para almacenar los JSON de cada usuario de forma persistente
DATA_DIR = "./user_data"
os.makedirs(DATA_DIR, exist_ok=True)

REQUIRED_COLUMNS = [
    "fecha", "horas_sueno", "calidad_sueno", "proteinas_g", 
    "carbohidratos_g", "grasas_g", "tipo_entrenamiento", "volumen_total_kg", "rpe"
]

# Funciones Utilitarias — Escapes Estrictos para Telegram MarkdownV2
def _escape_md(text: str) -> str:
    """
    Escapa caracteres especiales exigidos por el parseador MarkdownV2 de Telegram.
    Garantiza que strings con guiones, puntos o paréntesis del LLM no rompan el bot.
    """
    special_chars = r"_*[]()~`>#+-=|{}.!"
    return "".join(f"\\{c}" if c in special_chars else c for c in text)

# Funciones de Persistencia de Datos Local (File-Based DB)
def _get_user_filepath(user_id: str) -> str:
    return os.path.join(DATA_DIR, f"user_{user_id}.json")

def _load_user_logs(user_id: str) -> Dict[str, Any]:
    """Carga el historial del usuario desde su archivo JSON local."""
    filepath = _get_user_filepath(user_id)
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def _save_user_logs(user_id: str, data: Dict[str, Any]) -> None:
    """Guarda el historial del usuario en su archivo JSON local."""
    filepath = _get_user_filepath(user_id)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def _merge_and_pipeline(user_id: str, new_logs: List[DailyLog]) -> tuple[int, List[DailyLog]]:
    """
    Fusiona los nuevos logs con los existentes indexando por fecha para evitar duplicados.
    Retorna el total de días históricos acumulados y los últimos 14 días ordenados para la API.
    """
    user_data = _load_user_logs(user_id)
    
    # Insertar o actualizar logs usando la fecha como clave única string
    for log in new_logs:
        user_data[str(log.fecha)] = log.model_dump(mode="json")
    
    _save_user_logs(user_id, user_data)
    
    # Reconstruir lista completa ordenada por fecha cronológica
    all_sorted_dates = sorted(user_data.keys())
    full_history = [DailyLog(**user_data[d]) for d in all_sorted_dates]
    
    # Extraer estrictamente los últimos 14 días para alimentar la ventana móvil de la regresión
    window_14_days = full_history[-14:]
    
    return len(full_history), window_14_days

# Pipeline de comunicación con FastAPI
async def _trigger_prediction_pipeline(update: Update, user_id: str, window_logs: List[DailyLog], status_msg) -> None:
    """Envía la ventana móvil de datos a la API y renderiza el reporte final."""
    await status_msg.edit_text(" Ventana de datos lista. Ejecutando Ridge Regression y análisis de IA...")
    
    try:
        prediction_input = WeeklyPredictionInput(user_id=user_id, logs=window_logs)
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                PREDICT_ENDPOINT,
                json=prediction_input.model_dump(mode="json"),
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()

        latency_ms: str = response.headers.get("X-Process-Time-Ms", "N/A")
        report_data: dict = response.json()

        message = _format_report_markdown(report_data, latency_ms, len(_load_user_logs(user_id)))
        await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN_V2)
        await status_msg.delete()

    except Exception as exc:
        logger.exception("Error en pipeline: %s", exc)
        await status_msg.edit_text("❌ Error interno del servidor al procesar el reporte de rendimiento.")

# Handlers de Entrada de Datos
async def csv_document_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Parsea subidas masivas de CSV de cualquier longitud (Mínimo histórico acumulado de 14 días)."""
    user = update.effective_user
    user_id = str(user.id) if user else "unknown"
    document = update.message.document

    if not document or not document.file_name.endswith('.csv'):
        await update.message.reply_text("❌ Envía un archivo con extensión `.csv` nativa.")
        return

    status_message = await update.message.reply_text("📥 Descargando e indexando registros del CSV...")

    try:
        telegram_file = await context.bot.get_file(document.file_id)
        file_bytes = await telegram_file.download_as_bytearray()
        
        csv_text = io.StringIO(file_bytes.decode('utf-8'))
        reader = csv.DictReader(csv_text)

        if not reader.fieldnames or not all(col in reader.fieldnames for col in REQUIRED_COLUMNS):
            await status_message.edit_text("❌ Estructura de cabeceras de CSV inválida. Usa el formato de /start.")
            return

        new_logs: List[DailyLog] = []
        for row in reader:
            new_logs.append(DailyLog(
                fecha=date.fromisoformat(row["fecha"].strip()),
                horas_sueno=float(row["horas_sueno"]),
                calidad_sueno=int(row["calidad_sueno"]),
                proteinas_g=float(row["proteinas_g"]),
                carbohidratos_g=float(row["carbohidratos_g"]),
                grasas_g=float(row["grasas_g"]),
                tipo_entrenamiento=TrainingType(row["tipo_entrenamiento"].strip()),
                volumen_total_kg=float(row["volumen_total_kg"]),
                rpe=int(row["rpe"])
            ))

        # Fusionar datos en el almacén local del usuario
        total_accumulated, window_14_days = _merge_and_pipeline(user_id, new_logs)

        if total_accumulated < 14:
            await status_message.edit_text(
                f" *Historial guardado con éxito* (`{total_accumulated}/14 días`)\n\n"
                f"Tus datos se han guardado correctamente, pero el motor matemático aún necesita "
                f"un mínimo de *14 días acumulados* para trazar una predicción legítima\\.\n\n"
                f"Añade los `{14 - total_accumulated}` días restantes subiendo otro CSV o usando el comando `/hoy`\\.",
                parse_mode=ParseMode.MARKDOWN_V2
            )
            return

        # Si ya hay 14 días o más en total, dispara el análisis con los 14 más recientes
        await _trigger_prediction_pipeline(update, user_id, window_14_days, status_message)

    except Exception as exc:
        logger.exception("Fallo en procesamiento CSV: %s", exc)
        await status_message.edit_text(" Error de lectura. Asegúrate de que el CSV tenga formato UTF-8 válido.")


async def hoy_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Permite añadir de forma interactiva el registro del día actual.
    Sintaxis: /hoy horas_sueño calidad_sueño proteinas carbohidratos grasas entrenamiento volumen rpe
    """
    user = update.effective_user
    user_id = str(user.id) if user else "unknown"
    args = context.args

    if len(args) != 8:
        await update.message.reply_text(
            " *Uso correcto del comando /hoy:*\n"
            "`/hoy [sueño_hrs] [calidad_1-10] [proteina_g] [carbo_g] [grasa_g] [Fuerza/Boxeo/Descanso] [volumen_kg] [rpe_1-10]`\n\n"
            "*Ejemplo práctico:* `/hoy 7.5 8 160 300 70 Fuerza 4500 7`",
            parse_mode=ParseMode.MARKDOWN_V2
        )
        return

    status_message = await update.message.reply_text("📝 Procesando entrada diaria...")

    try:
        log_hoy = DailyLog(
            fecha=date.today(),
            horas_sueno=float(args[0]),
            calidad_sueno=int(args[1]),
            proteinas_g=float(args[2]),
            carbohidratos_g=float(args[3]),
            grasas_g=float(args[4]),
            tipo_entrenamiento=TrainingType(args[5].strip()),
            volumen_total_kg=float(args[6]),
            rpe=int(args[7])
        )

        total_accumulated, window_14_days = _merge_and_pipeline(user_id, [log_hoy])

        if total_accumulated < 14:
            await status_message.edit_text(
                f" *Día guardado con éxito\\.*\n\nHistorial acumulado actual: `{total_accumulated}/14 días`\\.\n"
                f"El motor matemático requiere un mínimo de 14 observaciones para activarse\\. Sigue registrando días\\.",
                parse_mode=ParseMode.MARKDOWN_V2
            )
            return

        await _trigger_prediction_pipeline(update, user_id, window_14_days, status_message)

    except Exception as exc:
        await status_message.edit_text(
            f"❌ *Error de validación de datos*\\.\n"
            f"Verifica que los números sean correctos y el tipo sea `Fuerza`, `Boxeo` o `Descanso`\\.\n"
            f"Detalle: `{_escape_md(str(exc))}`",
            parse_mode=ParseMode.MARKDOWN_V2
        )

# Formateador e Instrucciones
def _format_report_markdown(report: dict, latency_ms: str, total_database_days: int) -> str:
    """Convierte el BiometricReportResponse en un mensaje enriquecido de Telegram escapando de forma agresiva."""
    rendimiento: int = report["rendimiento_estimated_porcentaje"] if "rendimiento_estimated_porcentaje" in report else report.get("rendimiento_estimado_porcentaje", 0)
    
    riesgo: str = _escape_md(report.get("factor_riesgo_critico", "N/A"))
    recomendacion: str = _escape_md(report.get("recomendacion_ajuste_split", "N/A"))
    all_razones: List[str] = report.get("razones_tecnicas", [])

    filled = round(rendimiento / 10)
    bar = "🟩" * filled + "⬛" * (10 - filled)
    status_emoji = "🔴" if rendimiento < 40 else "🟡" if rendimiento < 70 else "🟢"
    
    razones_formatted = "\n".join(f"  • {_escape_md(r)}" for r in all_razones)

    header = _escape_md("🧬 DEEP-BIOHACKING TRACKER — Reporte Continuo Real")
    separator = _escape_md("──────────────────────────────────")
    db_info = _escape_md(f" Base de datos del atleta: {total_database_days} días registrados")
    window_info = _escape_md(" Ventana del análisis analítico: Últimos 14 días cronológicos")
    
    label_perf = _escape_md("Rendimiento estimado:")
    label_risk = _escape_md(" Factor de riesgo crítico:")
    label_recom = _escape_md(" Ajuste de split recomendado:")
    label_tech = _escape_md(" Razones técnicas (coeficientes Ridge):")
    
    footer = _escape_md(f" Latencia de infraestructura: {latency_ms} ms")
    engine_info = _escape_md(" Motor: Ridge Regression local + GPT-4o-mini")

    return (
        f"*{header}*\n"
        f"{separator}\n\n"
        f"{db_info}\n"
        f"{window_info}\n\n"
        f"{status_emoji} *{label_perf}* `{rendimiento}%`\n"
        f"{bar}\n\n"
        f"*{label_risk}*\n"
        f"_{riesgo}_\n\n"
        f"*{label_recom}*\n"
        f"_{recomendacion}_\n\n"
        f"*{label_tech}*\n"
        f"{razones_formatted}\n\n"
        f"{separator}\n"
        f"_{footer}_\n"
        f"_{engine_info}_"
    )

async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    instrucciones = (
        f" *Diario Predictivo de Biohacking Continúo*\n\n"
        f"Este bot funciona mediante una ventana móvil de datos reales para predecir tu fatiga muscular\\.\n\n"
        f" *PASO 1 (Inicialización):* Para activar el motor predictivo por primera vez, necesitas subir un archivo *\\.csv* con un historial base de *AL MENOS 14 días*\\.\n\n"
        f" *PASO 2 (Evolución diaria):* Una vez cargada la base, puedes ir enviando ficheros CSV con más días (15, 20, 40 días) o registrar de forma rápida tu día actual usando el comando interactivo `/hoy`\\.\n\n"
        f" *Estructura obligatoria de columnas para el CSV:*\n"
        f"`fecha,horas_sueno,calidad_sueno,proteinas_g,carbohidratos_g,grasas_g,"
        f"tipo_entrenamiento,volumen_total_kg,rpe`\n\n"
        f" *Sintaxis del registro manual diario:* \n"
        f"`/hoy [sueño_hrs] [calidad_1-10] [proteina_g] [carbo_g] [grasa_g] [Fuerza/Boxeo/Descanso] [volumen_kg] [rpe_1-10]`\n\n"
        f" *¿Listo?* Adjunta tu CSV inicial o añade datos para recalcular tu predicción instantáneamente con tu serie temporal real\\."
    )
    await update.message.reply_text(instrucciones, parse_mode=ParseMode.MARKDOWN_V2)

def main() -> None:
    logger.info("Iniciando Deep-Biohacking Tracker Bot con persistencia local...")
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start_handler))
    application.add_handler(CommandHandler("hoy", hoy_handler))
    application.add_handler(MessageHandler(filters.Document.ALL, csv_document_handler))

    logger.info("Bot en modo Diario continuo activo. Escuchando.")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()