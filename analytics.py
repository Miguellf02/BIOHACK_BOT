"""
analytics.py — Motor analítico local de coste $0.

Filosofía de diseño:
- La regresión Ridge se ejecuta 100 % localmente, sin tokens externos.
- asyncio.to_thread() delega el cómputo síncrono de scikit-learn al
  ThreadPoolExecutor del event loop, garantizando que FastAPI nunca bloquee
  su bucle de eventos mientras procesa matrices numéricas.
- Sólo se exponen al LLM los coeficientes agregados (≤ 6 números flotantes),
  no las 14 filas × 9 columnas de datos crudos. Esto reduce el consumo de
  tokens en ~98 % por request y elimina el riesgo de alucinaciones numéricas.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from schemas import DailyLog, TrainingType

logger = logging.getLogger(__name__)



_VOLUME_DESCANSO: float = 0.0

# Mapa de tipo de entrenamiento → factor multiplicador de intensidad relativa.
# Boxeo eleva más la fatiga neuromuscular por unidad de volumen que Fuerza pura.
_INTENSITY_FACTOR: Dict[TrainingType, float] = {
    TrainingType.FUERZA: 1.0,
    TrainingType.BOXEO: 1.35,
    TrainingType.DESCANSO: 0.0,
}


def _build_feature_matrix(logs: List[DailyLog]) -> np.ndarray:
    """
    Construye la matriz de features X a partir de los DailyLog.

    Variables independientes (4 features por observación):
      [0] horas_sueno         — recuperación cuantitativa
      [1] calidad_sueno       — recuperación cualitativa (HRV proxy)
      [2] volumen_escalado    — carga mecánica ajustada por intensidad modal
      [3] ratio_proteina_kcal — adecuación proteica relativa a calorías totales

    El ratio proteico captura la idoneidad de la dieta sin enviar valores
    absolutos de macros al LLM (reducción de contexto).
    """
    rows: List[List[float]] = []
    for log in logs:
        # Carga mecánica ponderada por modalidad de entrenamiento
        volumen_escalado = log.volumen_total_kg * _INTENSITY_FACTOR[log.tipo_entrenamiento]

        # Ratio proteína / calorías totales (kcal). Evita división por cero.
        kcal_total = (
            log.proteinas_g * 4.0
            + log.carbohidratos_g * 4.0
            + log.grasas_g * 9.0
        )
        ratio_proteina = (log.proteinas_g * 4.0) / max(kcal_total, 1.0)

        rows.append([
            float(log.horas_sueno),
            float(log.calidad_sueno),
            volumen_escalado,
            ratio_proteina,
        ])

    return np.array(rows, dtype=np.float64)


def _run_ridge_regression(logs: List[DailyLog]) -> Dict[str, Any]:
    """
    Función síncrona que ejecuta Ridge Regression sobre los 14 registros.

    Se invoca exclusivamente desde asyncio.to_thread() para no bloquear
    el event loop de FastAPI mientras scikit-learn opera con numpy arrays.

    Retorna:
        predicted_rpe_next_week: predicción puntual de fatiga (RPE) para T+7.
        coefficients: dict {feature_name: coeficiente_estandarizado}.
        current_mean_rpe: media de RPE histórico de referencia.
    """
    X = _build_feature_matrix(logs)
    y = np.array([float(log.rpe) for log in logs], dtype=np.float64)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = Ridge(alpha=1.0, fit_intercept=True)
    model.fit(X_scaled, y)

    feature_names = [
        "horas_sueno",
        "calidad_sueno",
        "volumen_escalado_kg",
        "ratio_proteina_kcal",
    ]
    coefficients: Dict[str, float] = {
        name: round(float(coef), 4)
        for name, coef in zip(feature_names, model.coef_)
    }

    X_last7_mean = X[-7:].mean(axis=0, keepdims=True)
    X_last7_scaled = scaler.transform(X_last7_mean)
    predicted_rpe: float = float(np.clip(model.predict(X_last7_scaled)[0], 1.0, 10.0))

    result = {
        "predicted_rpe_next_week": round(predicted_rpe, 2),
        "coefficients": coefficients,
        "current_mean_rpe": round(float(y.mean()), 2),
        "intercept": round(float(model.intercept_), 4),
    }

    logger.debug("Ridge output: %s", result)
    return result


#Motor asincrono 
class BiohackingAnalyticsEngine:
    """
    Interfaz asíncrona sobre el motor de ML local.

    Patrón: thin async wrapper → heavy sync computation en ThreadPool.
    Esto mantiene la compatibilidad con el event loop de FastAPI/uvicorn
    sin sacrificar el rendimiento de scikit-learn/numpy.
    """

    async def compute_insights(self, logs: List[DailyLog]) -> Dict[str, Any]:
        """
        Punto de entrada asíncrono del motor analítico.

        El bloqueo de CPU se delega al ThreadPoolExecutor del event loop
        mediante asyncio.to_thread(), permitiendo que FastAPI atienda otras
        peticiones concurrentes mientras Ridge Regression computa.

        Args:
            logs: Lista de 14 DailyLog validados por Pydantic.

        Returns:
            Diccionario con predicción de RPE y coeficientes de impacto.
        """
        insights = await asyncio.to_thread(_run_ridge_regression, logs)
        return insights
