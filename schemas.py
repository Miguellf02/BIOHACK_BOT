# schemas.py — Contratos de datos con Pydantic v2 (Flexibilizado para N >= 14 días)
from __future__ import annotations  # <-- Regla de oro: Siempre en la línea 1

from datetime import date
from enum import Enum
from typing import List

from pydantic import BaseModel, Field, field_validator

# Enumeraciones

class TrainingType(str, Enum):
    """Tipos de sesión de entrenamiento registrados diariamente."""
    FUERZA = "Fuerza"
    BOXEO = "Boxeo"
    DESCANSO = "Descanso"

# Modelos de entrada

class DailyLog(BaseModel):
    """Registro biométrico y de rendimiento de un día concreto."""
    fecha: date = Field(..., description="Fecha ISO 8601 del registro.")
    horas_sueno: float = Field(..., ge=0.0, le=24.0, description="Horas de sueño (0-24).")
    calidad_sueno: int = Field(..., ge=1, le=10, description="Calidad subjetiva del sueño (1-10).")
    proteinas_g: float = Field(..., ge=0.0, description="Proteínas consumidas en gramos.")
    carbohidratos_g: float = Field(..., ge=0.0, description="Carbohidratos consumidos en gramos.")
    grasas_g: float = Field(..., ge=0.0, description="Grasas consumidas en gramos.")
    tipo_entrenamiento: TrainingType = Field(..., description="Modalidad de entrenamiento del día.")
    volumen_total_kg: float = Field(..., ge=0.0, description="Volumen total levantado en kg.")
    rpe: int = Field(..., ge=1, le=10, description="Rate of Perceived Exertion — proxy de fatiga.")

    @field_validator("fecha", mode="before")
    @classmethod
    def parse_fecha(cls, v: object) -> date:
        if isinstance(v, str):
            return date.fromisoformat(v)
        return v  # type: ignore[return-value]


class WeeklyPredictionInput(BaseModel):
    """
    Contrato flexibilizado. Ahora acepta cualquier historial de entrenamiento
    siempre y cuando se cumpla el umbral mínimo de 14 observaciones (N >= 14)
    necesario para la estabilidad matemática del modelo Ridge.
    """
    user_id: str = Field(..., min_length=1, description="Identificador único del usuario.")
    logs: List[DailyLog] = Field(
        ...,
        min_length=14,  # Umbral mínimo de observaciones para el modelo predictivo
        description="Histórico continuo de registros biométricos (mínimo 14 días).",
    )

# Modelo de respuesta (Structured Output del agente LLM)

class BiometricReportResponse(BaseModel):
    """Reporte semántico estructurado generado por el agente LLM."""
    rendimiento_estimado_porcentaje: int = Field(
        ...,
        ge=0,
        le=100,
        description="Rendimiento global proyectado para la próxima semana (0-100 %).",
    )
    factor_riesgo_critico: str = Field(
        ...,
        description="El factor biométrico con mayor impacto negativo detectado por Ridge.",
    )
    recomendacion_ajuste_split: str = Field(
        ...,
        description="Ajuste concreto de volumen/intensidad/descanso recomendado para el split semanal.",
    )
    razones_tecnicas: List[str] = Field(
        ...,
        min_length=2,
        max_length=5,
        description="Lista de 2-5 razones técnicas basadas en los coeficientes de impacto.",
    )