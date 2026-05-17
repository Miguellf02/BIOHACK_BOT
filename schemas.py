"""
schemas.py — Contratos de datos con Pydantic v2.

Filosofía: este archivo es la única fuente de verdad (Single Source of Truth)
para todos los tipos del sistema. Pydantic v2 valida en tiempo de ejecución y
genera JSON Schema automáticamente, lo que permite a FastAPI documentar la API
sin código adicional y fuerza al agente LLM a devolver salidas estructuradas.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import List

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enumeraciones
# ---------------------------------------------------------------------------

class TrainingType(str, Enum):
    """Tipos de sesión de entrenamiento registrados diariamente."""
    FUERZA = "Fuerza"
    BOXEO = "Boxeo"
    DESCANSO = "Descanso"


# ---------------------------------------------------------------------------
# Modelos de entrada
# ---------------------------------------------------------------------------

class DailyLog(BaseModel):
    """
    Registro biométrico y de rendimiento de un día concreto.

    Cada campo está acotado con Field() para que Pydantic rechace datos
    fuera de rango antes de que lleguen al motor analítico, evitando
    anomalías estadísticas que contaminarían los coeficientes de Ridge.
    """

    fecha: date = Field(..., description="Fecha ISO 8601 del registro.")
    horas_sueno: float = Field(..., ge=0.0, le=24.0, description="Horas de sueño (0-24).")
    calidad_sueno: int = Field(..., ge=1, le=10, description="Calidad subjetiva del sueño (1-10).")
    proteinas_g: float = Field(..., ge=0.0, description="Proteínas consumidas en gramos.")
    carbohidratos_g: float = Field(..., ge=0.0, description="Carbohidratos consumidos en gramos.")
    grasas_g: float = Field(..., ge=0.0, description="Grasas consumidas en gramos.")
    tipo_entrenamiento: TrainingType = Field(..., description="Modalidad de entrenamiento del día.")
    volumen_total_kg: float = Field(..., ge=0.0, description="Volumen total levantado en kg (series × reps × peso).")
    rpe: int = Field(..., ge=1, le=10, description="Rate of Perceived Exertion — proxy de fatiga acumulada.")

    @field_validator("fecha", mode="before")
    @classmethod
    def parse_fecha(cls, v: object) -> date:
        """Acepta strings ISO 8601 además de objetos date nativos."""
        if isinstance(v, str):
            return date.fromisoformat(v)
        return v  # type: ignore[return-value]


class WeeklyPredictionInput(BaseModel):
    """
    Payload que recibe el endpoint /api/v1/predict-performance.

    Requiere exactamente 14 registros diarios (dos semanas) para que
    Ridge Regression disponga de suficientes observaciones para estimar
    coeficientes robustos sin sobreajustarse.
    """

    user_id: str = Field(..., min_length=1, description="Identificador único del usuario.")
    logs: List[DailyLog] = Field(
        ...,
        min_length=14,
        max_length=14,
        description="Histórico de exactamente 14 días de registros biométricos.",
    )


# ---------------------------------------------------------------------------
# Modelo de respuesta (Structured Output del agente LLM)
# ---------------------------------------------------------------------------

class BiometricReportResponse(BaseModel):
    """
    Reporte semántico generado por el agente LLM.

    IMPORTANTE: Este esquema actúa como 'grammar' para la salida estructurada
    de pydantic-ai. El LLM NUNCA recibe datos crudos; sólo interpreta los
    coeficientes numéricos calculados localmente por Ridge Regression.
    Esto mantiene el coste por llamada en microcéntimos y elimina alucinaciones
    causadas por series temporales extensas en el contexto del modelo.
    """

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
