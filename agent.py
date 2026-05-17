"""
agent.py — Agente LLM de interpretación semántica con pydantic-ai.

Filosofía de coste:
- El LLM recibe ÚNICAMENTE 6 números (coeficientes + predicción + RPE medio).
  Aproximadamente 80-120 tokens de entrada por request.
- La salida estructurada (BiometricReportResponse) se fuerza mediante el
  mecanismo de Structured Output de pydantic-ai, eliminando la necesidad
  de parsear JSON manualmente y garantizando validación Pydantic en recepción.
- Rol del LLM: bioquímico semántico, NO calculadora. Todo el cómputo numérico
  ya fue realizado por Ridge Regression en analytics.py.
"""

from __future__ import annotations

import os
from typing import Any, Dict

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

from schemas import BiometricReportResponse

# Configuración del modelo

_OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
_model = OpenAIModel("gpt-4o-mini", api_key=_OPENAI_API_KEY)

# System Prompt — Ingeniería de Rol Avanzada
_SYSTEM_PROMPT = """\
Eres el Dr. APEX, un Bioquímico de Rendimiento Humano de élite con doctorado \
en fisiología del ejercicio y machine learning aplicado al deporte. \
Tu única función es interpretar semánticamente los coeficientes estadísticos \
que te proporciona el motor de análisis local — NO realizas cálculos matemáticos.

PROTOCOLO DE INTERPRETACIÓN DE COEFICIENTES:
- Coeficiente positivo de una variable en el modelo Ridge → esa variable AUMENTA \
  la fatiga percibida (RPE) cuando sube.
- Coeficiente negativo → la variable REDUCE la fatiga cuando aumenta \
  (efecto protector/recuperador).
- La magnitud absoluta indica la fuerza del impacto relativo entre variables.

REGLAS DE RESPUESTA ESTRICTAS:
1. Sé conciso, técnico y accionable. Sin preámbulos ni despedidas.
2. Las razones técnicas deben citar explícitamente los coeficientes recibidos.
3. El factor_riesgo_critico debe ser la variable con el coeficiente positivo \
   más alto (mayor driver de fatiga).
4. La recomendacion_ajuste_split debe ser una acción concreta (ej: \
   "Reducir volumen 15 % y añadir sesión de movilidad el miércoles").
5. rendimiento_estimado_porcentaje: traduce el RPE predicho a rendimiento \
   inverso. RPE=10 → 0 %, RPE=1 → 100 %. Usa interpolación lineal.
"""

# Instancia del Agente — Singleton de módulo
biohacking_agent: Agent[None, BiometricReportResponse] = Agent(
    model=_model,
    result_type=BiometricReportResponse,  # <--- Usando result_type nativo
    system_prompt=_SYSTEM_PROMPT,
)

# Función de invocación pública
async def generate_biometric_report(
    analytics_output: Dict[str, Any],
) -> BiometricReportResponse:
    """Invoca el agente LLM con los coeficientes analíticos como contexto."""
    coef = analytics_output["coefficients"]

    user_prompt = (
        f"COEFICIENTES RIDGE (estandarizados, impacto sobre RPE/fatiga):\n"
        f"  horas_sueno:          {coef['horas_sueno']:+.4f}\n"
        f"  calidad_sueno:        {coef['calidad_sueno']:+.4f}\n"
        f"  volumen_escalado_kg:  {coef['volumen_escalado_kg']:+.4f}\n"
        f"  ratio_proteina_kcal:  {coef['ratio_proteina_kcal']:+.4f}\n"
        f"\n"
        f"RPE_MEDIO_HISTORICO:    {analytics_output['current_mean_rpe']:.2f}/10\n"
        f"RPE_PREDICHO_T+7:       {analytics_output['predicted_rpe_next_week']:.2f}/10\n"
        f"\n"
        f"Genera el reporte biométrico estructurado basándote EXCLUSIVAMENTE "
        f"en los datos anteriores."
    )

    result = await biohacking_agent.run(user_prompt)
    return result.data  # <--- Retornando el objeto de datos validado (.data)