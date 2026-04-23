"""
ReActAgent: bucle de Razonamiento + Acción para consultas de predicción de ventas del SRI Ecuador.

Patrón de ejecución
───────────────────
Cada invocación de run() ejecuta hasta MAX_ITERATIONS rondas de:
  1. [THINK]   — llama al LLM con el historial de mensajes y las herramientas disponibles.
  2. [ACT]     — si el LLM solicitó llamadas a herramientas, las ejecuta y añade
                 las observaciones al historial.
  3. [OBSERVE] — registra los resultados de cada herramienta.
  4. Repite hasta que el LLM devuelve un mensaje final sin llamadas a herramientas.

Secuencia típica de herramientas:
  iter 1 → get_province_data(provincia, mes)     [obtiene features del CSV]
  iter 2 → call_inference(provincia, mes, ...)   [llama a ml-inference]
  iter 3 → respuesta final en lenguaje natural (sin llamadas a herramientas)

Gateway LLM
───────────
LiteLLM se usa como gateway agnóstico al proveedor.  Groq es el proveedor
predeterminado (modelo: groq/llama3-8b-8192).  La variable de entorno GROQ_API_KEY
es obligatoria y es leída automáticamente por LiteLLM.

Para cambiar el modelo, usar la variable LITELLM_MODEL, p.ej.:
  LITELLM_MODEL=groq/mixtral-8x7b-32768
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import litellm

from metrics import agent_llm_calls_total, agent_tool_calls_total
from tools import TOOLS, call_inference, get_province_data

logger = logging.getLogger(__name__)

MODEL: str = os.environ.get("LITELLM_MODEL", "groq/llama3-8b-8192")
MAX_ITERATIONS: int = int(os.environ.get("AGENT_MAX_ITERATIONS", "8"))

# ─────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
Eres un asistente especializado en análisis de ventas y exportaciones del Ecuador, \
con acceso a datos oficiales del SRI (Servicio de Rentas Internas).

CONTEXTO DEL DATASET:
- El dataset cubre desde enero 2020 (registro 1) hasta septiembre 2025 (registro 69).
- No existe columna de año explícita; el año se deriva del número de registro:
    año = 2020 + (numero_registro - 1) // 12
    mes = ((numero_registro - 1) % 12) + 1
- El primer mes PREDECIBLE es octubre 2025 (primer mes fuera del dataset).
- El horizonte máximo de predicción es de 12 meses: octubre 2025 — septiembre 2026.

Tu misión es responder preguntas sobre predicciones de ventas de sociedades \
siguiendo OBLIGATORIAMENTE este proceso en orden:

PASO 1 — RAZONAMIENTO INICIAL
Identifica la provincia, el MES y el AÑO mencionados en la pregunta. \
Si el año no se menciona explícitamente, infiere el más lógico dentro del \
rango válido (oct 2025 — sep 2026). Si hay ambigüedad, elige el más probable.

PASO 2 — VALIDACIÓN TEMPORAL (OBLIGATORIA, antes de cualquier herramienta)
Verifica mentalmente que la fecha solicitada (mes + año) sea:
  a) Estrictamente posterior a septiembre 2025.
  b) No superior a septiembre 2026 (12 meses desde el corte).
Si la fecha NO cumple ambas condiciones, responde directamente en español \
explicando el rango válido SIN llamar ninguna herramienta.

PASO 3 — HERRAMIENTA get_province_data
Usa esta herramienta para obtener los datos históricos de exportaciones \
de la provincia y mes identificados. Son los features del modelo predictivo.

PASO 4 — HERRAMIENTA call_inference
Usa esta herramienta con: provincia, mes, AÑO (ano), y los datos del paso anterior. \
La herramienta también valida la fecha y retornará un error si está fuera del rango.

PASO 5 — RESPUESTA FINAL
Genera una respuesta en español, profesional y contextualizada que incluya:
• El valor predicho de ventas y exportaciones totales de sociedades
• El mes y año de la predicción
• Una breve interpretación del resultado (¿es alto/bajo para esa provincia?)
• Contexto sobre la estacionalidad del mes si es relevante

REGLAS OBLIGATORIAS:
- Provincias siempre en MAYÚSCULAS al llamar herramientas
- Los meses son números del 1 al 12
- El año (ano) es obligatorio en call_inference
- Responde siempre en español
- No inventes datos; usa únicamente lo que retornan las herramientas
- Si call_inference retorna error_tipo="fecha_fuera_de_rango", \
  transmite el mensaje de error directamente al usuario sin reintentar

Provincias válidas: AZUAY, BOLIVAR, CARCHI, CANAR, CHIMBORAZO, COTOPAXI, \
EL ORO, ESMERALDAS, GALAPAGOS, GUAYAS, IMBABURA, LOJA, LOS RIOS, MANABI, \
MORONA SANTIAGO, NAPO, ND, ORELLANA, PASTAZA, PICHINCHA, SANTA ELENA, \
SANTO DOMINGO DE LOS TSACHILAS, SUCUMBIOS, TUNGURAHUA, ZAMORA CHINCHIPE\
"""


# ─────────────────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────────────────


class ReActAgent:
    """
    Stateless ReAct agent.  Each call to run() is independent.
    Thread-safe: no mutable instance state is modified after __init__.
    """

    def __init__(self) -> None:
        self.model = MODEL

    def run(self, pregunta: str) -> dict[str, Any]:
        """
        Execute the ReAct loop for a single user question.

        Returns a dict with:
          respuesta          — natural-language answer in Spanish
          datos_usados       — feature dict from get_province_data
          prediccion_raw     — raw float from call_inference (or None)
          razonamiento       — step-by-step trace of reasoning + tool calls
          latencia_ms        — total wall-clock time in milliseconds
          _inference_payload — internal: args passed to call_inference
                               (used by app.py for the Kafka event, not
                                returned to the API caller)
        """
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": pregunta},
        ]

        razonamiento: list[str] = [f"Pregunta recibida: «{pregunta}»"]
        datos_usados: dict[str, Any] = {}
        prediccion_raw: float | None = None
        inference_payload: dict[str, Any] = {}

        t_start = time.perf_counter()

        for iteration in range(1, MAX_ITERATIONS + 1):
            razonamiento.append(
                f"[THINK] Iteración {iteration}/{MAX_ITERATIONS}: consultando {self.model}…"
            )
            logger.info("[THINK] iter=%d/%d  model=%s", iteration, MAX_ITERATIONS, self.model)
            agent_llm_calls_total.inc()

            response = litellm.completion(
                model=self.model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.1,
            )

            msg = response.choices[0].message
            tool_calls = msg.tool_calls or []

            # Append the assistant turn to the message history
            assistant_turn: dict[str, Any] = {
                "role": "assistant",
                "content": msg.content or "",
            }
            if tool_calls:
                assistant_turn["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ]
            messages.append(assistant_turn)

            # No tool calls → LLM produced its final answer
            if not tool_calls:
                razonamiento.append(
                    f"[THINK] Respuesta final generada en iteración {iteration}."
                )
                logger.info("[THINK] Respuesta final — iter=%d", iteration)
                return {
                    "respuesta": msg.content or "",
                    "datos_usados": datos_usados,
                    "prediccion_raw": prediccion_raw,
                    "razonamiento": "\n".join(razonamiento),
                    "latencia_ms": round((time.perf_counter() - t_start) * 1000, 1),
                    "_inference_payload": inference_payload,
                }

            # Execute tool calls and collect observations
            for tc in tool_calls:
                tool_name = tc.function.name
                try:
                    args: dict = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                    logger.warning(
                        "[ACT] iter=%d  tool=%s  error=JSON inválido en argumentos: %s",
                        iteration, tool_name, tc.function.arguments,
                    )

                razonamiento.append(f"[ACT] {tool_name}({_fmt_args(args)})")
                logger.info("[ACT] iter=%d  tool=%s  args=%s", iteration, tool_name, _fmt_args(args))
                agent_tool_calls_total.labels(tool_name=tool_name).inc()

                observation = self._dispatch(tool_name, args)

                razonamiento.append(f"[OBSERVE] {tool_name}: {_fmt_result(observation)}")
                logger.info(
                    "[OBSERVE] iter=%d  tool=%s  result_keys=%s",
                    iteration, tool_name, list(observation.keys()),
                )

                # Capture structured results for the response envelope
                if tool_name == "get_province_data" and "error" not in observation:
                    datos_usados = observation
                elif tool_name == "call_inference":
                    inference_payload = args
                    if "prediccion_total_ventas" in observation:
                        prediccion_raw = float(observation["prediccion_total_ventas"])

                # Append tool result as an observation message
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tool_name,
                    "content": json.dumps(observation, ensure_ascii=False),
                })

        # MAX_ITERATIONS reached without a final answer
        logger.warning(
            "[THINK] Máximo de iteraciones alcanzado — MAX_ITERATIONS=%d  pregunta=%s",
            MAX_ITERATIONS, pregunta,
        )
        razonamiento.append(
            f"[THINK] ADVERTENCIA: máximo de iteraciones ({MAX_ITERATIONS}) alcanzado sin respuesta."
        )
        return {
            "respuesta": (
                "Lo siento, no pude completar el análisis en el número máximo de pasos. "
                "Por favor, intente reformular la pregunta de forma más específica."
            ),
            "datos_usados": datos_usados,
            "prediccion_raw": prediccion_raw,
            "razonamiento": "\n".join(razonamiento),
            "latencia_ms": round((time.perf_counter() - t_start) * 1000, 1),
            "_inference_payload": inference_payload,
        }

    # ── Tool dispatcher ───────────────────────────────────

    @staticmethod
    def _dispatch(name: str, args: dict) -> dict[str, Any]:
        if name == "get_province_data":
            return get_province_data(**args)
        if name == "call_inference":
            return call_inference(**args)
        return {"error": f"Unknown tool '{name}'"}


# ─────────────────────────────────────────────────────────
# Formatting helpers for the razonamiento log
# ─────────────────────────────────────────────────────────


def _fmt_args(args: dict) -> str:
    return ", ".join(f"{k}={v!r}" for k, v in args.items())


def _fmt_result(result: dict) -> str:
    if "error" in result:
        return f"ERROR — {result['error']}"
    preview = dict(list(result.items())[:4])
    suffix = " …" if len(result) > 4 else ""
    return str(preview) + suffix
