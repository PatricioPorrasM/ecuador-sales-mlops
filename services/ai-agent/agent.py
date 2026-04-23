"""
ReActAgent: Reasoning + Acting loop for Ecuador SRI sales prediction queries.

Pattern
───────
Each invocation of run() executes up to MAX_ITERATIONS rounds of:
  1. REASON  — call the LLM with current message history + available tools.
  2. ACT     — if the LLM requested tool calls, execute them and append
               the observations to the message history.
  3. Repeat until the LLM returns a final message with no tool calls.

Expected tool call sequence for a typical question:
  iter 1 → get_province_data(provincia, mes)     [fetches CSV features]
  iter 2 → call_inference(provincia, mes, ...)   [calls ml-inference]
  iter 3 → final natural-language answer (no tool calls)

LLM gateway
───────────
LiteLLM is used as a provider-agnostic gateway.  Groq is the default
provider (model: groq/llama3-8b-8192).  The GROQ_API_KEY env var is
required and is picked up automatically by LiteLLM.

Override the model with the LITELLM_MODEL env var, e.g.:
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

Tu misión es responder preguntas sobre predicciones de ventas de sociedades \
siguiendo OBLIGATORIAMENTE este proceso en orden:

PASO 1 — RAZONAMIENTO INICIAL
Identifica la provincia y el mes mencionados en la pregunta. \
Si hay ambigüedad, elige el más probable y procede.

PASO 2 — HERRAMIENTA get_province_data
Usa esta herramienta para obtener los datos históricos de exportaciones \
de la provincia y mes identificados. Son los features del modelo predictivo.

PASO 3 — HERRAMIENTA call_inference
Usa esta herramienta con los datos del paso anterior para obtener la predicción.

PASO 4 — RESPUESTA FINAL
Genera una respuesta en español, profesional y contextualizada que incluya:
• El valor predicho de ventas y exportaciones totales de sociedades
• Una breve interpretación del resultado (¿es alto/bajo para esa provincia?)
• Contexto sobre la estacionalidad del mes si es relevante

REGLAS OBLIGATORIAS:
- Provincias siempre en MAYÚSCULAS al llamar herramientas
- Los meses son números del 1 al 12
- Responde siempre en español
- No inventes datos; usa únicamente lo que retornan las herramientas
- Si una herramienta retorna un error, infórmalo claramente al usuario

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
            logger.info("[ReAct iter %d/%d] Calling %s", iteration, MAX_ITERATIONS, self.model)
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
                razonamiento.append(f"[iter {iteration}] Respuesta final generada por el LLM.")
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
                    logger.warning("[iter %d] Invalid JSON in tool arguments: %s", iteration, tc.function.arguments)

                razonamiento.append(f"[iter {iteration}] → {tool_name}({_fmt_args(args)})")
                agent_tool_calls_total.labels(tool_name=tool_name).inc()

                observation = self._dispatch(tool_name, args)

                razonamiento.append(f"[iter {iteration}] ← {tool_name}: {_fmt_result(observation)}")
                logger.info("[iter %d] tool=%s  result_keys=%s", iteration, tool_name, list(observation.keys()))

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
        logger.warning("ReAct loop hit MAX_ITERATIONS=%d for question: %s", MAX_ITERATIONS, pregunta)
        razonamiento.append(f"[WARN] Máximo de iteraciones ({MAX_ITERATIONS}) alcanzado sin respuesta final.")
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
