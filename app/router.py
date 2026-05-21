"""Enrutador: decide si una pregunta se resuelve con RAG, TOOL, BOTH o NONE.

Estrategia (documentada en el README):
  1. Clasificador LLM con prompt breve y salida estructurada JSON.
  2. Si el LLM falla o no está disponible, se usa una heurística por reglas
     (palabras clave) como red de seguridad (fallback).

Se registra siempre la decisión y su razón.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from app import llm
from app.logging_utils import get_logger

log = get_logger("router")

VALID_ROUTES = {"RAG", "TOOL", "BOTH", "NONE"}

# Palabras clave que sugieren el uso de tools (datos en vivo).
_TOOL_KEYWORDS = (
    "tipo de cambio", "cambio de divisa", "convierte", "convertir", "conversión",
    "cotización", "divisa", "euros a", "dólares", "dolar", "usd", "eur",
    "mercado", "bolsa", "abierto", "cerrado", "nyse", "nasdaq", "ibex",
)

SYSTEM_PROMPT = """Eres un enrutador de un asistente. Decides cómo responder cada pregunta.

Opciones de ruta:
- RAG: la respuesta está en la documentación interna en PDF (manuales, procedimientos, normativa, cifras de los documentos).
- TOOL: la respuesta requiere datos en vivo que aportan las herramientas externas disponibles.
- BOTH: hace falta combinar un dato del PDF con una herramienta (p. ej. tomar una cifra del PDF y convertirla de divisa).
- NONE: la pregunta no se puede responder ni con el PDF ni con las herramientas.

Responde SOLO con un objeto JSON válido con esta forma exacta:
{"route": "RAG|TOOL|BOTH|NONE", "reason": "explicación muy breve en español"}"""


@dataclass
class RouteDecision:
    route: str
    reason: str
    via: str  # "llm" o "heuristica"


def _heuristic_route(query: str, tool_names: list[str]) -> RouteDecision:
    """Fallback por reglas cuando el LLM no está disponible."""
    q = query.lower()
    mentions_tool = any(kw in q for kw in _TOOL_KEYWORDS) and bool(tool_names)
    mentions_convert = any(w in q for w in ("convierte", "convertir", "conversión", "a dólares", "a usd"))
    if mentions_convert and tool_names:
        return RouteDecision("BOTH", "Heurística: conversión que combina cifra del PDF y una tool.", "heuristica")
    if mentions_tool:
        return RouteDecision("TOOL", "Heurística: la pregunta menciona datos en vivo (divisas/mercados).", "heuristica")
    return RouteDecision("RAG", "Heurística: por defecto se consulta la documentación en PDF.", "heuristica")


def classify(query: str, tools: list[Any], *,
             chat_json_fn: Callable[..., dict] | None = None) -> RouteDecision:
    """Clasifica la pregunta en RAG/TOOL/BOTH/NONE.

    `tools` es la lista de ToolSpec descubiertas del MCP. `chat_json_fn` permite
    inyectar un LLM simulado en los tests.
    """
    tool_names = [getattr(t, "name", str(t)) for t in tools]
    tools_desc = "\n".join(
        f"- {getattr(t, 'name', '?')}: {getattr(t, 'description', '')}" for t in tools
    ) or "(no hay herramientas disponibles en este momento)"

    chat_json = chat_json_fn or llm.chat_json
    try:
        data = chat_json([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content":
                f"Herramientas externas disponibles:\n{tools_desc}\n\n"
                f"Pregunta del usuario:\n{query}"},
        ])
        route = str(data.get("route", "")).strip().upper()
        reason = str(data.get("reason", "")).strip() or "Sin razón explícita."
        if route in VALID_ROUTES:
            # Si el LLM elige TOOL/BOTH pero no hay tools, degradamos a RAG.
            if route in ("TOOL", "BOTH") and not tool_names:
                return RouteDecision("RAG", "No hay herramientas MCP disponibles; se usa solo RAG.", "llm")
            decision = RouteDecision(route, reason, "llm")
            log.info("Ruta elegida por LLM: %s (%s)", route, reason)
            return decision
        log.warning("El LLM devolvió una ruta inválida: %r", route)
    except Exception as exc:  # noqa: BLE001
        log.warning("Fallo del clasificador LLM, se usa heurística: %s", exc)

    decision = _heuristic_route(query, tool_names)
    log.info("Ruta elegida por heurística: %s (%s)", decision.route, decision.reason)
    return decision
