"""Test del enrutador: decisión RAG / TOOL / BOTH / NONE y fallback heurístico."""
from __future__ import annotations

from app import router


class FakeTool:
    """Doble de ToolSpec: solo necesita name y description."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.description = f"herramienta {name}"


def fake_json(route: str):
    """Devuelve un chat_json simulado que responde una ruta fija."""
    def _fn(messages):
        return {"route": route, "reason": f"simulado: {route}"}
    return _fn


def test_clasifica_rag():
    decision = router.classify(
        "¿Cuántos días de vacaciones tengo?", [FakeTool("fx_rate")],
        chat_json_fn=fake_json("RAG"))
    assert decision.route == "RAG"
    assert decision.via == "llm"


def test_clasifica_tool():
    decision = router.classify(
        "¿Está abierto el NYSE?", [FakeTool("market_status")],
        chat_json_fn=fake_json("TOOL"))
    assert decision.route == "TOOL"


def test_clasifica_both():
    decision = router.classify(
        "Convierte el presupuesto a dólares", [FakeTool("fx_rate")],
        chat_json_fn=fake_json("BOTH"))
    assert decision.route == "BOTH"


def test_clasifica_none():
    decision = router.classify(
        "¿Quién ganó el mundial de 1930?", [FakeTool("fx_rate")],
        chat_json_fn=fake_json("NONE"))
    assert decision.route == "NONE"


def test_tool_sin_tools_degrada_a_rag():
    """Si el LLM elige TOOL pero no hay tools MCP, debe degradar a RAG."""
    decision = router.classify(
        "pregunta cualquiera", [], chat_json_fn=fake_json("TOOL"))
    assert decision.route == "RAG"


def test_ruta_invalida_usa_heuristica():
    decision = router.classify(
        "¿Está abierto el mercado NYSE?", [FakeTool("market_status")],
        chat_json_fn=fake_json("RUTA_INEXISTENTE"))
    assert decision.route in router.VALID_ROUTES
    assert decision.via == "heuristica"


def test_excepcion_del_llm_usa_heuristica():
    def boom(messages):
        raise RuntimeError("LLM no disponible")
    decision = router.classify(
        "¿qué dice el manual sobre seguridad?", [FakeTool("fx_rate")],
        chat_json_fn=boom)
    assert decision.via == "heuristica"
    assert decision.route == "RAG"


def test_heuristica_detecta_tool():
    decision = router._heuristic_route(
        "¿está abierto el mercado nyse ahora?", ["market_status"])
    assert decision.route in ("TOOL", "BOTH")


def test_heuristica_rag_por_defecto():
    decision = router._heuristic_route(
        "¿qué procedimiento sigo para pedir vacaciones?", ["market_status"])
    assert decision.route == "RAG"
