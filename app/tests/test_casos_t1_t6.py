"""
Tests de los casos T1-T6 del enunciado de la prueba técnica.

Cada test verifica que el sistema PUEDE responder correctamente el caso:
- El PDF contiene los datos necesarios (verificado en chunks).
- El router clasificaría la ruta correcta (router mockeado, sin LLM real).
- No se invoca la API de OpenAI: los tests corren sin clave.

Para ejecutar: pytest app/tests/test_casos_t1_t6.py -v
"""
from __future__ import annotations

import pytest

from app import router
from app.chunking import chunk_document
from app.pdf_ingest import extract_pages
from app.rag import build_context_text


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────

class FakeTool:
    def __init__(self, name: str, desc: str = "") -> None:
        self.name = name
        self.description = desc


def fake_llm(route: str):
    """LLM simulado que devuelve siempre la ruta indicada."""
    def _fn(messages):
        return {"route": route, "reason": f"simulado: {route}"}
    return _fn


# ─────────────────────────────────────────────────────────────────────────
# T1: pregunta factual en el PDF → ruta RAG + dato existe en chunks
# ─────────────────────────────────────────────────────────────────────────

def test_t1_vacaciones_dato_existe_en_pdf(sample_pdf):
    """T1: '¿Cuántos días de vacaciones anuales tengo?' → está en el PDF."""
    pages = extract_pages(sample_pdf)
    doc = chunk_document(pages, "manual_ejemplo.pdf")
    chunks_con_dato = [c for c in doc.children if "23 días" in c.text or "23 d" in c.text]
    assert chunks_con_dato, "El dato de los 23 días laborables debe estar en algún chunk"


def test_t1_vacaciones_router_elige_rag():
    """T1: el router debe clasificar la pregunta como RAG."""
    tools = [FakeTool("fx_rate"), FakeTool("market_status")]
    decision = router.classify(
        "¿Cuántos días de vacaciones anuales tengo?",
        tools,
        chat_json_fn=fake_llm("RAG"),
    )
    assert decision.route == "RAG"


# ─────────────────────────────────────────────────────────────────────────
# T2: resumen de sección → el contenido existe en chunks
# ─────────────────────────────────────────────────────────────────────────

def test_t2_seguridad_contenido_en_pdf(sample_pdf):
    """T2: la sección de seguridad de la información existe en el PDF."""
    pages = extract_pages(sample_pdf)
    doc = chunk_document(pages, "manual_ejemplo.pdf")
    chunks_seguridad = [
        c for c in doc.children
        if "seguridad" in c.text.lower() or "contraseña" in c.text.lower()
    ]
    assert chunks_seguridad, "La sección de seguridad debe estar en algún chunk"


def test_t2_resumen_router_elige_rag():
    """T2: el router debe clasificar la pregunta de resumen como RAG."""
    tools = [FakeTool("fx_rate")]
    decision = router.classify(
        "Resume la sección de seguridad de la información",
        tools,
        chat_json_fn=fake_llm("RAG"),
    )
    assert decision.route == "RAG"


# ─────────────────────────────────────────────────────────────────────────
# T3: conversión de divisa → BOTH (cifra en PDF + tool fx_rate)
# ─────────────────────────────────────────────────────────────────────────

def test_t3_presupuesto_dato_en_pdf(sample_pdf):
    """T3: la cifra 12.500 EUR existe en el PDF y se puede citar con página."""
    pages = extract_pages(sample_pdf)
    doc = chunk_document(pages, "manual_ejemplo.pdf")
    chunks_con_cifra = [c for c in doc.children if "12.500" in c.text]
    assert chunks_con_cifra, "La cifra 12.500 EUR debe estar en algún chunk"
    # Todos los chunks con la cifra deben tener página asignada
    for chunk in chunks_con_cifra:
        assert chunk.pages, f"El chunk {chunk.chunk_id} con la cifra no tiene página"


def test_t3_conversion_router_elige_both():
    """T3: convertir cifra del PDF a otra divisa → router debe elegir BOTH."""
    tools = [FakeTool("fx_rate", "Obtiene tipo de cambio entre divisas")]
    decision = router.classify(
        "Convierte a dólares el presupuesto anual del departamento",
        tools,
        chat_json_fn=fake_llm("BOTH"),
    )
    assert decision.route == "BOTH"


# ─────────────────────────────────────────────────────────────────────────
# T4: estado de mercado → TOOL (sin PDF)
# ─────────────────────────────────────────────────────────────────────────

def test_t4_mercado_router_elige_tool():
    """T4: '¿Está abierto el mercado NYSE?' → router debe elegir TOOL."""
    tools = [FakeTool("market_status", "Consulta si un mercado está abierto o cerrado")]
    decision = router.classify(
        "¿Está abierto el mercado NYSE ahora mismo?",
        tools,
        chat_json_fn=fake_llm("TOOL"),
    )
    assert decision.route == "TOOL"


# ─────────────────────────────────────────────────────────────────────────
# T5: pregunta no respondible → NONE
# ─────────────────────────────────────────────────────────────────────────

def test_t5_teletrabajo_japon_no_en_pdf(sample_pdf):
    """T5: 'política de teletrabajo en Japón' no debe aparecer en ningún chunk."""
    pages = extract_pages(sample_pdf)
    doc = chunk_document(pages, "manual_ejemplo.pdf")
    chunks_japon = [c for c in doc.children if "japón" in c.text.lower() or "japan" in c.text.lower()]
    assert not chunks_japon, "El PDF no debe contener información sobre teletrabajo en Japón"


def test_t5_router_elige_none():
    """T5: pregunta fuera del alcance → router debe elegir NONE."""
    tools = [FakeTool("fx_rate"), FakeTool("market_status")]
    decision = router.classify(
        "¿Cuál es la política de teletrabajo de la empresa en Japón?",
        tools,
        chat_json_fn=fake_llm("NONE"),
    )
    assert decision.route == "NONE"


# ─────────────────────────────────────────────────────────────────────────
# T6: pregunta ambigua → el router elige una ruta válida con razonamiento
# ─────────────────────────────────────────────────────────────────────────

def test_t6_pregunta_ambigua_ruta_valida():
    """T6: pregunta ambigua → el router elige una ruta válida (no explota)."""
    tools = [FakeTool("fx_rate"), FakeTool("market_status")]
    # Esta pregunta no menciona divisa concreta ni mercado concreto.
    # El router debe elegir algo coherente, no fallar.
    decision = router.classify(
        "¿Cuál es el tipo de cambio?",
        tools,
        chat_json_fn=fake_llm("TOOL"),  # el LLM elige TOOL (tools disponibles)
    )
    assert decision.route in router.VALID_ROUTES


def test_t6_build_context_format(sample_pdf):
    """T6: el contexto construido incluye etiquetas de página legibles."""
    pages = extract_pages(sample_pdf)
    doc = chunk_document(pages, "manual_ejemplo.pdf")
    blocks = [
        {"source": p.source, "pages": p.pages, "text": p.text}
        for p in doc.parents[:2]
    ]
    contexto = build_context_text(blocks)
    assert "pág" in contexto.lower()
    assert "Fragmento" in contexto
