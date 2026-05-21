"""RAG jerárquico: recuperación de dos niveles + generación con citas.

Flujo:
  1. Nivel 1: se recuperan los chunks HIJOS más similares a la consulta.
  2. Nivel 2: se identifican sus PADRES y se reconstruye el contexto amplio.
  3. Generación: el LLM responde SOLO con ese contexto y cita las páginas.
     Si la información no está en el contexto, debe decirlo explícitamente.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app import llm
from app.config import settings
from app.logging_utils import Trace, get_logger
from app.vectorstore import ParentStore, VectorStore

log = get_logger("rag")

NO_CONTEXT_MSG = (
    "No he encontrado información sobre eso en la documentación PDF indexada. "
    "No puedo responder sin una fuente verificable."
)

GENERATION_SYSTEM = """Eres un asistente que responde EXCLUSIVAMENTE en español y
SOLO a partir del CONTEXTO de documentación interna que se te proporciona.

Reglas estrictas:
- Usa únicamente la información del CONTEXTO. No uses conocimiento general.
- Cita SIEMPRE la página entre paréntesis tras cada afirmación, así: (pág. 3).
  Si un dato abarca varias páginas, usa (págs. 3-4).
- Si la respuesta NO está en el contexto, dilo claramente: "No aparece en la
  documentación indexada." No inventes nada.
- Sé claro y conciso. Responde en español."""


@dataclass
class Citation:
    source: str
    pages: list[int]
    chunk_id: str
    snippet: str

    @property
    def label(self) -> str:
        if len(self.pages) == 1:
            return f"{self.source}, pág. {self.pages[0]}"
        if self.pages:
            return f"{self.source}, págs. {self.pages[0]}-{self.pages[-1]}"
        return self.source


@dataclass
class RetrievalResult:
    context_blocks: list[dict[str, Any]] = field(default_factory=list)
    child_hits: list[dict[str, Any]] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)

    @property
    def has_context(self) -> bool:
        return bool(self.context_blocks)


def retrieve(query: str, vector_store: VectorStore, parent_store: ParentStore,
             *, trace: Trace | None = None) -> RetrievalResult:
    """Recuperación jerárquica nivel 1 -> nivel 2."""
    child_hits = vector_store.query(query, k=settings.top_k_children)
    result = RetrievalResult(child_hits=child_hits)
    if not child_hits:
        return result

    # Nivel 2: padres únicos, preservando el orden de relevancia de los hijos.
    parent_ids: list[str] = []
    for hit in child_hits:
        pid = hit.get("parent_id") or ""
        if pid and pid not in parent_ids:
            parent_ids.append(pid)
    parent_ids = parent_ids[: settings.max_parents]

    parents = parent_store.get(parent_ids)
    result.context_blocks = parents

    # Citas: una por padre usado en el contexto.
    seen: set[str] = set()
    for parent in parents:
        key = f"{parent['source']}|{parent['pages']}"
        if key in seen:
            continue
        seen.add(key)
        result.citations.append(Citation(
            source=parent["source"],
            pages=parent["pages"],
            chunk_id=parent["parent_id"],
            snippet=parent["text"][:200].strip(),
        ))

    if trace is not None:
        for hit in child_hits:
            trace.add_chunk(
                chunk_id=hit["chunk_id"],
                page=hit.get("page"),
                source=hit.get("source", ""),
                snippet=hit.get("text", ""),
                distance=hit.get("distance"),
            )
    return result


def build_context_text(blocks: list[dict[str, Any]]) -> str:
    """Construye el bloque de CONTEXTO etiquetado con fuente y páginas."""
    parts: list[str] = []
    for i, block in enumerate(blocks, start=1):
        pages = block.get("pages") or []
        if len(pages) == 1:
            page_label = f"pág. {pages[0]}"
        elif pages:
            page_label = f"págs. {pages[0]}-{pages[-1]}"
        else:
            page_label = "pág. ?"
        parts.append(
            f"[Fragmento {i} — Fuente: {block.get('source', '?')}, {page_label}]\n"
            f"{block.get('text', '').strip()}"
        )
    return "\n\n".join(parts)


def generate_answer(query: str, retrieval: RetrievalResult,
                    history: list[dict[str, str]] | None = None) -> str:
    """Genera la respuesta final del RAG a partir del contexto recuperado."""
    if not retrieval.has_context:
        return NO_CONTEXT_MSG

    context_text = build_context_text(retrieval.context_blocks)
    messages: list[dict[str, Any]] = [{"role": "system", "content": GENERATION_SYSTEM}]
    for turn in (history or [])[-4:]:
        messages.append(turn)
    messages.append({
        "role": "user",
        "content": (
            f"CONTEXTO:\n{context_text}\n\n"
            f"PREGUNTA: {query}\n\n"
            "Responde en español citando las páginas. Si no está en el contexto, "
            "indica que no aparece en la documentación."
        ),
    })
    return llm.chat_text(messages)


def answer_with_rag(query: str, vector_store: VectorStore, parent_store: ParentStore,
                    *, history: list[dict[str, str]] | None = None,
                    trace: Trace | None = None) -> tuple[str, RetrievalResult]:
    """Atajo: recuperación + generación. Devuelve (texto, RetrievalResult)."""
    retrieval = retrieve(query, vector_store, parent_store, trace=trace)
    text = generate_answer(query, retrieval, history=history)
    return text, retrieval
