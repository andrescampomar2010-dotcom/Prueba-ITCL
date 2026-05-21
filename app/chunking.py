"""Chunking jerárquico de dos niveles con mapeo de páginas.

Estrategia (documentada también en el README):

  - Nivel 2 (padres): ventanas grandes de texto (~1800 caracteres) que aportan
    el contexto amplio para la generación final.
  - Nivel 1 (hijos): fragmentos pequeños (~450 caracteres) derivados de cada
    padre; son los que se indexan y se recuperan por similitud.

Cada hijo guarda el `parent_id` para poder reconstruir el contexto grande
(patrón "small-to-big" / parent retrieval).

El texto del PDF se concatena conservando un mapa de páginas (offset ->
página), de modo que cada chunk conoce su página o rango de páginas.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Separadores ordenados de mayor a menor relevancia semántica.
_SEPARATORS = ["\n\n", "\n", ". ", "; ", ", ", " ", ""]


@dataclass
class PageSpan:
    page: int
    start: int  # offset de carácter inicial (inclusive) en el texto completo
    end: int    # offset de carácter final (exclusive)


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source: str
    char_start: int
    char_end: int
    pages: list[int]
    parent_id: str | None = None
    index: int = 0

    @property
    def page(self) -> int:
        """Página principal del chunk (la primera de su rango)."""
        return self.pages[0] if self.pages else 0

    @property
    def page_label(self) -> str:
        if not self.pages:
            return "?"
        if len(self.pages) == 1:
            return f"pág. {self.pages[0]}"
        return f"págs. {self.pages[0]}-{self.pages[-1]}"


@dataclass
class DocumentChunks:
    source: str
    parents: list[Chunk] = field(default_factory=list)
    children: list[Chunk] = field(default_factory=list)


def build_page_map(pages_text: list[str]) -> tuple[str, list[PageSpan]]:
    """Concatena las páginas y devuelve (texto_completo, mapa_de_páginas).

    Las páginas se numeran desde 1. Entre páginas se inserta un salto de línea
    para no fusionar palabras de páginas distintas.
    """
    full_parts: list[str] = []
    spans: list[PageSpan] = []
    cursor = 0
    for i, text in enumerate(pages_text):
        page_number = i + 1
        chunk = (text or "").strip()
        start = cursor
        full_parts.append(chunk)
        cursor += len(chunk)
        spans.append(PageSpan(page=page_number, start=start, end=cursor))
        if i < len(pages_text) - 1:
            full_parts.append("\n")
            cursor += 1
    return "".join(full_parts), spans


def pages_for_range(start: int, end: int, spans: list[PageSpan]) -> list[int]:
    """Devuelve la lista de páginas que solapan con el rango [start, end)."""
    result: list[int] = []
    for span in spans:
        if span.start < end and start < span.end:
            result.append(span.page)
    return result or ([spans[0].page] if spans else [1])


def _split_recursive(text: str, size: int, overlap: int) -> list[tuple[int, int]]:
    """Divide `text` en ventanas <= size respetando límites semánticos.

    Devuelve una lista de pares (offset_inicio, offset_fin) sobre `text`.
    """
    text_len = len(text)
    if text_len <= size:
        return [(0, text_len)] if text_len > 0 else []

    spans: list[tuple[int, int]] = []
    start = 0
    while start < text_len:
        end = min(start + size, text_len)
        if end < text_len:
            # Busca el mejor punto de corte hacia atrás dentro de la ventana.
            window = text[start:end]
            cut = -1
            for sep in _SEPARATORS:
                if sep == "":
                    break
                pos = window.rfind(sep)
                if pos > size * 0.5:  # corte razonablemente avanzado
                    cut = pos + len(sep)
                    break
            if cut > 0:
                end = start + cut
        spans.append((start, end))
        if end >= text_len:
            break
        start = max(end - overlap, start + 1)
    return spans


def chunk_document(pages_text: list[str], source: str, *,
                   parent_size: int = 1800, parent_overlap: int = 200,
                   child_size: int = 450, child_overlap: int = 80) -> DocumentChunks:
    """Genera la jerarquía de chunks (padres + hijos) para un documento."""
    full_text, spans = build_page_map(pages_text)
    doc = DocumentChunks(source=source)
    if not full_text.strip():
        return doc

    parent_spans = _split_recursive(full_text, parent_size, parent_overlap)
    for p_idx, (p_start, p_end) in enumerate(parent_spans):
        parent_text = full_text[p_start:p_end]
        parent_id = f"{source}::p{p_idx}"
        parent = Chunk(
            chunk_id=parent_id,
            text=parent_text,
            source=source,
            char_start=p_start,
            char_end=p_end,
            pages=pages_for_range(p_start, p_end, spans),
            parent_id=None,
            index=p_idx,
        )
        doc.parents.append(parent)

        child_spans = _split_recursive(parent_text, child_size, child_overlap)
        for c_idx, (c_start, c_end) in enumerate(child_spans):
            abs_start = p_start + c_start
            abs_end = p_start + c_end
            child = Chunk(
                chunk_id=f"{parent_id}::c{c_idx}",
                text=parent_text[c_start:c_end],
                source=source,
                char_start=abs_start,
                char_end=abs_end,
                pages=pages_for_range(abs_start, abs_end, spans),
                parent_id=parent_id,
                index=c_idx,
            )
            doc.children.append(child)
    return doc
