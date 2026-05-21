"""Ingesta de PDFs: extracción de texto, chunking jerárquico e indexado.

Expone funciones para indexar un PDF concreto, indexar todo el corpus y
reindexar desde cero. Usado tanto por la UI como por el script de reindexado.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from pypdf import PdfReader

from app.chunking import chunk_document
from app.config import settings
from app.logging_utils import get_logger, log_event
from app.vectorstore import ParentStore, VectorStore

log = get_logger("ingest")


@dataclass
class IngestResult:
    source: str
    pages: int
    parents: int
    children: int
    ok: bool = True
    error: str = ""


def extract_pages(pdf_path: str) -> list[str]:
    """Extrae el texto de cada página del PDF (se asume texto seleccionable)."""
    reader = PdfReader(pdf_path)
    pages: list[str] = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception as exc:  # noqa: BLE001
            log.warning("No se pudo extraer una página de %s: %s", pdf_path, exc)
            pages.append("")
    return pages


def ingest_pdf(pdf_path: str, vector_store: VectorStore,
               parent_store: ParentStore) -> IngestResult:
    """Indexa un único PDF en los almacenes de nivel 1 y nivel 2."""
    source = os.path.basename(pdf_path)
    try:
        pages = extract_pages(pdf_path)
        if not any(p.strip() for p in pages):
            return IngestResult(source, len(pages), 0, 0, ok=False,
                                error="El PDF no contiene texto seleccionable.")

        doc = chunk_document(
            pages, source,
            parent_size=settings.parent_chunk_size,
            parent_overlap=settings.parent_chunk_overlap,
            child_size=settings.child_chunk_size,
            child_overlap=settings.child_chunk_overlap,
        )
        # Reindexado idempotente: eliminamos versión previa de esta fuente.
        vector_store.delete_source(source)
        parent_store.delete_source(source)

        parent_store.add_parents(doc.parents)
        vector_store.add_children(doc.children)

        result = IngestResult(source, len(pages), len(doc.parents), len(doc.children))
        log_event(log, "PDF indexado", source=source, pages=result.pages,
                  parents=result.parents, children=result.children)
        return result
    except Exception as exc:  # noqa: BLE001
        log.exception("Fallo al indexar %s", pdf_path)
        return IngestResult(source, 0, 0, 0, ok=False, error=str(exc))


def list_pdf_files(pdf_dir: str | None = None) -> list[str]:
    """Lista las rutas de los PDFs presentes en el directorio de corpus."""
    pdf_dir = pdf_dir or settings.pdf_dir
    if not os.path.isdir(pdf_dir):
        return []
    return sorted(
        os.path.join(pdf_dir, f)
        for f in os.listdir(pdf_dir)
        if f.lower().endswith(".pdf")
    )


def ingest_corpus(vector_store: VectorStore, parent_store: ParentStore,
                  pdf_dir: str | None = None) -> list[IngestResult]:
    """Indexa todos los PDFs del directorio de corpus."""
    results = [ingest_pdf(path, vector_store, parent_store)
               for path in list_pdf_files(pdf_dir)]
    log_event(log, "Corpus indexado", documentos=len(results),
              chunks_hijos=sum(r.children for r in results))
    return results


def reindex_all(vector_store: VectorStore, parent_store: ParentStore,
                pdf_dir: str | None = None) -> list[IngestResult]:
    """Borra los índices y reindexa todo el corpus desde cero."""
    log_event(log, "Reindexado completo iniciado")
    vector_store.reset()
    parent_store.reset()
    return ingest_corpus(vector_store, parent_store, pdf_dir)
