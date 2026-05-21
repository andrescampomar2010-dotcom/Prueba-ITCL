"""Arranque automático: deja el sistema usable sin pasos manuales.

- Copia los PDFs semilla (incluido el PDF de ejemplo) al corpus si está vacío.
- Indexa el corpus automáticamente si el índice todavía no tiene contenido.

Esto cumple el requisito "docker compose up levanta todo y el sistema es
usable" sin obligar a reindexar a mano.
"""
from __future__ import annotations

import os
import shutil

from app.config import settings
from app.logging_utils import get_logger, log_event
from app.pdf_ingest import ingest_corpus, list_pdf_files
from app.vectorstore import ParentStore, VectorStore

log = get_logger("bootstrap")


def ensure_seed_pdfs() -> int:
    """Copia los PDFs semilla al directorio de corpus si este está vacío."""
    settings.ensure_dirs()
    if list_pdf_files(settings.pdf_dir):
        return 0  # el corpus ya tiene PDFs (p. ej. subidos por el usuario)

    seed_dir = settings.seed_pdf_dir
    if not os.path.isdir(seed_dir):
        return 0
    copied = 0
    for name in sorted(os.listdir(seed_dir)):
        if name.lower().endswith(".pdf"):
            shutil.copy2(os.path.join(seed_dir, name),
                         os.path.join(settings.pdf_dir, name))
            copied += 1
    if copied:
        log_event(log, "PDFs semilla copiados al corpus", copiados=copied)
    return copied


def auto_index_if_empty(vector_store: VectorStore, parent_store: ParentStore) -> bool:
    """Indexa el corpus si el índice está vacío. Devuelve True si indexó algo."""
    if vector_store.count() > 0:
        return False
    if not list_pdf_files(settings.pdf_dir):
        return False
    if not settings.openai_configured:
        log.warning("No se puede autoindexar: falta OPENAI_API_KEY.")
        return False
    log_event(log, "Autoindexado inicial del corpus")
    ingest_corpus(vector_store, parent_store)
    return True
