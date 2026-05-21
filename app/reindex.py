"""Reindexado del corpus por línea de comandos.

Uso:
    python -m app.reindex          # reindexa todo el corpus desde cero
    python -m app.reindex --stats  # solo muestra el estado del índice

Invocado también por `make reindex`. La UI tiene además un botón equivalente.
"""
from __future__ import annotations

import sys

from app import llm
from app.config import settings
from app.logging_utils import get_logger
from app.pdf_ingest import list_pdf_files, reindex_all
from app.vectorstore import ParentStore, VectorStore

log = get_logger("reindex")


def _stores() -> tuple[VectorStore, ParentStore]:
    settings.ensure_dirs()
    vs = VectorStore(embedder=llm.embed_texts, persist_dir=settings.chroma_dir)
    ps = ParentStore(db_path=settings.parent_db_path)
    return vs, ps


def show_stats() -> None:
    vs, ps = _stores()
    print(f"PDFs en el corpus : {len(list_pdf_files())}")
    print(f"Fuentes indexadas : {', '.join(vs.sources()) or '(ninguna)'}")
    print(f"Chunks hijo       : {vs.count()}")
    print(f"Chunks padre      : {ps.count()}")


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if "--stats" in argv:
        show_stats()
        return 0

    if not settings.openai_configured:
        print("ERROR: OPENAI_API_KEY no está configurada. "
              "Copia .env.example a .env y añade tu clave.", file=sys.stderr)
        return 1

    vs, ps = _stores()
    pdfs = list_pdf_files()
    if not pdfs:
        print(f"No hay PDFs en {settings.pdf_dir}. Añade alguno y reintenta.")
        return 1

    print(f"Reindexando {len(pdfs)} PDF(s) desde {settings.pdf_dir} ...")
    results = reindex_all(vs, ps)
    for r in results:
        estado = "OK" if r.ok else f"ERROR ({r.error})"
        print(f"  - {r.source}: {r.pages} págs, {r.parents} padres, "
              f"{r.children} hijos  [{estado}]")
    print("Reindexado completado.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
