"""Test de citación/página: el chunking jerárquico conserva la página correcta.

Se trabaja sobre el PDF de ejemplo real (generado por el fixture). No requiere
ni clave de OpenAI ni red: valida directamente el mapeo de páginas.
"""
from __future__ import annotations

from app import rag
from app.chunking import build_page_map, chunk_document, pages_for_range
from app.pdf_ingest import extract_pages


def test_mapeo_de_paginas_basico():
    full_text, spans = build_page_map(["Página uno.", "Página dos.", "Página tres."])
    assert "Página uno." in full_text and "Página tres." in full_text
    assert [s.page for s in spans] == [1, 2, 3]
    # Un offset dentro de la primera página devuelve la página 1.
    assert pages_for_range(0, 5, spans) == [1]


def test_chunks_conservan_pagina_valida(sample_pdf):
    pages = extract_pages(sample_pdf)
    assert len(pages) >= 5, "El manual de ejemplo debe tener varias páginas"

    doc = chunk_document(pages, "manual_ejemplo.pdf")
    assert doc.parents, "Debe haber chunks padre"
    assert doc.children, "Debe haber chunks hijo"

    n_pages = len(pages)
    for child in doc.children:
        assert child.pages, "Todo chunk hijo debe tener al menos una página"
        for page in child.pages:
            assert 1 <= page <= n_pages
        assert child.parent_id is not None


def test_cifra_presupuesto_en_su_pagina(sample_pdf):
    """La cifra 12.500 EUR debe citarse en la página donde realmente aparece."""
    pages = extract_pages(sample_pdf)

    pagina_real = None
    for numero, texto in enumerate(pages, start=1):
        if "12.500" in texto:
            pagina_real = numero
            break
    assert pagina_real is not None, "El PDF de ejemplo debe contener 12.500 EUR"

    doc = chunk_document(pages, "manual_ejemplo.pdf")
    chunks_con_cifra = [c for c in doc.children if "12.500" in c.text]
    assert chunks_con_cifra, "Algún chunk hijo debe contener la cifra"
    for chunk in chunks_con_cifra:
        assert pagina_real in chunk.pages, (
            f"El chunk con la cifra dice páginas {chunk.pages} "
            f"pero la cifra está en la página {pagina_real}"
        )


def test_dato_vacaciones_presente(sample_pdf):
    pages = extract_pages(sample_pdf)
    doc = chunk_document(pages, "manual_ejemplo.pdf")
    chunks_vacaciones = [c for c in doc.children if "23 días" in c.text]
    assert chunks_vacaciones, "El dato de los 23 días debe estar en algún chunk"
    for chunk in chunks_vacaciones:
        assert all(1 <= p <= len(pages) for p in chunk.pages)


def test_reconstruccion_de_contexto_padre(sample_pdf):
    """Cada hijo apunta a un padre existente cuyo texto lo contiene."""
    pages = extract_pages(sample_pdf)
    doc = chunk_document(pages, "manual_ejemplo.pdf")

    parents_by_id = {p.chunk_id: p for p in doc.parents}
    for child in doc.children:
        assert child.parent_id in parents_by_id
        parent = parents_by_id[child.parent_id]
        assert child.text in parent.text


def test_contexto_incluye_etiqueta_de_pagina(sample_pdf):
    pages = extract_pages(sample_pdf)
    doc = chunk_document(pages, "manual_ejemplo.pdf")
    blocks = [
        {"source": p.source, "pages": p.pages, "text": p.text}
        for p in doc.parents[:2]
    ]
    contexto = rag.build_context_text(blocks)
    assert "pág" in contexto.lower()
    assert "Fuente:" in contexto
