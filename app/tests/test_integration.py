"""Pruebas de integración: flujos end-to-end del sistema.

Estas pruebas verifican que los componentes funcionan juntos correctamente.
No son puramente unitarias: usan almacenes reales (aunque efímeros).
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from app.chunking import chunk_document
from app.orchestrator import Orchestrator
from app.vectorstore import ParentStore, VectorStore


# Fixture: almacenes en memoria para tests
@pytest.fixture
def temp_vectorstore():
    """Crea un VectorStore temporal para tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Mock embedder determinista para tests
        def mock_embedder(texts):
            # Retorna embeddings sintéticos basados en hash del texto
            return [[(hash(t) % 768) / 768.0 for _ in range(768)] for t in texts]
        
        yield VectorStore(embedder=mock_embedder, persist_dir=tmpdir)


@pytest.fixture
def temp_parentstore():
    """Crea un ParentStore temporal para tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "test.db")
        yield ParentStore(db_path=db_path)


# ─────────────────────────────────────────────────────────────────────────
# Pruebas de integración
# ─────────────────────────────────────────────────────────────────────────

def test_end_to_end_pdf_indexing(temp_vectorstore, temp_parentstore):
    """Verifica el flujo completo: PDF → chunking → indexación."""
    # Simular un PDF con 2 páginas
    pages = [
        "Página 1: Políticas de vacaciones\nTodo empleado tiene 20 días.",
        "Página 2: Detalles\nLos días se distribuyen en bloques de 5 días.",
    ]
    
    # Chunking jerárquico
    doc = chunk_document(pages, source="test.pdf", parent_size=200, parent_overlap=20,
                         child_size=80, child_overlap=10)
    
    # Debe haber parents y children
    assert len(doc.parents) > 0, "Debería haber chunks padre"
    assert len(doc.children) > 0, "Debería haber chunks hijo"
    
    # Indexar
    n_parents = temp_parentstore.add_parents(doc.parents)
    n_children = temp_vectorstore.add_children(doc.children)
    
    assert n_parents == len(doc.parents)
    assert n_children == len(doc.children)
    
    # Verificar que se guardaron
    assert temp_vectorstore.count() == n_children
    assert temp_parentstore.count() == n_parents


def test_end_to_end_vector_search(temp_vectorstore, temp_parentstore):
    """Verifica que la búsqueda vectorial recupera contexto correctamente."""
    pages = [
        "Información sobre vacaciones: cada empleado recibe 20 días",
        "Información sobre salarios: se pagan mensualmente",
    ]
    
    doc = chunk_document(pages, source="test.pdf", parent_size=150, parent_overlap=15,
                         child_size=60, child_overlap=5)
    
    temp_parentstore.add_parents(doc.parents)
    temp_vectorstore.add_children(doc.children)
    
    # Buscar algo relacionado con vacaciones
    results = temp_vectorstore.query("vacaciones días", k=3)
    
    assert len(results) > 0, "Debería encontrar al menos un resultado"
    
    # Verificar estructura del resultado
    for result in results:
        assert "chunk_id" in result
        assert "text" in result
        assert "source" in result
        assert "pages" in result
        assert "parent_id" in result


def test_end_to_end_orchestrator_se_construye(temp_vectorstore, temp_parentstore):
    """Smoke test: el orquestador se instancia sin errores con almacenes reales.

    No invoca al LLM (eso requeriría clave OpenAI). Solo comprueba que la
    composición de almacenes + orquestador es válida.
    """
    orchestrator = Orchestrator(temp_vectorstore, temp_parentstore)
    assert orchestrator.vs is temp_vectorstore
    assert orchestrator.ps is temp_parentstore


def test_end_to_end_multiple_pdfs(temp_vectorstore, temp_parentstore):
    """Verifica que se pueden indexar múltiples PDFs en paralelo."""
    pdfs = [
        ("politicas.pdf", ["Política 1: Vacaciones", "Política 2: Salud"]),
        ("procedimientos.pdf", ["Procedimiento 1: Solicitud", "Procedimiento 2: Aprobación"]),
        ("normativa.pdf", ["Norma 1: Horarios", "Norma 2: Conducta"]),
    ]
    
    total_parents = 0
    total_children = 0
    
    for source, pages in pdfs:
        doc = chunk_document(pages, source=source, parent_size=150, parent_overlap=15,
                             child_size=60, child_overlap=5)
        
        temp_parentstore.add_parents(doc.parents)
        temp_vectorstore.add_children(doc.children)
        
        total_parents += len(doc.parents)
        total_children += len(doc.children)
    
    # Verificar que todos se guardaron
    assert temp_parentstore.count() == total_parents
    assert temp_vectorstore.count() == total_children
    
    # Verificar que se pueden recuperar por fuente
    sources = temp_vectorstore.sources()
    assert len(sources) == 3
    assert all(source in sources for source, _ in pdfs)


def test_end_to_end_page_tracking_in_chunks(temp_vectorstore, temp_parentstore):
    """Verifica que el seguimiento de páginas funciona en chunks."""
    pages = [
        "Página 1: Introducción",
        "Página 2: Detalles técnicos",
        "Página 3: Conclusiones",
    ]
    
    doc = chunk_document(pages, source="test.pdf", parent_size=150, parent_overlap=15,
                         child_size=70, child_overlap=5)
    
    # Verificar que cada chunk tiene páginas
    for chunk in doc.children:
        assert len(chunk.pages) > 0, f"Child {chunk.chunk_id} sin páginas"
        for page in chunk.pages:
            assert 1 <= page <= len(pages), f"Página {page} fuera de rango"
    
    for chunk in doc.parents:
        assert len(chunk.pages) > 0, f"Parent {chunk.chunk_id} sin páginas"
        for page in chunk.pages:
            assert 1 <= page <= len(pages), f"Página {page} fuera de rango"


def test_end_to_end_delete_and_reindex(temp_vectorstore, temp_parentstore):
    """Verifica que se puede eliminar y reindexar correctamente."""
    source1 = "doc1.pdf"
    source2 = "doc2.pdf"
    
    pages1 = ["Contenido del documento 1"]
    pages2 = ["Contenido del documento 2"]
    
    # Indexar ambos
    doc1 = chunk_document(pages1, source=source1, parent_size=150, parent_overlap=15,
                          child_size=60, child_overlap=5)
    doc2 = chunk_document(pages2, source=source2, parent_size=150, parent_overlap=15,
                          child_size=60, child_overlap=5)
    
    temp_parentstore.add_parents(doc1.parents)
    temp_vectorstore.add_children(doc1.children)
    temp_parentstore.add_parents(doc2.parents)
    temp_vectorstore.add_children(doc2.children)
    
    initial_count = temp_vectorstore.count()
    assert initial_count > 0
    
    # Eliminar doc1
    temp_vectorstore.delete_source(source1)
    temp_parentstore.delete_source(source1)
    
    # Verificar que se eliminó
    assert temp_vectorstore.count() < initial_count
    
    # Reindexar doc1 con contenido diferente
    doc1_v2 = chunk_document(["Contenido actualizado del documento 1"], source=source1,
                             parent_size=150, parent_overlap=15, child_size=60, child_overlap=5)
    
    temp_parentstore.add_parents(doc1_v2.parents)
    temp_vectorstore.add_children(doc1_v2.children)
    
    # Verificar que se reindexó
    sources = temp_vectorstore.sources()
    assert source1 in sources
    assert source2 in sources
