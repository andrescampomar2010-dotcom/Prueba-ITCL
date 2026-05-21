"""Configuración compartida de pytest: rutas, fixtures y dobles de prueba."""
from __future__ import annotations

import hashlib
import os
import sys

import pytest

# Asegura que la raíz del proyecto está en sys.path para importar `app`,
# `mcp_server` y `scripts` al ejecutar pytest.
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts.generate_sample_pdf import build_pdf  # noqa: E402


@pytest.fixture(scope="session")
def sample_pdf(tmp_path_factory) -> str:
    """Genera el PDF de ejemplo en un directorio temporal una sola vez."""
    out = tmp_path_factory.mktemp("pdfs") / "manual_ejemplo.pdf"
    return build_pdf(str(out))


def _deterministic_vector(text: str, dim: int = 64) -> list[float]:
    """Embedding determinista basado en hash (sin red, para tests)."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [digest[i % len(digest)] / 255.0 for i in range(dim)]


@pytest.fixture
def fake_embedder():
    """Embedder determinista que sustituye a OpenAI en los tests."""
    def embed(texts: list[str]) -> list[list[float]]:
        return [_deterministic_vector(t) for t in texts]
    return embed
