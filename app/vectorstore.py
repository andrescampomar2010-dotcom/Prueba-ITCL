"""Almacén vectorial (nivel 1) y almacén de padres (nivel 2).

- VectorStore: colección ChromaDB persistente con los chunks HIJOS y sus
  embeddings. Es el índice de recuperación por similitud.
- ParentStore: pequeña base SQLite con los chunks PADRE, recuperables por id.
  Los padres no necesitan embedding: solo se buscan por clave tras recuperar
  a sus hijos (reconstrucción del contexto amplio).

El callable de embeddings se inyecta, de modo que los tests pueden usar uno
determinista sin llamar a OpenAI.
"""
from __future__ import annotations

import os
import sqlite3
from typing import Any, Callable

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.chunking import Chunk
from app.config import settings
from app.logging_utils import get_logger

log = get_logger("vectorstore")

Embedder = Callable[[list[str]], list[list[float]]]
_CHILDREN_COLLECTION = "rag_children"


def _pages_to_str(pages: list[int]) -> str:
    return ",".join(str(p) for p in pages)


def _pages_from_str(value: str) -> list[int]:
    return [int(p) for p in value.split(",") if p.strip().isdigit()]


class VectorStore:
    """Índice vectorial de chunks hijos sobre ChromaDB."""

    def __init__(self, embedder: Embedder, persist_dir: str | None = None) -> None:
        self.embedder = embedder
        self.persist_dir = persist_dir or settings.chroma_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        self._collection = self._client.get_or_create_collection(
            name=_CHILDREN_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    def add_children(self, children: list[Chunk]) -> int:
        """Indexa una lista de chunks hijos. Devuelve cuántos se añadieron."""
        if not children:
            return 0
        embeddings = self.embedder([c.text for c in children])
        self._collection.add(
            ids=[c.chunk_id for c in children],
            embeddings=embeddings,
            documents=[c.text for c in children],
            metadatas=[
                {
                    "source": c.source,
                    "page": c.page,
                    "pages": _pages_to_str(c.pages),
                    "parent_id": c.parent_id or "",
                    "char_start": c.char_start,
                    "char_end": c.char_end,
                    "child_index": c.index,
                }
                for c in children
            ],
        )
        return len(children)

    def query(self, text: str, k: int = 6) -> list[dict[str, Any]]:
        """Recupera los k chunks hijos más similares a `text`."""
        if self.count() == 0:
            return []
        embedding = self.embedder([text])[0]
        res = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(k, self.count()),
            include=["documents", "metadatas", "distances"],
        )
        hits: list[dict[str, Any]] = []
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for i, chunk_id in enumerate(ids):
            meta = metas[i] or {}
            hits.append({
                "chunk_id": chunk_id,
                "text": docs[i],
                "source": meta.get("source", ""),
                "page": meta.get("page"),
                "pages": _pages_from_str(meta.get("pages", "")),
                "parent_id": meta.get("parent_id", ""),
                "distance": dists[i] if i < len(dists) else None,
            })
        return hits

    def count(self) -> int:
        return self._collection.count()

    def sources(self) -> list[str]:
        """Lista las fuentes (nombres de PDF) actualmente indexadas."""
        if self.count() == 0:
            return []
        data = self._collection.get(include=["metadatas"])
        found = {(m or {}).get("source", "") for m in data.get("metadatas", [])}
        return sorted(s for s in found if s)

    def delete_source(self, source: str) -> None:
        self._collection.delete(where={"source": source})

    def reset(self) -> None:
        """Vacía por completo la colección de hijos."""
        try:
            self._client.delete_collection(_CHILDREN_COLLECTION)
        except Exception:  # noqa: BLE001 - la colección puede no existir
            pass
        self._collection = self._client.get_or_create_collection(
            name=_CHILDREN_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )


class ParentStore:
    """Almacén de chunks padre en SQLite, recuperables por id."""

    def __init__(self, db_path: str | None = None) -> None:
        self.db_path = db_path or settings.parent_db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS parents (
                    id          TEXT PRIMARY KEY,
                    source      TEXT NOT NULL,
                    text        TEXT NOT NULL,
                    pages       TEXT NOT NULL,
                    char_start  INTEGER NOT NULL,
                    char_end    INTEGER NOT NULL
                )
                """
            )

    def add_parents(self, parents: list[Chunk]) -> int:
        if not parents:
            return 0
        with self._connect() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO parents VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (p.chunk_id, p.source, p.text, _pages_to_str(p.pages),
                     p.char_start, p.char_end)
                    for p in parents
                ],
            )
        return len(parents)

    def get(self, ids: list[str]) -> list[dict[str, Any]]:
        """Devuelve los padres pedidos, preservando el orden de `ids`."""
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM parents WHERE id IN ({placeholders})", ids
            ).fetchall()
        by_id = {
            row["id"]: {
                "parent_id": row["id"],
                "source": row["source"],
                "text": row["text"],
                "pages": _pages_from_str(row["pages"]),
                "char_start": row["char_start"],
                "char_end": row["char_end"],
            }
            for row in rows
        }
        return [by_id[i] for i in ids if i in by_id]

    def delete_source(self, source: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM parents WHERE source = ?", (source,))

    def reset(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM parents")

    def count(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM parents").fetchone()[0]
