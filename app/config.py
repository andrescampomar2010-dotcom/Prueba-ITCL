"""Configuración central de la aplicación.

Todo lo sensible o dependiente del entorno (claves, modelos, URLs) se lee de
variables de entorno. No se hardcodea ningún secreto.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    return _env(name, str(default)).strip().lower() in ("1", "true", "yes", "on", "si", "sí")


@dataclass
class Settings:
    # --- OpenAI ---
    openai_api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY"))
    openai_base_url: str = field(default_factory=lambda: _env("OPENAI_BASE_URL"))
    llm_model: str = field(default_factory=lambda: _env("OPENAI_MODEL", "gpt-4o-mini"))
    embedding_model: str = field(
        default_factory=lambda: _env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    )
    request_timeout: float = field(default_factory=lambda: _env_float("OPENAI_TIMEOUT", 30.0))

    # --- Servidor MCP ---
    mcp_server_url: str = field(
        default_factory=lambda: _env("MCP_SERVER_URL", "http://localhost:8000")
    )
    tool_timeout: float = field(default_factory=lambda: _env_float("MCP_TOOL_TIMEOUT", 12.0))
    tool_max_retries: int = field(default_factory=lambda: _env_int("MCP_TOOL_RETRIES", 2))

    # --- Rutas de datos ---
    data_dir: str = field(default_factory=lambda: _env("DATA_DIR", "./data"))
    seed_pdf_dir: str = field(default_factory=lambda: _env("SEED_PDF_DIR", "./seed_pdfs"))

    # --- Chunking jerárquico ---
    parent_chunk_size: int = field(default_factory=lambda: _env_int("PARENT_CHUNK_SIZE", 1800))
    parent_chunk_overlap: int = field(default_factory=lambda: _env_int("PARENT_CHUNK_OVERLAP", 200))
    child_chunk_size: int = field(default_factory=lambda: _env_int("CHILD_CHUNK_SIZE", 450))
    child_chunk_overlap: int = field(default_factory=lambda: _env_int("CHILD_CHUNK_OVERLAP", 80))

    # --- Recuperación ---
    top_k_children: int = field(default_factory=lambda: _env_int("TOP_K_CHILDREN", 6))
    max_parents: int = field(default_factory=lambda: _env_int("MAX_PARENTS", 3))

    # --- Observabilidad ---
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))

    # --- Rutas derivadas ---
    @property
    def pdf_dir(self) -> str:
        return os.path.join(self.data_dir, "pdfs")

    @property
    def chroma_dir(self) -> str:
        return os.path.join(self.data_dir, "chroma")

    @property
    def parent_db_path(self) -> str:
        return os.path.join(self.data_dir, "chroma", "parents.sqlite")

    @property
    def log_dir(self) -> str:
        return os.path.join(self.data_dir, "logs")

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.pdf_dir, self.chroma_dir, self.log_dir):
            os.makedirs(path, exist_ok=True)

    @property
    def openai_configured(self) -> bool:
        return bool(self.openai_api_key)


settings = Settings()
