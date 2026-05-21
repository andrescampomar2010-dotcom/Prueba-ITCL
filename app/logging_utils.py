"""Logging estructurado en JSON y objeto de traza por petición.

El enunciado pide trazabilidad: logs claros (recomendado JSON) con la query,
la ruta elegida (RAG/TOOL/BOTH/NONE), los chunks recuperados y las tool calls.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any

from app.config import settings

_CONFIGURED = False


class JsonFormatter(logging.Formatter):
    """Formatea cada registro de log como una línea JSON."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Campos estructurados añadidos vía `extra={"event": {...}}`.
        if hasattr(record, "event"):
            payload["event"] = record.event  # type: ignore[attr-defined]
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging() -> None:
    """Configura logging JSON a stdout y a un fichero rotativo en data/logs."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    root = logging.getLogger()
    root.setLevel(settings.log_level.upper())
    for handler in list(root.handlers):
        root.removeHandler(handler)

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(JsonFormatter())
    root.addHandler(stream)

    try:
        os.makedirs(settings.log_dir, exist_ok=True)
        file_handler = logging.FileHandler(
            os.path.join(settings.log_dir, "app.log"), encoding="utf-8"
        )
        file_handler.setFormatter(JsonFormatter())
        root.addHandler(file_handler)
    except OSError:
        # Si no se puede escribir el fichero, seguimos solo con stdout.
        pass

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)


def log_event(logger: logging.Logger, msg: str, **event: Any) -> None:
    """Emite un log con un objeto `event` estructurado."""
    logger.info(msg, extra={"event": event})


@dataclass
class Trace:
    """Acumula la traza de una petición para mostrarla en el modo depuración."""

    query: str = ""
    route: str = ""
    route_reason: str = ""
    retrieved_chunks: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)

    def set_route(self, route: str, reason: str) -> None:
        self.route = route
        self.route_reason = reason

    def add_chunk(self, chunk_id: str, page: Any, source: str, snippet: str,
                  distance: float | None = None) -> None:
        self.retrieved_chunks.append({
            "chunk_id": chunk_id,
            "page": page,
            "source": source,
            "snippet": snippet[:240],
            "distance": round(distance, 4) if distance is not None else None,
        })

    def add_tool_call(self, name: str, args: dict[str, Any], status: str,
                      duration_ms: float, result: Any = None, error: str = "") -> None:
        self.tool_calls.append({
            "name": name,
            "args": args,
            "status": status,
            "duration_ms": round(duration_ms, 1),
            "result": result,
            "error": error,
        })

    def note(self, text: str) -> None:
        self.notes.append(text)

    @property
    def elapsed_ms(self) -> float:
        return round((time.time() - self.started_at) * 1000, 1)

    def as_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "route": self.route,
            "route_reason": self.route_reason,
            "retrieved_chunks": self.retrieved_chunks,
            "tool_calls": self.tool_calls,
            "notes": self.notes,
            "elapsed_ms": self.elapsed_ms,
        }
