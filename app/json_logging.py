"""Sistema de logging estructurado en JSON.

Proporciona logging que produce JSON estructurado para facilitar análisis
en sistemas como ELK, CloudWatch, Datadog, etc.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from typing import Any


class JSONFormatter(logging.Formatter):
    """Formateador que produce logs en JSON estructurado.
    
    Cada log es un objeto JSON en una sola línea (JSON Lines format).
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Formatea un LogRecord como JSON.
        
        Incluye:
        - timestamp (ISO-8601)
        - level
        - logger name
        - message
        - extra fields (si las hay)
        - exception info (si aplica)
        """
        log_obj: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Agregar campos extra si existen
        if hasattr(record, "extra_fields"):
            log_obj.update(record.extra_fields)
        
        # Agregar información de excepción si aplica
        if record.exc_info and record.exc_info[0] is not None:
            log_obj["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info),
            }
        
        return json.dumps(log_obj, ensure_ascii=False)


def setup_json_logging(name: str, level: str = "INFO") -> logging.Logger:
    """Configura logging JSON estructurado para un módulo.
    
    Args:
        name: Nombre del logger (típicamente __name__)
        level: Nivel de logging ("DEBUG", "INFO", "WARNING", "ERROR")
        
    Returns:
        Logger configurado con formato JSON
        
    Example:
        log = setup_json_logging(__name__)
        log.info("App started", extra={"version": "1.0"})
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    # Remover handlers previos para evitar duplicados
    logger.handlers.clear()
    
    # Handler a stdout con formato JSON
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)
    
    # No propagar a root logger (evita duplicados)
    logger.propagate = False
    
    return logger


def log_with_fields(logger: logging.Logger, level: str, message: str, **fields: Any) -> None:
    """Registra un mensaje con campos adicionales.
    
    Los campos se incluyen como propiedades en el JSON.
    
    Args:
        logger: Logger instance
        level: Nivel ("INFO", "WARNING", "ERROR", "DEBUG")
        message: Mensaje principal
        **fields: Campos adicionales para el JSON
        
    Example:
        log_with_fields(log, "INFO", "Query processed",
                       query="test", duration_ms=125, chunks=3)
        # Genera JSON:
        # {"timestamp":"...", "level":"INFO", "message":"Query processed",
        #  "query":"test", "duration_ms":125, "chunks":3}
    """
    record = logger.makeRecord(
        logger.name,
        getattr(logging, level.upper(), logging.INFO),
        "(structured)",
        0,
        message,
        (),
        None
    )
    record.extra_fields = fields
    logger.handle(record)
