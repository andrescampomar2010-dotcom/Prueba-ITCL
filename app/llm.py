"""Cliente LLM y de embeddings (OpenAI).

Se aísla aquí toda la dependencia de OpenAI para poder simularla en tests.
El proveedor se configura por variables de entorno (ver config.py).
"""
from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from app.config import settings
from app.logging_utils import get_logger

log = get_logger("llm")

_client: OpenAI | None = None


def get_client() -> OpenAI:
    """Devuelve un cliente OpenAI reutilizable (singleton perezoso)."""
    global _client
    if _client is None:
        if not settings.openai_configured:
            raise RuntimeError(
                "OPENAI_API_KEY no está configurada. Copia .env.example a .env "
                "y añade tu clave."
            )
        kwargs: dict[str, Any] = {
            "api_key": settings.openai_api_key,
            "timeout": settings.request_timeout,
        }
        if settings.openai_base_url:
            kwargs["base_url"] = settings.openai_base_url
        _client = OpenAI(**kwargs)
    return _client


def chat(messages: list[dict[str, Any]], *, tools: list[dict] | None = None,
         tool_choice: str | None = None, temperature: float = 0.1) -> Any:
    """Llamada de chat genérica. Devuelve el objeto `message` de la respuesta."""
    kwargs: dict[str, Any] = {
        "model": settings.llm_model,
        "messages": messages,
        "temperature": temperature,
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice or "auto"
    response = get_client().chat.completions.create(**kwargs)
    return response.choices[0].message


def chat_text(messages: list[dict[str, Any]], *, temperature: float = 0.1) -> str:
    """Llamada de chat que devuelve solo el texto de la respuesta."""
    message = chat(messages, temperature=temperature)
    return (message.content or "").strip()


def chat_json(messages: list[dict[str, Any]], *, temperature: float = 0.0) -> dict[str, Any]:
    """Llamada de chat forzando salida JSON. Devuelve un dict ya parseado."""
    response = get_client().chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content or "{}"
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        log.warning("Respuesta JSON malformada del LLM: %s", raw[:200])
        return {}


# Tamaño de lote para embeddings: evita superar el límite de la API de OpenAI
# cuando un PDF genera muchos fragmentos.
_EMBED_BATCH_SIZE = 96


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Genera embeddings para una lista de textos, en lotes."""
    if not texts:
        return []
    client = get_client()
    embeddings: list[list[float]] = []
    for start in range(0, len(texts), _EMBED_BATCH_SIZE):
        batch = texts[start:start + _EMBED_BATCH_SIZE]
        response = client.embeddings.create(
            model=settings.embedding_model,
            input=batch,
        )
        embeddings.extend(item.embedding for item in response.data)
    return embeddings
