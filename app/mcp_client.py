"""Cliente MCP sobre SSE.

Se conecta al servidor MCP independiente, DESCUBRE las tools disponibles
(no están hardcodeadas en el cliente) y permite invocarlas.

Robustez (requisito del enunciado):
  - timeouts razonables en conexión y en cada llamada,
  - reintentos con backoff exponencial,
  - manejo de errores típicos: servidor caído, timeout, respuesta malformada.

Como Streamlit es síncrono, las corrutinas se ejecutan en un hilo propio con
su event loop, lo que evita conflictos con loops existentes.
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from mcp import ClientSession
from mcp.client.sse import sse_client

from app.config import settings
from app.logging_utils import get_logger, log_event

log = get_logger("mcp_client")

# Cache expiration (1 hour)
_TOOL_CACHE_TTL = 3600


@dataclass
class ToolSpec:
    """Especificación de una herramienta MCP.
    
    Attributes:
        name: Identificador único de la herramienta
        description: Descripción legible de qué hace
        input_schema: JSON Schema con los parámetros esperados
    """
    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)

    def to_openai_tool(self) -> dict[str, Any]:
        """Convierte la tool al formato de function-calling de OpenAI.
        
        Returns:
            Diccionario en formato OpenAI Function Call
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema or {"type": "object", "properties": {}},
            },
        }


@dataclass
class ToolResult:
    """Resultado de ejecutar una herramienta MCP.
    
    Attributes:
        name: Nombre de la herramienta ejecutada
        ok: True si ejecutó correctamente
        result: Resultado si ok=True, None en caso contrario
        error: Mensaje de error si ok=False
        duration_ms: Tiempo total de ejecución en milisegundos
    """
    name: str
    ok: bool
    result: Any = None
    error: str = ""
    duration_ms: float = 0.0


def _run(coro: asyncio.coroutine) -> Any:
    """Ejecuta una corrutina en un hilo con su propio event loop.
    
    Necesario porque Streamlit es síncrono pero MCP client es async.
    
    Args:
        coro: Corrutina para ejecutar
        
    Returns:
        Resultado de la corrutina
        
    Raises:
        Exception: Si la corrutina falla
    """
    box: dict[str, Any] = {}

    def runner() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            box["value"] = loop.run_until_complete(coro)
        except Exception as exc:  # noqa: BLE001
            box["error"] = exc
        finally:
            loop.close()

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()
    if "error" in box:
        raise box["error"]
    return box["value"]


def _sse_url() -> str:
    """Construye la URL del endpoint SSE del servidor MCP."""
    return settings.mcp_server_url.rstrip("/") + "/sse"


async def _adiscover() -> list[ToolSpec]:
    """Descubre las herramientas disponibles en el servidor MCP (async)."""
    async with sse_client(_sse_url(), timeout=settings.tool_timeout) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            response = await session.list_tools()
            return [
                ToolSpec(
                    name=t.name,
                    description=t.description or "",
                    input_schema=getattr(t, "inputSchema", None) or {},
                )
                for t in response.tools
            ]


async def _acall(name: str, arguments: dict[str, Any]) -> Any:
    """Ejecuta una herramienta en el servidor MCP (async)."""
    async with sse_client(_sse_url(), timeout=settings.tool_timeout) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await session.call_tool(name, arguments)


def parse_tool_payload(call_result: Any) -> Any:
    """Extrae el resultado de un CallToolResult del SDK MCP.

    Lanza ValueError si la tool devolvió error o si el contenido está
    malformado (requisito: manejar respuestas malformadas).
    """
    if getattr(call_result, "isError", False):
        text = _first_text(call_result)
        raise ValueError(text or "La tool devolvió un error sin detalle.")

    text = _first_text(call_result)
    if text is None:
        raise ValueError("La tool no devolvió contenido interpretable.")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # No todo resultado tiene que ser JSON; devolvemos el texto plano.
        return text


def _first_text(call_result: Any) -> str | None:
    """Extrae el primer campo de texto de un resultado MCP."""
    content = getattr(call_result, "content", None) or []
    for block in content:
        text = getattr(block, "text", None)
        if text is not None:
            return text
    return None


# --------------------------------------------------------------------------
# Cache de tools con TTL
# --------------------------------------------------------------------------
_tools_cache: list[ToolSpec] | None = None
_tool_cache_time: float = 0.0


def _is_cache_valid() -> bool:
    """Verifica si el cache de tools sigue siendo válido."""
    return (time.time() - _tool_cache_time) < _TOOL_CACHE_TTL


# --------------------------------------------------------------------------
# API pública (síncrona)
# --------------------------------------------------------------------------

def discover_tools(*, force: bool = False) -> list[ToolSpec]:
    """Descubre las tools del servidor MCP con TTL cache.
    
    El resultado se cachea por 1 hora para evitar llamadas frecuentes.
    
    Args:
        force: Si True, ignora el cache y redescubre las tools
        
    Returns:
        Lista de herramientas disponibles en el servidor MCP
        
    Example:
        tools = discover_tools()
        tool_names = [t.name for t in tools]
        
    Note:
        Si el servidor no está disponible, devuelve el ultimo resultado
        en cache o lista vacía si nunca se conectó.
    """
    global _tools_cache, _tool_cache_time
    if _tools_cache is not None and not force and _is_cache_valid():
        return _tools_cache
    try:
        tools = _run(_adiscover())
        _tools_cache = tools
        _tool_cache_time = time.time()
        log_event(log, "Tools MCP descubiertas",
                  tools=[t.name for t in tools], url=_sse_url(), ttl_seconds=_TOOL_CACHE_TTL)
        return tools
    except Exception as exc:  # noqa: BLE001
        log.warning("No se pudieron descubrir tools MCP: %s", exc)
        return _tools_cache or []


def call_tool(name: str, arguments: dict[str, Any]) -> ToolResult:
    """Invoca una tool MCP con timeout y reintentos automáticos.
    
    Implementa robustez con:
    - Timeout configurable por llamada
    - Reintentos con backoff exponencial
    - Manejo de errores de conexión
    - Logging detallado de cada intento
    
    Args:
        name: Nombre de la herramienta a invocar
        arguments: Argumentos en formato dict (se pasan como JSON)
        
    Returns:
        ToolResult con resultado o error detallado
        
    Example:
        result = call_tool("fx_rate", {"base": "EUR", "quote": "USD"})
        if result.ok:
            print(f"Rate: {result.result['rate']}")
        else:
            print(f"Error: {result.error}")
    """
    attempts = settings.tool_max_retries + 1
    last_error = ""
    for attempt in range(1, attempts + 1):
        started = time.time()
        try:
            raw = _run(asyncio.wait_for(_acall(name, arguments),
                                        timeout=settings.tool_timeout))
            payload = parse_tool_payload(raw)
            duration = (time.time() - started) * 1000
            log_event(log, "Tool MCP ejecutada", tool=name, args=arguments,
                      status="ok", duration_ms=round(duration, 1), attempt=attempt)
            return ToolResult(name=name, ok=True, result=payload, duration_ms=duration)
        except asyncio.TimeoutError:
            last_error = f"Timeout tras {settings.tool_timeout}s"
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)

        duration = (time.time() - started) * 1000
        log_event(log, "Fallo en tool MCP", tool=name, args=arguments,
                  status="error", error=last_error, attempt=attempt,
                  duration_ms=round(duration, 1))
        if attempt < attempts:
            time.sleep(min(0.5 * (2 ** (attempt - 1)), 4.0))  # backoff exponencial

    return ToolResult(name=name, ok=False, error=last_error)


def server_health() -> dict[str, Any]:
    """Consulta el endpoint /health del servidor MCP.
    
    Returns:
        {"ok": True, "data": {...}} si el servidor responde
        {"ok": False, "error": "..."} si hay error de conexión
    """
    import urllib.request

    url = settings.mcp_server_url.rstrip("/") + "/health"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return {"ok": True, "data": json.loads(resp.read().decode("utf-8"))}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}
