"""Servidor MCP independiente que expone tools por SSE.

Cumple el requisito del enunciado: las tools NO están hardcodeadas en el
cliente; el cliente las descubre conectándose a este servidor.

Endpoints:
  - GET  /sse        -> canal SSE del protocolo MCP (descubrimiento + llamadas)
  - POST /messages/  -> canal de mensajes del protocolo MCP
  - GET  /health     -> healthcheck simple (usado por docker-compose)

Tools expuestas (mock determinista, ver tools_logic.py):
  - fx_rate(base, quote)        -> tipo de cambio de un par de divisas
  - market_status(market)      -> estado abierto/cerrado de un mercado
"""
from __future__ import annotations

import json
import logging
import os
import sys

import uvicorn
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import TextContent, Tool
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

# Permite ejecutar tanto `python mcp_server/server.py` como módulo.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mcp_server.tools_logic import (  # noqa: E402
    ToolError,
    compute_fx_rate,
    compute_market_status,
    supported_fx_pairs,
    supported_markets,
)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='{"ts":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}',
)
log = logging.getLogger("mcp-server")

SERVER_NAME = "rag-mcp-tools"
HOST = os.getenv("MCP_HOST", "0.0.0.0")
PORT = int(os.getenv("MCP_PORT", "8000"))

# --------------------------------------------------------------------------
# Definición de las tools (esquema declarado en el servidor, no en el cliente)
# --------------------------------------------------------------------------
TOOLS: list[Tool] = [
    Tool(
        name="fx_rate",
        description=(
            "Obtiene el tipo de cambio actual para un par de divisas base/quote "
            "(por ejemplo EUR a USD). Úsala para convertir importes entre monedas."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "base": {
                    "type": "string",
                    "description": "Código ISO de la divisa de origen, p. ej. 'EUR'.",
                },
                "quote": {
                    "type": "string",
                    "description": "Código ISO de la divisa de destino, p. ej. 'USD'.",
                },
            },
            "required": ["base", "quote"],
        },
    ),
    Tool(
        name="market_status",
        description=(
            "Consulta si un mercado financiero está abierto o cerrado en este "
            "momento. Úsala para preguntas sobre el estado actual de un mercado."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "market": {
                    "type": "string",
                    "description": (
                        "Identificador del mercado, p. ej. 'NYSE', 'NASDAQ', "
                        "'LSE', 'BME', 'XETRA', 'TSE', 'HKEX'."
                    ),
                },
            },
            "required": ["market"],
        },
    ),
]

server: Server = Server(SERVER_NAME)


@server.list_tools()
async def list_tools() -> list[Tool]:
    """Devuelve el catálogo de tools para que el cliente las descubra."""
    log.info("list_tools solicitado (%d tools)", len(TOOLS))
    return TOOLS


@server.call_tool()
async def call_tool(name: str, arguments: dict | None) -> list[TextContent]:
    """Ejecuta una tool y devuelve el resultado serializado como JSON.

    Si la tool falla, se lanza la excepción: el SDK la transforma en una
    respuesta de error MCP (isError=True), que el cliente sabe interpretar.
    """
    arguments = arguments or {}
    log.info("call_tool name=%s args=%s", name, json.dumps(arguments, ensure_ascii=False))

    if name == "fx_rate":
        result = compute_fx_rate(
            base=arguments.get("base", ""),
            quote=arguments.get("quote", ""),
        )
    elif name == "market_status":
        result = compute_market_status(market=arguments.get("market", ""))
    else:
        raise ToolError(f"Tool desconocida: {name}")

    return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]


# --------------------------------------------------------------------------
# Transporte SSE + app Starlette
# --------------------------------------------------------------------------
sse_transport = SseServerTransport("/messages/")


async def handle_sse(request: Request):
    """Maneja la conexión SSE entrante del cliente MCP."""
    async with sse_transport.connect_sse(
        request.scope, request.receive, request._send
    ) as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


async def health(request: Request) -> JSONResponse:
    """Healthcheck. Usado por docker-compose para el `depends_on`."""
    return JSONResponse(
        {
            "status": "ok",
            "service": SERVER_NAME,
            "tools": [t.name for t in TOOLS],
            "fx_pairs": supported_fx_pairs(),
            "markets": supported_markets(),
        }
    )


app = Starlette(
    debug=False,
    routes=[
        Route("/health", endpoint=health, methods=["GET"]),
        Route("/sse", endpoint=handle_sse, methods=["GET"]),
        Mount("/messages/", app=sse_transport.handle_post_message),
    ],
)


if __name__ == "__main__":
    log.info("Arrancando servidor MCP en http://%s:%d (SSE en /sse)", HOST, PORT)
    uvicorn.run(app, host=HOST, port=PORT, log_level=os.getenv("LOG_LEVEL", "info").lower())
