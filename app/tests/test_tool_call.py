"""Test de las tools MCP (mock determinista) y del parseo de resultados."""
from __future__ import annotations

import datetime as dt
import sys
import os

# Para tests locales, intenta cargar mcp_server si está disponible
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _root not in sys.path:
    sys.path.insert(0, _root)

import pytest

from app import mcp_client

# Intenta importar las tools locales; si no está disponible, se saltarán esos tests
try:
    from mcp_server.tools_logic import (
        ToolError,
        compute_fx_rate,
        compute_market_status,
    )
    HAS_TOOLS_LOGIC = True
except ImportError:
    HAS_TOOLS_LOGIC = False


# --------------------------------------------------------------------------
# Tool fx_rate (solo si mcp_server está disponible localmente)
# --------------------------------------------------------------------------
@pytest.mark.skipif(not HAS_TOOLS_LOGIC, reason="mcp_server.tools_logic no disponible")
def test_fx_rate_par_basico():
    result = compute_fx_rate("EUR", "USD")
    assert result["base"] == "EUR"
    assert result["quote"] == "USD"
    assert result["rate"] == 1.08
    assert result["as_of"].endswith("Z")  # ISO-8601


@pytest.mark.skipif(not HAS_TOOLS_LOGIC, reason="mcp_server.tools_logic no disponible")
def test_fx_rate_misma_divisa():
    assert compute_fx_rate("USD", "USD")["rate"] == 1.0


@pytest.mark.skipif(not HAS_TOOLS_LOGIC, reason="mcp_server.tools_logic no disponible")
def test_fx_rate_par_no_soportado():
    with pytest.raises(ToolError):
        compute_fx_rate("EUR", "XYZ")


@pytest.mark.skipif(not HAS_TOOLS_LOGIC, reason="mcp_server.tools_logic no disponible")
def test_fx_rate_argumento_vacio():
    with pytest.raises(ToolError):
        compute_fx_rate("", "USD")


# --------------------------------------------------------------------------
# Tool market_status (con `now` inyectado para que sea determinista)
# --------------------------------------------------------------------------
@pytest.mark.skipif(not HAS_TOOLS_LOGIC, reason="mcp_server.tools_logic no disponible")
def test_market_status_abierto_en_dia_laborable():
    # Martes 13/01/2026, 15:00 UTC -> NYSE abre 14:00-21:00 UTC.
    now = dt.datetime(2026, 1, 13, 15, 0, tzinfo=dt.timezone.utc)
    result = compute_market_status("NYSE", now=now)
    assert result["is_open"] is True
    assert result["market"] == "NYSE"


@pytest.mark.skipif(not HAS_TOOLS_LOGIC, reason="mcp_server.tools_logic no disponible")
def test_market_status_cerrado_en_fin_de_semana():
    # Sábado 17/01/2026, 15:00 UTC -> cerrado.
    now = dt.datetime(2026, 1, 17, 15, 0, tzinfo=dt.timezone.utc)
    assert compute_market_status("NYSE", now=now)["is_open"] is False


@pytest.mark.skipif(not HAS_TOOLS_LOGIC, reason="mcp_server.tools_logic no disponible")
def test_market_status_cerrado_fuera_de_horario():
    # Martes 13/01/2026, 06:00 UTC -> antes de la apertura.
    now = dt.datetime(2026, 1, 13, 6, 0, tzinfo=dt.timezone.utc)
    assert compute_market_status("NYSE", now=now)["is_open"] is False


@pytest.mark.skipif(not HAS_TOOLS_LOGIC, reason="mcp_server.tools_logic no disponible")
def test_market_status_mercado_no_soportado():
    with pytest.raises(ToolError):
        compute_market_status("MERCADO_INVENTADO")


# --------------------------------------------------------------------------
# Parseo de resultados MCP: manejo de respuestas correctas, de error y
# malformadas (requisito del enunciado).
# --------------------------------------------------------------------------
class _FakeText:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeCallResult:
    def __init__(self, content: list, is_error: bool = False) -> None:
        self.content = content
        self.isError = is_error


def test_parse_resultado_correcto():
    res = _FakeCallResult([_FakeText('{"rate": 1.08, "base": "EUR"}')])
    payload = mcp_client.parse_tool_payload(res)
    assert payload == {"rate": 1.08, "base": "EUR"}


def test_parse_resultado_de_error():
    res = _FakeCallResult([_FakeText("par de divisas no soportado")], is_error=True)
    with pytest.raises(ValueError):
        mcp_client.parse_tool_payload(res)


def test_parse_contenido_no_json_devuelve_texto():
    res = _FakeCallResult([_FakeText("respuesta en texto plano")])
    assert mcp_client.parse_tool_payload(res) == "respuesta en texto plano"


def test_parse_contenido_vacio_lanza_error():
    res = _FakeCallResult([])
    with pytest.raises(ValueError):
        mcp_client.parse_tool_payload(res)


# --------------------------------------------------------------------------
# Integración: descubrimiento de tools desde el servidor MCP
# --------------------------------------------------------------------------
def test_mcp_tools_discovered():
    """Verifica que se puedan descubrir las tools desde el servidor MCP."""
    # Este test espera que el servidor MCP esté corriendo en http://mcp-server:8000
    # En CI/local, ajustar según la URL del servidor.
    tools = mcp_client.discover_tools()
    tool_names = [t.name for t in tools]
    assert "fx_rate" in tool_names
    assert "market_status" in tool_names
