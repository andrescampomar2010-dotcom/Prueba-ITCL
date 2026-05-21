"""Lógica determinista de las herramientas (tools) del servidor MCP.

Estas funciones son puras y deterministas para que puedan probarse con
tests unitarios sin necesidad de internet ni de servicios externos.
El enunciado permite explícitamente un "mock determinista" para las tools.
"""
from __future__ import annotations

from datetime import datetime, timezone

# Tabla fija de tipos de cambio (mock determinista). Claves "BASE/QUOTE".
_FX_TABLE: dict[str, float] = {
    "EUR/USD": 1.08,
    "USD/EUR": 0.9259,
    "EUR/GBP": 0.85,
    "GBP/EUR": 1.1765,
    "USD/GBP": 0.7870,
    "GBP/USD": 1.2706,
    "EUR/JPY": 167.50,
    "JPY/EUR": 0.005970,
    "USD/JPY": 155.10,
    "JPY/USD": 0.006447,
    "EUR/CHF": 0.9700,
    "CHF/EUR": 1.0309,
    "EUR/MXN": 19.50,
    "MXN/EUR": 0.05128,
    "USD/MXN": 18.05,
    "MXN/USD": 0.05540,
}

# Horario de apertura (hora UTC) de cada mercado, de lunes a viernes.
# Formato: (apertura_utc, cierre_utc, nombre_completo). Mock determinista.
_MARKET_HOURS: dict[str, tuple[int, int, str]] = {
    "NYSE": (14, 21, "New York Stock Exchange"),
    "NASDAQ": (14, 21, "NASDAQ"),
    "LSE": (8, 16, "London Stock Exchange"),
    "BME": (8, 16, "Bolsa de Madrid"),
    "XETRA": (8, 16, "Deutsche Boerse XETRA"),
    "TSE": (0, 6, "Tokyo Stock Exchange"),
    "HKEX": (1, 8, "Hong Kong Exchange"),
}


class ToolError(ValueError):
    """Error de validación de entrada de una tool."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def compute_fx_rate(base: str, quote: str, *, now: datetime | None = None) -> dict:
    """Devuelve el tipo de cambio para el par base/quote.

    Lanza ToolError si falta algún argumento o el par no está soportado.
    """
    if not base or not quote:
        raise ToolError("Los campos 'base' y 'quote' son obligatorios.")
    base = str(base).strip().upper()
    quote = str(quote).strip().upper()
    now = now or _utc_now()

    if base == quote:
        rate = 1.0
    else:
        key = f"{base}/{quote}"
        if key not in _FX_TABLE:
            raise ToolError(
                f"Par de divisas no soportado: {key}. "
                f"Pares disponibles: {', '.join(sorted(_FX_TABLE))}."
            )
        rate = _FX_TABLE[key]

    return {
        "base": base,
        "quote": quote,
        "rate": rate,
        "as_of": _iso(now),
    }


def compute_market_status(market: str, *, now: datetime | None = None) -> dict:
    """Devuelve si un mercado financiero está abierto o cerrado.

    El estado se calcula de forma determinista a partir de la hora UTC y de
    un horario fijo de apertura por mercado (lunes a viernes).
    Lanza ToolError si falta el argumento o el mercado no está soportado.
    """
    if not market:
        raise ToolError("El campo 'market' es obligatorio.")
    market = str(market).strip().upper()
    now = now or _utc_now()

    if market not in _MARKET_HOURS:
        raise ToolError(
            f"Mercado no soportado: {market}. "
            f"Mercados disponibles: {', '.join(sorted(_MARKET_HOURS))}."
        )

    open_h, close_h, full_name = _MARKET_HOURS[market]
    is_weekday = now.weekday() < 5  # 0=lunes ... 4=viernes
    is_open = bool(is_weekday and open_h <= now.hour < close_h)

    return {
        "market": market,
        "name": full_name,
        "is_open": is_open,
        "session_hours_utc": f"{open_h:02d}:00-{close_h:02d}:00",
        "as_of": _iso(now),
    }


def supported_markets() -> list[str]:
    return sorted(_MARKET_HOURS)


def supported_fx_pairs() -> list[str]:
    return sorted(_FX_TABLE)
