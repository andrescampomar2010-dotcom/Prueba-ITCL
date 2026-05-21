"""Monitoreo de performance del sistema.

Proporciona métricas sobre:
- Tiempo de respuesta de queries
- Coste estimado de API
- Uso de vectorstore
- Estadísticas de herramientas MCP
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class QueryMetrics:
    """Métricas de una consulta individual.
    
    Attributes:
        query: Texto de la consulta
        route: Ruta elegida (RAG/TOOL/BOTH/NONE)
        duration_ms: Tiempo total en milisegundos
        embeddings_used: Número de embeddings generados
        chunks_retrieved: Número de chunks recuperados
        tool_calls: Número de herramientas invocadas
        estimated_cost_cents: Coste estimado en centavos
        timestamp: Unix timestamp de la consulta
    """
    query: str
    route: str
    duration_ms: float
    embeddings_used: int = 0
    chunks_retrieved: int = 0
    tool_calls: int = 0
    estimated_cost_cents: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class SystemMetrics:
    """Estadísticas globales del sistema.
    
    Attributes:
        total_queries: Número total de consultas procesadas
        total_duration_ms: Tiempo total de todas las consultas
        avg_duration_ms: Tiempo promedio por consulta
        total_estimated_cost_cents: Coste total estimado
        queries_by_route: Conteo por ruta elegida
        most_common_route: Ruta más utilizada
        last_query_duration_ms: Duración de la última consulta
        uptime_seconds: Tiempo desde el inicio del sistema
    """
    total_queries: int = 0
    total_duration_ms: float = 0.0
    total_estimated_cost_cents: float = 0.0
    queries_by_route: dict[str, int] = field(default_factory=lambda: {
        "RAG": 0, "TOOL": 0, "BOTH": 0, "NONE": 0
    })
    embeddings_count: int = 0
    chunks_retrieved_count: int = 0
    tool_calls_count: int = 0
    last_query_duration_ms: float = 0.0
    startup_time: float = field(default_factory=time.time)
    
    @property
    def avg_duration_ms(self) -> float:
        """Tiempo promedio por consulta."""
        if self.total_queries == 0:
            return 0.0
        return self.total_duration_ms / self.total_queries
    
    @property
    def most_common_route(self) -> str:
        """Ruta más utilizada."""
        return max(self.queries_by_route.items(), key=lambda x: x[1])[0]
    
    @property
    def uptime_seconds(self) -> float:
        """Tiempo desde el inicio del sistema en segundos."""
        return time.time() - self.startup_time
    
    def to_dict(self) -> dict[str, Any]:
        """Convierte a diccionario para serialización JSON."""
        return {
            "total_queries": self.total_queries,
            "total_duration_ms": round(self.total_duration_ms, 1),
            "avg_duration_ms": round(self.avg_duration_ms, 1),
            "last_duration_ms": round(self.last_query_duration_ms, 1),
            "total_cost_cents": round(self.total_estimated_cost_cents, 2),
            "queries_by_route": self.queries_by_route,
            "most_common_route": self.most_common_route,
            "embeddings_count": self.embeddings_count,
            "chunks_retrieved": self.chunks_retrieved_count,
            "tool_calls": self.tool_calls_count,
            "uptime_seconds": round(self.uptime_seconds, 1),
        }


class MetricsCollector:
    """Recopila y agrega métricas del sistema."""
    
    def __init__(self):
        self.metrics = SystemMetrics()
        self.query_history: list[QueryMetrics] = []
        self._max_history = 100  # Guardar últimas 100 queries
    
    def record_query(self, query_metric: QueryMetrics) -> None:
        """Registra una consulta completada.
        
        Args:
            query_metric: Métricas de la consulta
        """
        self.metrics.total_queries += 1
        self.metrics.total_duration_ms += query_metric.duration_ms
        self.metrics.total_estimated_cost_cents += query_metric.estimated_cost_cents
        self.metrics.queries_by_route[query_metric.route] += 1
        self.metrics.last_query_duration_ms = query_metric.duration_ms
        self.metrics.embeddings_count += query_metric.embeddings_used
        self.metrics.chunks_retrieved_count += query_metric.chunks_retrieved
        self.metrics.tool_calls_count += query_metric.tool_calls
        
        # Guardar en historial (circular)
        self.query_history.append(query_metric)
        if len(self.query_history) > self._max_history:
            self.query_history.pop(0)
    
    def get_metrics(self) -> SystemMetrics:
        """Devuelve las métricas globales actuales."""
        return self.metrics
    
    def get_query_history(self, limit: int = 20) -> list[dict[str, Any]]:
        """Devuelve el historial de consultas más recientes.
        
        Args:
            limit: Número máximo de consultas a retornar
            
        Returns:
            Lista de últimas N consultas como dicts
        """
        recent = self.query_history[-limit:]
        return [
            {
                "query": q.query,
                "route": q.route,
                "duration_ms": round(q.duration_ms, 1),
                "embeddings": q.embeddings_used,
                "chunks": q.chunks_retrieved,
                "tools": q.tool_calls,
                "cost_cents": round(q.estimated_cost_cents, 2),
            }
            for q in reversed(recent)
        ]
    
    def reset(self) -> None:
        """Reinicia todas las métricas."""
        self.metrics = SystemMetrics()
        self.query_history.clear()


# Instancia global de colector de métricas
_metrics_collector = MetricsCollector()


def record_query(query_metric: QueryMetrics) -> None:
    """Función global para registrar una consulta."""
    _metrics_collector.record_query(query_metric)


def get_system_metrics() -> dict[str, Any]:
    """Función global para obtener métricas del sistema."""
    return _metrics_collector.get_metrics().to_dict()


def get_query_history(limit: int = 20) -> list[dict[str, Any]]:
    """Función global para obtener historial de consultas."""
    return _metrics_collector.get_query_history(limit)
