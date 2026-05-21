"""Orquestador: une enrutado, RAG jerárquico y tools MCP.

Para cada pregunta:
  1. Descubre las tools del MCP (cacheado) — nunca hardcodeadas.
  2. El router decide la ruta: RAG / TOOL / BOTH / NONE.
  3. Ejecuta la rama correspondiente y construye una respuesta trazable.

Para TOOL y BOTH se usa function-calling nativo de OpenAI: el modelo decide
qué tool llamar y con qué argumentos, y el orquestador las ejecuta vía MCP.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from app import llm, mcp_client, rag, router
from app.config import settings
from app.logging_utils import Trace, get_logger, log_event
from app.vectorstore import ParentStore, VectorStore

log = get_logger("orchestrator")

MAX_TOOL_ITERATIONS = 4

NONE_MESSAGE = (
    "No puedo responder a esa pregunta: no aparece en la documentación PDF "
    "indexada y tampoco hay ninguna herramienta disponible que la cubra."
)

TOOL_SYSTEM = """Eres un asistente que responde SIEMPRE en español. Dispones de
herramientas externas para obtener datos en vivo (tipos de cambio, estado de
mercados). Usa las herramientas cuando hagan falta y básate en sus resultados.
Si una herramienta falla, explícalo con claridad. No inventes datos."""

BOTH_SYSTEM = """Eres un asistente que responde SIEMPRE en español. Tienes dos
fuentes:
1) CONTEXTO de documentación interna en PDF (úsalo para datos del documento y
   CITA la página entre paréntesis, p. ej. (pág. 3)).
2) Herramientas externas para datos en vivo (tipos de cambio, mercados).
Combina ambas cuando sea necesario (p. ej. toma una cifra del PDF y conviértela
con la herramienta de tipo de cambio). Muestra el cálculo. No inventes datos;
si algo no está disponible, dilo."""


@dataclass
class Answer:
    text: str
    route: str
    route_reason: str
    citations: list[rag.Citation] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    trace: dict[str, Any] = field(default_factory=dict)


class Orchestrator:
    """Coordina router, RAG y tools para responder una pregunta."""

    def __init__(self, vector_store: VectorStore, parent_store: ParentStore) -> None:
        self.vs = vector_store
        self.ps = parent_store

    # -- API principal -----------------------------------------------------
    def answer(self, query: str, *, history: list[dict[str, str]] | None = None,
               debug: bool = False) -> Answer:
        trace = Trace(query=query)
        history = history or []

        tools = mcp_client.discover_tools()
        if not tools:
            trace.note("Servidor MCP no disponible o sin tools: solo RAG/NONE.")

        decision = router.classify(query, tools)
        trace.set_route(decision.route, decision.reason)
        log_event(log, "Ruta decidida", query=query, route=decision.route,
                  reason=decision.reason, via=decision.via, tools=[t.name for t in tools])

        if decision.route == "NONE":
            answer = self._handle_none(trace)
        elif decision.route == "RAG":
            answer = self._handle_rag(query, history, trace)
        elif decision.route == "TOOL":
            answer = self._handle_tool(query, history, tools, trace)
        else:  # BOTH
            answer = self._handle_both(query, history, tools, trace)

        log_event(log, "Respuesta generada", route=answer.route,
                  n_citas=len(answer.citations), n_tool_calls=len(answer.tool_calls),
                  elapsed_ms=trace.elapsed_ms)
        answer.trace = trace.as_dict()
        return answer

    # -- Ramas -------------------------------------------------------------
    def _handle_none(self, trace: Trace) -> Answer:
        return Answer(text=NONE_MESSAGE, route="NONE", route_reason=trace.route_reason)

    def _handle_rag(self, query: str, history: list[dict[str, str]],
                    trace: Trace) -> Answer:
        text, retrieval = rag.answer_with_rag(
            query, self.vs, self.ps, history=history, trace=trace)
        return Answer(
            text=text, route="RAG", route_reason=trace.route_reason,
            citations=retrieval.citations,
        )

    def _handle_tool(self, query: str, history: list[dict[str, str]],
                     tools: list[mcp_client.ToolSpec], trace: Trace) -> Answer:
        messages: list[dict[str, Any]] = [{"role": "system", "content": TOOL_SYSTEM}]
        messages += history[-4:]
        messages.append({"role": "user", "content": query})
        text, tool_calls = self._tool_loop(messages, tools, trace)
        return Answer(text=text, route="TOOL", route_reason=trace.route_reason,
                      tool_calls=tool_calls)

    def _handle_both(self, query: str, history: list[dict[str, str]],
                     tools: list[mcp_client.ToolSpec], trace: Trace) -> Answer:
        retrieval = rag.retrieve(query, self.vs, self.ps, trace=trace)
        context_text = rag.build_context_text(retrieval.context_blocks) or \
            "(no se ha encontrado contexto en el PDF)"
        messages: list[dict[str, Any]] = [{"role": "system", "content": BOTH_SYSTEM}]
        messages += history[-4:]
        messages.append({
            "role": "user",
            "content": f"CONTEXTO (documentación PDF):\n{context_text}\n\nPREGUNTA: {query}",
        })
        text, tool_calls = self._tool_loop(messages, tools, trace)
        return Answer(text=text, route="BOTH", route_reason=trace.route_reason,
                      citations=retrieval.citations, tool_calls=tool_calls)

    # -- Bucle de function-calling ----------------------------------------
    def _tool_loop(self, messages: list[dict[str, Any]],
                   tools: list[mcp_client.ToolSpec],
                   trace: Trace) -> tuple[str, list[dict[str, Any]]]:
        """Ejecuta el bucle de llamadas a tools del LLM contra el MCP."""
        openai_tools = [t.to_openai_tool() for t in tools]
        executed: list[dict[str, Any]] = []

        for iteration in range(MAX_TOOL_ITERATIONS):
            message = llm.chat(messages, tools=openai_tools, tool_choice="auto")
            tool_calls = getattr(message, "tool_calls", None)

            if not tool_calls:
                return (message.content or "").strip(), executed

            # Reinsertamos el mensaje del asistente con sus tool_calls.
            messages.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name,
                                     "arguments": tc.function.arguments},
                    }
                    for tc in tool_calls
                ],
            })

            for tc in tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}

                result = mcp_client.call_tool(name, args)
                status = "ok" if result.ok else "error"
                trace.add_tool_call(
                    name=name, args=args, status=status,
                    duration_ms=result.duration_ms,
                    result=result.result if result.ok else None,
                    error=result.error,
                )
                executed.append({
                    "name": name, "args": args, "status": status,
                    "result": result.result, "error": result.error,
                    "duration_ms": round(result.duration_ms, 1),
                })

                if result.ok:
                    tool_content = json.dumps(result.result, ensure_ascii=False)
                else:
                    tool_content = json.dumps(
                        {"error": result.error,
                         "nota": "La herramienta falló; informa de ello al usuario."},
                        ensure_ascii=False)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_content,
                })

        # Si se agotan las iteraciones, pedimos una respuesta final sin tools.
        trace.note("Se alcanzó el máximo de iteraciones de tools.")
        final = llm.chat_text(messages + [{
            "role": "user",
            "content": "Redacta la respuesta final en español con la información disponible.",
        }])
        return final, executed


def build_orchestrator() -> Orchestrator:
    """Crea un orquestador con los almacenes por defecto."""
    vs = VectorStore(embedder=llm.embed_texts, persist_dir=settings.chroma_dir)
    ps = ParentStore(db_path=settings.parent_db_path)
    return Orchestrator(vs, ps)
