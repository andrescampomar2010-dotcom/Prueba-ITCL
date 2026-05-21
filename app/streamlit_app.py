"""Interfaz de chat (Streamlit) — íntegramente en español.

Funcionalidades:
  - Chat conversacional con el asistente RAG + MCP.
  - Muestra las citas de página de las respuestas basadas en el PDF.
  - Modo depuración: ruta elegida, chunks recuperados y tool calls.
  - Subida de PDFs y su indexado en la base de datos desde la propia interfaz.
  - Listado y borrado de PDFs indexados; botón de reindexado completo.

Nota de diseño: los mensajes de resultado (éxito/error) se guardan en
`st.session_state` y se pintan TRAS el `st.rerun()`. Si se pintaran antes del
rerun, este los descartaría y los botones parecerían "no hacer nada".
"""
from __future__ import annotations

import os
import sys
import traceback

# Permite importar el paquete `app` al ejecutar con `streamlit run`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st  # noqa: E402

from app import llm, mcp_client  # noqa: E402
from app.bootstrap import auto_index_if_empty, ensure_seed_pdfs  # noqa: E402
from app.config import settings  # noqa: E402
from app.file_validation import validate_pdf_file  # noqa: E402
from app.logging_utils import get_logger  # noqa: E402
from app.orchestrator import Orchestrator  # noqa: E402
from app.pdf_ingest import ingest_pdf, list_pdf_files, reindex_all  # noqa: E402
from app.vectorstore import ParentStore, VectorStore  # noqa: E402

log = get_logger("ui")

st.set_page_config(page_title="Asistente RAG jerárquico + MCP",
                   page_icon="📚", layout="wide")


# --------------------------------------------------------------------------
# Mensajes "flash": sobreviven al st.rerun()
# --------------------------------------------------------------------------
def flash(kind: str, message: str) -> None:
    """Guarda un mensaje para mostrarlo después del próximo rerun."""
    st.session_state.setdefault("_flash", []).append((kind, message))


def render_flash() -> None:
    """Pinta y limpia los mensajes flash pendientes."""
    for kind, message in st.session_state.pop("_flash", []):
        getattr(st, kind, st.info)(message)


# --------------------------------------------------------------------------
# Inicialización (cacheada): almacenes, orquestador y autoindexado.
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Inicializando el sistema...")
def init_system() -> dict:
    settings.ensure_dirs()
    vs = VectorStore(embedder=llm.embed_texts, persist_dir=settings.chroma_dir)
    ps = ParentStore(db_path=settings.parent_db_path)
    indexed = False
    try:
        ensure_seed_pdfs()
        indexed = auto_index_if_empty(vs, ps)
    except Exception as exc:  # noqa: BLE001
        log.warning("Autoindexado fallido: %s", exc)
    orchestrator = Orchestrator(vs, ps)
    return {"vs": vs, "ps": ps, "orchestrator": orchestrator, "autoindexado": indexed}


SYS = init_system()
VS: VectorStore = SYS["vs"]
PS: ParentStore = SYS["ps"]
ORCH: Orchestrator = SYS["orchestrator"]

if "messages" not in st.session_state:
    st.session_state.messages = []  # historial: [{role, content, meta}]


# --------------------------------------------------------------------------
# Acciones (subida, borrado, reindexado). Cada una deja un mensaje flash y
# fuerza un rerun para refrescar las listas y contadores de la barra lateral.
# --------------------------------------------------------------------------
def accion_guardar_pdfs(uploaded_files) -> None:
    if not settings.openai_configured:
        flash("error", "Falta `OPENAI_API_KEY`. Configúrala en el archivo `.env`.")
        st.rerun()
        return
    settings.ensure_dirs()
    for file in uploaded_files:
        # Validar archivo antes de procesarlo
        file_bytes = file.getbuffer().tobytes()
        validation = validate_pdf_file(file_bytes, file.name)
        
        if not validation.valid:
            flash("error", f"❌ {file.name}: {validation.error}")
            continue
        
        try:
            dest = os.path.join(settings.pdf_dir, file.name)
            with open(dest, "wb") as fh:
                fh.write(file_bytes)
            result = ingest_pdf(dest, VS, PS)
            if result.ok:
                flash("success", f"✅ {file.name}: {result.pages} págs., "
                                 f"{result.children} fragmentos indexados.")
            else:
                flash("error", f"❌ {file.name}: {result.error}")
        except Exception as exc:  # noqa: BLE001
            log.exception("Error al subir/indexar %s", file.name)
            flash("error", f"❌ {file.name}: {exc}")
    st.rerun()


def accion_reindexar() -> None:
    if not settings.openai_configured:
        flash("error", "Falta `OPENAI_API_KEY`. Configúrala en el archivo `.env`.")
        st.rerun()
        return
    try:
        results = reindex_all(VS, PS)
        if not results:
            flash("warning", "No hay PDFs en el corpus. Sube uno antes de reindexar.")
        else:
            ok = sum(1 for r in results if r.ok)
            total = sum(r.children for r in results)
            flash("success", f"✅ Corpus reindexado: {ok}/{len(results)} PDF(s), "
                             f"{total} fragmentos.")
            for r in results:
                if not r.ok:
                    flash("error", f"❌ {r.source}: {r.error}")
    except Exception as exc:  # noqa: BLE001
        log.exception("Error en el reindexado")
        flash("error", f"❌ Error al reindexar: {exc}")
    st.rerun()


def accion_eliminar(source: str) -> None:
    try:
        VS.delete_source(source)
        PS.delete_source(source)
        path = os.path.join(settings.pdf_dir, source)
        if os.path.exists(path):
            os.remove(path)
        flash("info", f"🗑️ '{source}' eliminado del índice y del corpus.")
    except Exception as exc:  # noqa: BLE001
        log.exception("Error al eliminar %s", source)
        flash("error", f"❌ No se pudo eliminar '{source}': {exc}")
    st.rerun()


# --------------------------------------------------------------------------
# Barra lateral
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Panel de control")
    render_flash()  # mensajes de la acción anterior (sobreviven al rerun)

    debug_mode = st.toggle("Modo depuración", value=False,
                           help="Muestra la ruta elegida, los chunks recuperados "
                                "y las llamadas a herramientas MCP.")

    st.divider()
    st.subheader("Estado del servidor MCP")
    health = mcp_client.server_health()
    if health.get("ok"):
        data = health.get("data", {})
        st.success("MCP conectado")
        st.caption(f"Servicio: {data.get('service', '?')}")
    else:
        st.error("MCP no disponible")
        st.caption(str(health.get("error", ""))[:160])

    tools = mcp_client.discover_tools()
    if tools:
        st.caption("Herramientas descubiertas (no hardcodeadas):")
        for tool in tools:
            with st.expander(f"🔧 {tool.name}"):
                st.write(tool.description)
                st.json(tool.input_schema)
    else:
        st.caption("No se han descubierto herramientas.")

    st.divider()
    st.subheader("📥 Añadir PDF al sistema")
    st.caption("Sube un PDF: se guarda en el corpus y se indexa en la base de datos.")
    if not settings.openai_configured:
        st.warning("Configura `OPENAI_API_KEY` en `.env` para poder indexar.")
    uploaded = st.file_uploader("Selecciona un PDF", type=["pdf"],
                                accept_multiple_files=True,
                                label_visibility="collapsed")
    if st.button("💾 Guardar e indexar", use_container_width=True,
                 disabled=not uploaded):
        with st.spinner("Guardando e indexando..."):
            accion_guardar_pdfs(uploaded)

    st.divider()
    st.subheader("📚 PDFs indexados")
    try:
        sources = VS.sources()
        total_chunks = VS.count()
    except Exception as exc:  # noqa: BLE001
        log.exception("Error al leer el índice")
        sources, total_chunks = [], 0
        st.error(f"No se pudo leer el índice: {exc}")

    if sources:
        for src in sources:
            col1, col2 = st.columns([4, 1])
            col1.write(f"• {src}")
            if col2.button("🗑️", key=f"del_{src}", help=f"Eliminar {src}"):
                accion_eliminar(src)
        st.caption(f"{total_chunks} fragmentos indexados en total.")
    else:
        st.info("Todavía no hay ningún PDF indexado.")

    if st.button("🔄 Reindexar todo el corpus", use_container_width=True):
        with st.spinner("Reindexando el corpus completo..."):
            accion_reindexar()


# --------------------------------------------------------------------------
# Cabecera y avisos
# --------------------------------------------------------------------------
st.title("📚 Asistente RAG jerárquico + MCP")
st.caption("Pregunta sobre la documentación interna en PDF. El asistente cita las "
           "páginas y, cuando hace falta, consulta herramientas externas vía MCP.")

if not settings.openai_configured:
    st.warning("⚠️ No se ha detectado `OPENAI_API_KEY`. Copia `.env.example` a "
               "`.env`, añade tu clave y reinicia. Sin clave no se puede indexar "
               "ni responder.")

if not list_pdf_files():
    st.info("ℹ️ No hay PDFs en el corpus. Sube uno desde el panel lateral.")


# --------------------------------------------------------------------------
# Render del historial
# --------------------------------------------------------------------------
def render_meta(meta: dict) -> None:
    """Pinta citas, herramientas y, si procede, la traza de depuración."""
    citations = meta.get("citations") or []
    if citations:
        with st.expander(f"📄 Fuentes citadas ({len(citations)})"):
            for cit in citations:
                st.markdown(f"**{cit['label']}**")
                st.caption(cit["snippet"] + "…")

    tool_calls = meta.get("tool_calls") or []
    if tool_calls:
        with st.expander(f"🔧 Herramientas usadas ({len(tool_calls)})"):
            for call in tool_calls:
                icono = "✅" if call["status"] == "ok" else "❌"
                st.markdown(f"{icono} **{call['name']}** "
                            f"({call['duration_ms']} ms)")
                st.caption(f"Argumentos: `{call['args']}`")
                if call["status"] == "ok":
                    st.json(call["result"])
                else:
                    st.error(call["error"])

    if meta.get("debug") and meta.get("trace"):
        trace = meta["trace"]
        with st.expander("🐞 Traza de depuración", expanded=True):
            st.markdown(f"**Ruta elegida:** `{trace.get('route', '?')}`")
            st.caption(f"Motivo: {trace.get('route_reason', '')}")
            st.caption(f"Tiempo total: {trace.get('elapsed_ms', '?')} ms")
            chunks = trace.get("retrieved_chunks") or []
            if chunks:
                st.markdown(f"**Chunks recuperados (nivel 1): {len(chunks)}**")
                for ch in chunks:
                    dist = ch.get("distance")
                    dist_txt = f" · dist={dist}" if dist is not None else ""
                    st.caption(f"`{ch['chunk_id']}` · pág. {ch['page']}{dist_txt}")
                    st.text(ch["snippet"])
            if trace.get("tool_calls"):
                st.markdown("**Tool calls (traza):**")
                st.json(trace["tool_calls"])
            if trace.get("notes"):
                st.markdown("**Notas:**")
                for note in trace["notes"]:
                    st.caption(f"– {note}")


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("meta"):
            render_meta(msg["meta"])


# --------------------------------------------------------------------------
# Entrada de chat
# --------------------------------------------------------------------------
prompt = st.chat_input("Escribe tu pregunta en español...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not settings.openai_configured:
            text = ("No puedo responder: falta configurar `OPENAI_API_KEY` "
                    "en el archivo `.env`.")
            st.markdown(text)
            st.session_state.messages.append(
                {"role": "assistant", "content": text, "meta": {}})
        else:
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]
                if m["role"] in ("user", "assistant")
            ]
            with st.spinner("Pensando..."):
                try:
                    answer = ORCH.answer(prompt, history=history, debug=debug_mode)
                    text = answer.text
                    meta = {
                        "citations": [
                            {"label": c.label, "snippet": c.snippet}
                            for c in answer.citations
                        ],
                        "tool_calls": answer.tool_calls,
                        "trace": answer.trace,
                        "debug": debug_mode,
                        "route": answer.route,
                    }
                except Exception as exc:  # noqa: BLE001
                    log.exception("Error al responder")
                    text = (f"Se ha producido un error al procesar la pregunta: "
                            f"{exc}")
                    if debug_mode:
                        text += f"\n\n```\n{traceback.format_exc()}\n```"
                    meta = {}

            st.markdown(text)
            if debug_mode and meta.get("route"):
                st.caption(f"Ruta: `{meta['route']}`")
            render_meta(meta)
            st.session_state.messages.append(
                {"role": "assistant", "content": text, "meta": meta})
