# Asistente RAG jerárquico + MCP

Asistente conversacional **en español** que responde preguntas apoyándose en un
**RAG jerárquico sobre PDF** y, cuando procede, llama **herramientas externas
expuestas vía MCP (SSE)**. Reproducible con Docker Compose y con trazabilidad
completa de cada decisión.

> Regla de oro del sistema: toda respuesta correcta se apoya en (a) citas del
> PDF, (b) una tool MCP, o (c) ambas. Si la información no está disponible, el
> asistente lo dice explícitamente y no inventa.

---

## 1. Arquitectura

Dos servicios en Docker Compose (más un tercer almacén embebido):

```
   Navegador
       |
       v
+-------------------+        descubre tools + llama tools (MCP sobre SSE)
|   app  (Streamlit |  <----------------------------------------------+
|   + Orquestador)  |                                                 |
|                   |        +----------------------------+           |
|  - Router         |        |  mcp-server                |           |
|  - RAG jerárquico | -----> |  Tools por SSE:            | <---------+
|  - Cliente MCP    |        |   - fx_rate                |
|                   |        |   - market_status         |
|  VectorStore      |        |  Endpoint /health          |
|  (ChromaDB) +     |        +----------------------------+
|  ParentStore      |
|  (SQLite)         |
+-------------------+
```

- **Servicio `app`** (puerto 8501): interfaz de chat en Streamlit y orquestación
  completa — router RAG/TOOL, acceso al vector store y generación de la
  respuesta. Incluye ChromaDB embebido (índice vectorial) y SQLite (almacén de
  chunks padre), persistidos en un volumen Docker.
- **Servicio `mcp-server`** (puerto 8000): servidor MCP independiente que expone
  las tools por **SSE** y un endpoint `/health`.

El cliente **descubre** las tools conectándose al servidor MCP al arrancar: no
hay ninguna tool hardcodeada en el cliente.

### Estructura del proyecto

```
rag-mcp-assistant/
├── docker-compose.yml        # 2 servicios + volumen persistente
├── .env.example              # plantilla de configuración (sin secretos)
├── Makefile                  # atajos (up, reindex, test, ...)
├── README.md
├── pytest.ini
├── mcp_server/               # SERVICIO 2 — servidor MCP
│   ├── server.py             # transporte SSE + endpoints
│   ├── tools_logic.py        # lógica determinista de las tools
│   ├── requirements.txt
│   └── Dockerfile
├── app/                      # SERVICIO 1 — cliente / orquestador
│   ├── streamlit_app.py      # interfaz de chat (español)
│   ├── orchestrator.py       # une router + RAG + tools
│   ├── router.py             # clasificador RAG/TOOL/BOTH/NONE
│   ├── rag.py                # recuperación jerárquica + generación
│   ├── chunking.py           # chunking jerárquico + mapeo de páginas
│   ├── vectorstore.py        # ChromaDB (hijos) + SQLite (padres)
│   ├── pdf_ingest.py         # extracción e indexado de PDFs
│   ├── mcp_client.py         # cliente MCP sobre SSE (descubre/llama tools)
│   ├── llm.py                # cliente OpenAI (LLM + embeddings)
│   ├── bootstrap.py          # autoindexado al primer arranque
│   ├── reindex.py            # reindexado por línea de comandos
│   ├── config.py             # configuración por variables de entorno
│   ├── logging_utils.py      # logs JSON + traza por petición
│   ├── requirements.txt
│   ├── Dockerfile
│   └── tests/                # test_router, test_tool_call, test_citation
├── scripts/
│   └── generate_sample_pdf.py  # genera el PDF de ejemplo en español
└── data/                     # corpus, índice y logs (volumen persistente)
```

---

## 2. Cómo ejecutar

Requisitos: Docker y Docker Compose. Una clave de la API de OpenAI.

```bash
# 1. Configura el entorno
cp .env.example .env
#    edita .env y pon tu OPENAI_API_KEY

# 2. Arranca todo
docker compose up --build

# 3. Abre la interfaz
#    http://localhost:8501
```

`docker compose up --build` levanta los dos servicios sin pasos manuales:
el `app` espera a que `mcp-server` esté *healthy*, **genera el PDF de ejemplo**
(durante el build de la imagen), lo copia al corpus y lo **indexa
automáticamente** en el primer arranque. El sistema queda usable de inmediato.

Atajos equivalentes con `make`: `make up`, `make down`, `make logs`,
`make reindex`, `make test` (ver `make help`).

### Reindexado

- **Desde la interfaz**: botón *"Reindexar todo el corpus"* en el panel lateral
  (no hay que borrar carpetas a mano).
- **Por línea de comandos**: `make reindex` (ejecuta `python -m app.reindex`
  dentro del contenedor).
- Subir un PDF nuevo desde la barra lateral lo guarda en el corpus y lo indexa
  al instante.

---

## 3. RAG jerárquico — estrategia de chunking

Se implementa un RAG jerárquico de **dos niveles** (patrón *small-to-big* /
*parent retrieval*):

- **Nivel 1 — chunks hijo (~450 caracteres).** Son los fragmentos pequeños que
  se **indexan y se recuperan por similitud** en ChromaDB (distancia coseno
  sobre embeddings `text-embedding-3-small`). Al ser pequeños, la recuperación
  es precisa.
- **Nivel 2 — chunks padre (~1800 caracteres).** Cada hijo guarda el `parent_id`
  de su chunk padre. Tras recuperar los hijos más relevantes, se **reconstruye
  el contexto amplio** recuperando sus padres únicos. La generación final usa
  ese contexto grande, no los fragmentos sueltos.

El texto del PDF se concatena conservando un **mapa de páginas** (offset de
carácter → página), de modo que cada chunk conoce su página o rango de páginas.
El troceado es recursivo y respeta límites semánticos (párrafos, frases,
palabras), por lo que nunca parte una palabra ni una cifra.

**Metadatos por chunk hijo:** `chunk_id`, `source` (nombre del PDF), `page`,
`pages` (rango), `parent_id`, `char_start`, `char_end`.

**Citación:** toda respuesta basada en el PDF incluye la página entre paréntesis
`(pág. N)`. En modo depuración se muestran los `chunk_id` recuperados, su página
y el fragmento concreto. Si la información no está en el PDF, el asistente lo
indica explícitamente y no responde.

Parámetros configurables por entorno: `PARENT_CHUNK_SIZE`, `CHILD_CHUNK_SIZE`,
`TOP_K_CHILDREN`, `MAX_PARENTS` (ver `.env.example`).

---

## 4. Estrategia de enrutado (RAG vs Tools)

El orquestador decide la ruta de cada pregunta entre **RAG / TOOL / BOTH / NONE**:

1. **Clasificador LLM** con un prompt breve y **salida estructurada JSON**
   (`{"route": "...", "reason": "..."}`). Recibe la lista de tools descubiertas
   del MCP para decidir con conocimiento de lo que hay disponible.
2. **Heurística por reglas como red de seguridad (fallback):** si el LLM falla,
   no está disponible o devuelve una ruta inválida, se aplica una clasificación
   por palabras clave.
3. **Ejecución de la ruta:**
   - `RAG` → recuperación jerárquica + generación con citas.
   - `TOOL` → *function-calling* nativo de OpenAI restringido a las tools del
     MCP; el modelo elige tool y argumentos, el orquestador la ejecuta vía SSE.
   - `BOTH` → se recupera contexto del PDF **y** se habilitan las tools: el
     modelo combina la cifra del documento con la herramienta (p. ej. conversión
     de divisa) y muestra el cálculo.
   - `NONE` → responde explícitamente que no está disponible.

La decisión y su motivo se registran siempre en los logs y en la traza visible
en modo depuración.

---

## 5. Integración MCP (tools vía SSE)

El servidor MCP (`mcp_server/`) es un **contenedor independiente** que usa el
SDK oficial de MCP con transporte **SSE**. Expone:

| Tool            | Descripción                                   | Input                              | Output (campos)                            |
|-----------------|-----------------------------------------------|------------------------------------|--------------------------------------------|
| `fx_rate`       | Tipo de cambio de un par de divisas           | `{"base": "EUR", "quote": "USD"}`  | `base, quote, rate, as_of` (ISO-8601)      |
| `market_status` | Estado abierto/cerrado de un mercado          | `{"market": "NYSE"}`               | `market, name, is_open, session_hours_utc, as_of` |

Ambas son **mocks deterministas** (no requieren internet), según permite el
enunciado. El cliente:

- **Descubre** las tools al arrancar (`tools/list` del protocolo MCP) — nunca
  están hardcodeadas.
- Maneja los **errores típicos**: servidor caído, *timeout* (configurable) y
  respuestas malformadas o de error.
- Aplica **reintentos con backoff exponencial** en cada llamada.
- Registra en los logs cada *tool call*: nombre, argumentos, duración y estado.

Endpoints del servidor: `GET /sse` (canal MCP), `POST /messages/` (mensajes
MCP) y `GET /health` (usado por el `healthcheck` de Docker Compose).

---

## 6. Observabilidad

- **Logs JSON estructurados** a stdout y a `data/logs/app.log`, con: query, ruta
  elegida y su motivo, chunks recuperados, y cada *tool call* con su duración y
  estado.
- **Modo depuración** en la interfaz (interruptor del panel lateral): muestra la
  ruta elegida, los chunks de nivel 1 recuperados con su página y distancia, y
  las tool calls con sus argumentos y resultados.

---

## 7. Casos de prueba (T1–T6)

Con el PDF de ejemplo incluido (`Manual de Procedimientos Internos`):

| ID | Pregunta de ejemplo                                                | Ruta esperada |
|----|--------------------------------------------------------------------|---------------|
| T1 | ¿Cuántos días de vacaciones anuales tengo?                         | RAG           |
| T2 | Resume la sección de seguridad de la información.                  | RAG           |
| T3 | Convierte a dólares el presupuesto anual del departamento.         | BOTH          |
| T4 | ¿Está abierto el mercado NYSE ahora mismo?                         | TOOL          |
| T5 | ¿Cuál es la política de teletrabajo de la empresa en Japón?        | NONE          |
| T6 | ¿Cuál es el tipo de cambio? (pregunta ambigua)                     | RAG/TOOL      |

- **T1/T2** se responden con cita de página del PDF.
- **T3** extrae los `12.500 EUR` del PDF (con cita), llama a `fx_rate` y calcula.
- **T4** llama a `market_status` y responde con el estado del mercado.
- **T5** no está en el PDF ni lo cubren las tools → el asistente lo indica.
- **T6** al ser ambigua, el router elige una ruta razonada (o el asistente pide
  aclaración).

---

## 8. Tests

Tres tests mínimos (más casos adicionales), sin necesidad de clave ni de red:

- `test_router.py` — decisión RAG/TOOL/BOTH/NONE y fallback heurístico.
- `test_tool_call.py` — tools MCP (mock determinista) y parseo de resultados,
  incluyendo respuestas de error y malformadas.
- `test_citation.py` — el chunking jerárquico conserva la página correcta;
  la cifra del PDF se cita en su página real.

Ejecución: `make test` (dentro del contenedor) o `make test-local`
(`pip install -r app/requirements.txt && python -m pytest app/tests -v`).

---

## 9. Decisiones técnicas

- **Stack Python** (recomendado por el enunciado): Streamlit (UI), ChromaDB
  (vector store persistente), SDK oficial de MCP con SSE, OpenAI para LLM y
  embeddings.
- **Vector store de hijos en ChromaDB; padres en SQLite.** Los padres solo se
  buscan por clave (no necesitan embedding), así que SQLite es más simple y
  barato que indexarlos también.
- **Embeddings inyectables.** `VectorStore` recibe el *embedder* como
  dependencia: en producción es OpenAI; en los tests, un embedder determinista.
- **Function-calling nativo para TOOL/BOTH**, alimentado con los esquemas de
  tools descubiertos del MCP: aprovecha el routing del modelo manteniendo el
  desacople (las tools no se hardcodean).
- **Autoindexado al primer arranque** para que el sistema sea usable sin pasos
  manuales; reindexado disponible desde la UI y por CLI.
- **Configuración 100 % por variables de entorno**; ningún secreto en el repo
  (`.env` está en `.gitignore`, se entrega `.env.example`).

---

## 10. Estrategia de Enrutado (Router RAG vs Tools)
El sistema implementa un enrutado inteligente que decide dinámicamente si responder una pregunta utilizando:
- **Solo RAG (documentación interna del PDF)**;
- **Solo Tools (herramientas externas vía MCP)**;
- **Both (RAG + Tool)**;
- **None (no se puede responder)**;

Enfoque técnico elegido
Se ha optado por Tool Calling nativo del LLM (function calling) como estrategia principal, combinado con un fallback basado en análisis de intención.
- **Funcionamiento paso a paso:**
Descubrimiento dinámico de tools
Al iniciar la aplicación, el cliente se conecta al servidor MCP mediante SSE y obtiene la lista completa de herramientas disponibles junto con sus esquemas JSON (input_schema). De esta forma, ninguna tool está hardcodeada en el cliente.
Construcción del prompt del router
Para cada pregunta del usuario, se construye un prompt que incluye:
La pregunta del usuario
Descripción de todas las tools disponibles (nombre + descripción)
Instrucciones claras sobre cuándo usar RAG, Tools o ambos
Decisión mediante Function Calling
Se envía al LLM una llamada con las tools disponibles (market_status, fx_rate, etc.) más una tool virtual interna llamada retrieve_documents (que representa el RAG).
El modelo decide automáticamente qué tools llamar (o ninguna).
Lógica de ejecución
Si el LLM decide llamar a retrieve_documents → se ejecuta el RAG jerárquico.
Si decide llamar a una o varias tools MCP → se ejecutan las tools correspondientes.
Si decide ambas → se ejecutan en paralelo (RAG + Tools).
Si no llama a ninguna → se responde con un mensaje claro indicando que no se dispone de información suficiente.
Fallback de seguridad
En caso de que el function calling falle o el modelo no decida correctamente, existe un fallback basado en keywords + verificación secundaria del LLM para evitar respuestas sin fuente.
Ventajas de este enfoque
Mayor precisión que un clasificador binario simple.
Flexibilidad para añadir nuevas tools sin tocar el código del cliente.
Soporta naturalmente el caso BOTH (ej: extraer una cifra del PDF y convertirla con fx_rate).
Total trazabilidad: se registra la decisión del router, las tools llamadas y los chunks recuperados.

## 10. Limitaciones

- Se asume **PDF con texto seleccionable** (sin OCR), según el alcance del
  enunciado. Un PDF escaneado no se indexará correctamente.
- Las tools `fx_rate` y `market_status` son **mocks deterministas**: los tipos
  de cambio y horarios provienen de tablas fijas, no de datos reales en vivo.
- El **PDF de ejemplo se genera durante el build** de la imagen Docker (con
  `scripts/generate_sample_pdf.py`) y se indexa en el primer arranque; así el
  repositorio no necesita versionar un binario. Puede regenerarse con
  `make sample-pdf`.
- Las **versiones de dependencias están fijadas** para reproducibilidad; si tu
  índice de PyPI no dispusiera de alguna versión exacta, ajústala en los
  `requirements.txt`.
- La calidad de las respuestas depende del modelo OpenAI configurado
  (`OPENAI_MODEL`, por defecto `gpt-4o-mini`).
