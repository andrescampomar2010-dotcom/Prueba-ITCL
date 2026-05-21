FROM python:3.11-slim

WORKDIR /srv

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/srv \
    DATA_DIR=/data \
    SEED_PDF_DIR=/srv/seed_pdfs

# build-essential cubre la compilación de dependencias nativas (chromadb).
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Instala dependencias primero para aprovechar la caché de capas.
COPY app/requirements.txt /srv/app/requirements.txt
RUN pip install --no-cache-dir -r app/requirements.txt

COPY app /srv/app
COPY scripts /srv/scripts
COPY mcp_server /srv/mcp_server

# Genera el PDF de ejemplo dentro de la imagen (semilla del corpus).
# Paso no crítico: si fallara, el build continúa y el corpus arranca vacío.
RUN mkdir -p /srv/seed_pdfs \
    && (python scripts/generate_sample_pdf.py /srv/seed_pdfs/manual_ejemplo.pdf \
        || echo "AVISO: no se pudo generar el PDF de ejemplo")

EXPOSE 8501

# Healthcheck propio de Streamlit.
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=3)" || exit 1

CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
