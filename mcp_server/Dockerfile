FROM python:3.11-slim

WORKDIR /srv

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/srv

# Instala dependencias primero para aprovechar la caché de capas de Docker.
COPY mcp_server/requirements.txt /srv/mcp_server/requirements.txt
RUN pip install --no-cache-dir -r mcp_server/requirements.txt

COPY mcp_server /srv/mcp_server

EXPOSE 8000

CMD ["python", "mcp_server/server.py"]
