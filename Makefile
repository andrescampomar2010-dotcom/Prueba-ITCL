# Atajos del Asistente RAG jerárquico + MCP
.PHONY: help up down build rebuild logs reindex stats test test-local sample-pdf clean

help:
	@echo "Comandos disponibles:"
	@echo "  make up         - Arranca todo el sistema (docker compose up --build)"
	@echo "  make down       - Detiene y elimina los contenedores"
	@echo "  make build      - Construye las imágenes"
	@echo "  make rebuild    - Reconstruye las imágenes sin caché"
	@echo "  make logs       - Muestra los logs en streaming"
	@echo "  make reindex    - Reindexa el corpus de PDFs desde cero"
	@echo "  make stats      - Muestra el estado del índice"
	@echo "  make test       - Ejecuta los tests dentro del contenedor"
	@echo "  make test-local - Ejecuta los tests con el Python local"
	@echo "  make sample-pdf - Regenera el PDF de ejemplo"
	@echo "  make clean      - Elimina contenedores y el volumen de datos"

up:
	docker compose up --build

down:
	docker compose down

build:
	docker compose build

rebuild:
	docker compose build --no-cache

logs:
	docker compose logs -f

# Reindexado del corpus (requiere que el contenedor 'app' esté en marcha).
reindex:
	docker compose exec app python -m app.reindex

stats:
	docker compose exec app python -m app.reindex --stats

# Tests dentro de la imagen ya construida (no requiere Python local).
test:
	docker compose run --rm --no-deps app python -m pytest app/tests -v

# Tests con el intérprete local (requiere: pip install -r app/requirements.txt).
test-local:
	python -m pytest app/tests -v

sample-pdf:
	python scripts/generate_sample_pdf.py data/pdfs/manual_ejemplo.pdf

clean:
	docker compose down -v
