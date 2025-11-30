.PHONY: up build run test lint clean

# Build docker image
build:
	docker build -t contract-intel .

# Bring up with docker-compose (used in Loom demo)
up:
	docker-compose up --build -d

# Run container (single)
run:
	docker run --rm -it -p 8000:8000 \
	  -e STORE_DIR=/app/store \
	  -e FORCE_SYNC_EXTRACT=1 \
	  -e LOGLEVEL=INFO \
	  -v $(PWD)/data:/app/data:ro \
	  -v $(PWD)/store:/app/store \
	  contract-intel

# Run tests locally in venv
test:
	pytest -q

# simple lint (if you have flake8 installed)
lint:
	flake8 .

# remove compose resources and local image
clean:
	docker-compose down --rmi local -v || true
	rm -rf store/*
