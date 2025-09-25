.DEFAULT_GOAL := default

.PHONY: test
test:
	uv run --no-sync pytest --cov=cccv --cov-report=xml --cov-report=html

.PHONY: lint
lint:
	uv run --no-sync pre-commit install
	uv run --no-sync pre-commit run --all-files

.PHONY: build
build:
	uv build --wheel

.PHONY: vs
vs:
	rm -f encoded.mkv
	vspipe -c y4m example/sr_vs.py - | ffmpeg -i - -vcodec libx264 encoded.mp4

.PHONY: dev
dev:
	docker compose -f cccv-docker-compose.yml down
	docker compose -f cccv-docker-compose.yml up -d
