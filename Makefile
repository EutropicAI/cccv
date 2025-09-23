.DEFAULT_GOAL := default

.PHONY: test
test:
	uv run --no-sync pytest --cov=ccrestoration --cov-report=xml --cov-report=html

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
	vspipe -c y4m example/vapoursynth.py - | ffmpeg -i - -vcodec libx265 -crf 16 encoded.mkv
