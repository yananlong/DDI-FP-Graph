# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VIRTUALENVS_CREATE=0

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && pip install "poetry>=1.5"

WORKDIR /workspace

COPY pyproject.toml ./
RUN poetry install --no-interaction --no-ansi --no-root

COPY . .

ENTRYPOINT ["python", "-m", "GPU.train"]
CMD ["--config", "GPU/configs/graph.yaml"]
