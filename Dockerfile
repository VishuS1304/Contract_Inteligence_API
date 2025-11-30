# Dockerfile - production-ish image for local dev
FROM python:3.11-slim

# Avoid interactive prompts from apt
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install system deps for building and small utilities (curl, ps)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        procps \
        libgl1 \
        git \
        poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements early to leverage layer cache
COPY requirements.txt .

# Install python deps
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt \
    && pip cache purge

# Copy app
COPY . .

# Create store directories and non-root user
RUN mkdir -p /app/store/pdfs /app/store/texts \
    && adduser --disabled-password --gecos "" appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# Uvicorn command (use --proxy-headers if reverse proxy used)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--loop", "uvloop"]
