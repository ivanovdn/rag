# The Python the production container runs (2026-10-08). A bare 3.12 tag floats
# to each new patch release on a cache miss, same as the >= ranges did.
FROM python:3.12.15-slim

WORKDIR /app

# System deps for python-docx, sentence-transformers, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (better layer caching).
# Slim runtime deps — no HuggingFace embeddings (production uses Ollama embeddings).
# requirements-bot.lock pins every package to the versions proven in production;
# -c constrains versions without adding packages. See the lock's header.
COPY requirements-bot.txt requirements-bot.lock ./
RUN pip install --no-cache-dir -r requirements-bot.txt -c requirements-bot.lock

# Copy only runtime code
COPY config.py .
COPY rag/ ./rag/
COPY channels/ ./channels/
COPY scripts/start_teams_bot.py ./scripts/start_teams_bot.py

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# The commit this image was built from, printed as "Build:" in the startup banner.
# Declared down here, below the dependency layer, on purpose: every RUN below an
# ARG sees it as an environment variable, so declared above `pip install` it would
# bust that layer's cache on every commit -- reinstalling, and with
# requirements-bot.txt's >= ranges re-resolving, every dependency on every deploy.
# docker-compose-remote.yml passes it in; without it the banner says "unknown".
ARG GIT_COMMIT=unknown
ENV GIT_COMMIT=${GIT_COMMIT}

CMD ["python", "scripts/start_teams_bot.py"]
