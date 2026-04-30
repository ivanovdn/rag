FROM python:3.12-slim

WORKDIR /app

# System deps for python-docx, sentence-transformers, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only runtime code
COPY config.py .
COPY rag/ ./rag/
COPY channels/ ./channels/
COPY scripts/start_teams_bot.py ./scripts/start_teams_bot.py

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

CMD ["python", "scripts/start_teams_bot.py"]
