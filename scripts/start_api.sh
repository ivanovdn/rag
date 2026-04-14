#!/bin/bash
# Start the Compliance Bot API
#
# Prerequisites:
#   - Reranker running on port 8081 (llama-server)
#   - Ollama running with the configured LLM model
#   - Qdrant running on port 6333
#   - Documents ingested
#
# Usage:
#   ./scripts/start_api.sh

PYTHONPATH=. uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
