#!/bin/bash
# Start the Qwen3 Reranker via llama-server
# Model is cached at ~/Library/Caches/llama.cpp/ after first download (~609 MB)

llama-server \
  -hf Voodisss/Qwen3-Reranker-0.6B-GGUF-llama_cpp:Q8_0 \
  --reranking \
  --pooling rank \
  --embedding \
  --port 8081
