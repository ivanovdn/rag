from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # HuggingFace
    hf_token: str = ""

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_remote_url: str = "http://192.168.100.2:11434"
    use_remote_ollama: bool = False
    llm_model: str = "qwen3:14b"
    embedding_model: str = "nomic-embed-text"
    embedding_query_prefix: str = ""
    embedding_passage_prefix: str = ""
    llm_temperature: float = 0.0
    llm_request_timeout: int = 120
    llm_remote_request_timeout: int = 300

    # LLM backend
    llm_backend: str = "ollama"  # "ollama" or "openai-compatible"
    openai_api_base: str = "http://localhost:8082/v1"
    openai_api_key: str = "not-needed"
    openai_model: str = "qwen2.5-32b"

    @property
    def active_ollama_url(self) -> str:
        return self.ollama_remote_url if self.use_remote_ollama else self.ollama_base_url

    @property
    def active_request_timeout(self) -> int:
        return self.llm_remote_request_timeout if self.use_remote_ollama else self.llm_request_timeout

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_remote_url: str = "http://localhost:6333"
    use_remote_qdrant: bool = False
    qdrant_collection: str = "compliance_policies"
    qdrant_vector_dim: int = 768

    @property
    def active_qdrant_url(self) -> str:
        return self.qdrant_remote_url if self.use_remote_qdrant else self.qdrant_url

    # Embeddings
    embedding_source: str = "huggingface"  # "huggingface" or "ollama"
    ollama_embedding_url: str = "http://localhost:11434"

    # Documents
    policy_docs_folder: str = "./policies"
    policy_base_url: str = "http://intranet.company.com/policies"

    # Retrieval
    retrieval_top_k: int = 10
    min_confidence_score: float = 0.45

    # Reranker (any /v1/rerank-compatible server: llama-server, vLLM, etc.)
    reranker_enabled: bool = False
    reranker_backend: str = "llama-server"  # "llama-server" or "vllm"
    reranker_url: str = "http://localhost:8081"
    reranker_model: str = "qwen3-reranker-0.6b-q8"
    reranker_query_template: str = "<Instruct>: {instruction}\n<Query>: {query}"
    reranker_top_n: int = 6
    reranker_candidates: int = 20
    reranker_instruction: str = "Given an employee compliance question, retrieve the internal policy clause that answers it"

    # Hybrid search
    bm25_enabled: bool = True
    hybrid_vector_candidates: int = 20
    hybrid_bm25_candidates: int = 20

    # Pipeline
    pipeline_mode: str = "agentic"  # "agentic" or "vanilla"

    # Agent
    agent_max_iterations: int = 8
    agent_timeout: int = 120

    # Chunking
    chunk_min_tokens: int = 50
    chunk_max_tokens: int = 400

    # Escalation Email
    smtp_host: str = "smtp.company.com"
    smtp_port: int = 587
    smtp_user: str = "bot@company.com"
    smtp_password: str = ""
    compliance_team_email: str = "compliance@company.com"
    escalation_ticket_prefix: str = "ESC"

    # API
    api_secret_key: str = "changeme"
    admin_api_key: str = "changeme"

    # SQLite
    database_url: str = "sqlite:///./compliance_bot.db"

    # Teams Bot
    teams_tenant_id: str = ""
    teams_client_id: str = ""
    teams_client_secret: str = ""
    teams_refresh_token: str = ""
    teams_poll_interval: int = 5
    teams_api_timeout: int = 10
    teams_initial_lookback_minutes: int = 5
    teams_max_consecutive_errors: int = 5
    teams_max_processed_messages: int = 1000

    # Observability (Phoenix)
    phoenix_enabled: bool = True
    phoenix_endpoint: str = "http://localhost:6006/v1/traces"
    phoenix_project_name: str = "compliance-bot"

    # Evaluation
    eval_dataset_path: str = "eval/datasets"
    eval_confidence_threshold: float = 0.45

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
