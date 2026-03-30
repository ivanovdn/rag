from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # HuggingFace
    hf_token: str = ""

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "qwen3:14b"
    embedding_model: str = "nomic-embed-text"
    llm_temperature: float = 0.0
    llm_request_timeout: int = 120

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "compliance_policies"
    qdrant_vector_dim: int = 768

    # Documents
    policy_docs_folder: str = "./policies"
    policy_base_url: str = "http://intranet.company.com/policies"

    # Retrieval
    retrieval_top_k: int = 10
    min_confidence_score: float = 0.45

    # Hybrid search
    bm25_enabled: bool = True
    hybrid_vector_candidates: int = 20
    hybrid_bm25_candidates: int = 20

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
