from functools import lru_cache

from pydantic import model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # HuggingFace
    hf_token: str = ""

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_remote_url: str = "http://172.20.0.22:11434"
    use_remote_ollama: bool = False
    llm_model: str = "qwen3:14b"
    embedding_model: str = "nomic-embed-text"
    embedding_query_prefix: str = ""
    embedding_passage_prefix: str = ""
    llm_temperature: float = 0.0
    llm_request_timeout: int = 120
    llm_remote_request_timeout: int = 300
    # How long Ollama keeps the model resident after a request, as a duration
    # string (e.g. "30m", "1h"). Ollama's own default is "5m"; sporadic use
    # then pays a reload (measured 3.8s-51.7s on the shared host). Not
    # unbounded: the box is shared with other projects.
    ollama_keep_alive: str = "30m"
    # Allocated Ollama context window (num_ctx). 4096 is the ONLY value the
    # upstream MoE+CUDA crash matrix proved safe — see
    # docs/superpowers/specs/2026-09-17-ollama-moe-cuda-crash.md. This is a
    # crash-avoidance value, not a performance knob: do not raise it without
    # re-measuring against that spec.
    ollama_num_ctx: int = 4096

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

    # Agent
    agent_max_iterations: int = 8
    agent_timeout: int = 120

    # Router (pre-retrieval classification)
    router_enabled: bool = True
    router_llm_model: str = ""            # classifier model override; empty -> main LLM
    router_confidence_floor: float = 0.6  # below this -> safe default IN_SCOPE

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
    teams_idle_poll_interval: int = 30        # outside business hours / weekends
    teams_business_hours_start_utc: int = 7   # fast polling from this UTC hour (inclusive), Mon-Fri...
    teams_business_hours_end_utc: int = 19    # ...until this UTC hour (exclusive)
    teams_messages_page_size: int = 5         # $top on the per-chat message fetch (Graph default: 20)
    teams_api_timeout: int = 10
    teams_initial_lookback_minutes: int = 5
    teams_max_state_age_minutes: int = 60   # clamp last_check older than this on startup (anti-backlog-flood)
    teams_max_consecutive_errors: int = 5
    # Bounds the drain in TeamsBot._graceful_shutdown. It must stay BELOW the
    # container's stop grace — docker-compose-remote.yml pins the bot service's
    # stop_grace_period to 15s for exactly this reason, and says so — so the bot
    # finishes draining, saves state and exits on its own rather than being
    # SIGKILLed, which is the hard crash this whole path exists to avoid. Raising
    # this past that grace silently gives the behaviour back, and nothing fails
    # loudly when it does: change both together.
    #
    # 12, not 8: measured on the VM 2026-09-23, a warm answer takes ~6.9s end to
    # end, so an 8s drain left 1.1s of margin and timed out on its very first
    # production restart. 12s clears a warm answer comfortably while still
    # leaving 3s under the 15s container grace to save state and exit. A COLD
    # answer (~15.9s, first question after a deploy) still exceeds this and will
    # time out — that is accepted, not overlooked: the drain is best-effort, and
    # a message it abandons stays in _inflight, so _save_state keeps its id out
    # of the persisted set and holds the watermark behind it, and the next start
    # re-delivers it. Timing out costs one wasted GPU run and a few seconds of
    # delay, never a lost or duplicated answer.
    teams_shutdown_grace_seconds: int = 12
    # A TRIGGER threshold, not a hard cap (branch review Fix B): crossing it makes
    # _cleanup_processed_messages fire, and that removes only 20% of the current
    # size — see its comment in bot.py. Resident size can run to roughly 5x this
    # value, worse during a hold, when cleanup is gated (Ruling H) to fire at most
    # once per teams_max_state_age_minutes instead of every cycle. Tune expecting
    # "~5x this number" of memory/file size, not "this number".
    teams_max_processed_messages: int = 1000

    @model_validator(mode="after")
    def _validate_business_hours(self) -> "Settings":
        """Warn (never crash) on a business-hours window that silently misbehaves.

        TeamsBot._current_poll_interval compares these as plain ints against
        datetime.hour (0-23); it never raises on a bad value, it just quietly
        always returns one interval — start == end is an always-empty window
        (always idle), and anything outside 0-23 (e.g. END=24, meant as
        "midnight") doesn't behave the way that value implies.
        """
        start = self.teams_business_hours_start_utc
        end = self.teams_business_hours_end_utc
        if not (0 <= start <= 23) or not (0 <= end <= 23):
            print(
                f"WARNING: TEAMS_BUSINESS_HOURS_START_UTC/_END_UTC must be 0-23 "
                f"(got start={start}, end={end}); hour comparisons will not behave as expected."
            )
        elif start == end:
            print(
                f"WARNING: TEAMS_BUSINESS_HOURS_START_UTC == TEAMS_BUSINESS_HOURS_END_UTC "
                f"({start}); that window is always empty, so polling will always use the "
                "idle interval, never the fast one."
            )
        return self

    @model_validator(mode="after")
    def _validate_hold_bound(self) -> "Settings":
        """Warn (never crash) if the startup clamp would undo its own purpose.

        _load_state clamps a stale last_check — one older than
        teams_max_state_age_minutes — back to `now - teams_initial_lookback_minutes`,
        so a long-stopped bot cannot answer the whole backlog into the channel.
        That only works while the lookback is strictly smaller than the bound.
        At or above it the clamp re-opens a window at least as wide as the age
        it just rejected as too stale, so the very messages the clamp exists to
        suppress are handed straight back as new. A realistic way to reach this:
        raising TEAMS_INITIAL_LOOKBACK_MINUTES after an incident without also
        raising TEAMS_MAX_STATE_AGE_MINUTES.

        Note this used to justify itself by the runtime force-advance, which
        targeted `now - lookback` under Ruling M. It no longer does: that target
        is now plain `now` (see process_new_messages), so the force-advance
        always clears the hold whatever this setting says. The startup clamp is
        the only reason left to warn — but it is reason enough.
        """
        lookback = self.teams_initial_lookback_minutes
        bound = self.teams_max_state_age_minutes
        if lookback >= bound:
            print(
                f"WARNING: TEAMS_INITIAL_LOOKBACK_MINUTES ({lookback}) >= "
                f"TEAMS_MAX_STATE_AGE_MINUTES ({bound}); the startup clamp would "
                "re-open a window at least as old as the staleness it rejects, so a "
                "long-stopped bot can still answer the backlog it exists to suppress."
            )
        return self

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
