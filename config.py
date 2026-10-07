import logging
import os
from collections.abc import Callable, Mapping
from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    # The three live credentials are SecretStr, not str, so neither repr(settings)
    # nor str(settings) can carry them. pytest reprs the subexpressions of a
    # failing assert into its message, and monkeypatch.setattr interpolates
    # repr(target) on a mistyped attribute name -- so before this, a wrong NUMBER
    # in an unrelated test could print all three into CI output. Read them with
    # .get_secret_value(); a missed unwrap raises (TypeError at os.environ, at
    # requests' urlencode and at json.dump), it does not send the mask.

    # HuggingFace
    hf_token: SecretStr = SecretStr("")

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
    min_confidence_score: float = 0.45

    @property
    def cosine_floor_applies(self) -> bool:
        """Whether retrieval's top score is a cosine similarity, and so whether
        `min_confidence_score` -- a cosine threshold -- is allowed to judge it.

        True only with the reranker off AND BM25 off, and both terms are
        load-bearing. A reranked result carries a 0.0-1.0 rerank score, whose
        floor is `reranker_min_score`, not this one. A fused result carries an
        RRF score of about 1/(k + rank) ~ 0.016, with k pinned to 60 in
        rag/vector_store.py -- so dropping the bm25 term compares 0.016 against
        0.45 and returns NO_MATCH for every question in the corpus, with no
        error raised anywhere.

        That total failure is the diagnosable one. At Qdrant's default k of 2,
        RRF scores land in the same range as cosine ones and the same mistake
        would fail only for some questions. Same class as the
        rerank_score-presence guard in search_policies Step 3b: the test is on
        what the number MEANS, never on which component produced it.

        One definition on purpose -- it was written out longhand in four places
        and is pinned there by
        test_nothing_rederives_the_cosine_floor_predicate_inline.
        """
        return not self.reranker_enabled and not self.bm25_enabled

    # Reranker (any /v1/rerank-compatible server: llama-server, vLLM, etc.)
    reranker_enabled: bool = False
    reranker_backend: str = "llama-server"  # "llama-server" or "vllm"

    @property
    def reranker_uses_chat_template(self) -> bool:
        """Whether the backend builds the Qwen3 chat template in code.

        True for `vllm` and `vllm-score`, which wrap both the query and each
        document themselves and therefore never read `reranker_query_template`
        -- `reranker_instruction` is the knob that reaches them. `llama-server`
        takes the simple template instead. Defined here rather than in
        rag/reranker.py because `_INERT_WHEN` below has to ask the same question
        and config cannot import rag.
        """
        return self.reranker_backend in {"vllm", "vllm-score"}
    reranker_url: str = "http://localhost:8081"
    reranker_model: str = "qwen3-reranker-0.6b-q8"
    reranker_query_template: str = "<Instruct>: {instruction}\n<Query>: {query}"
    reranker_top_n: int = 6
    # The ONLY candidate count in retrieval. It sizes all three places that have
    # to agree: the dense prefetch, the sparse prefetch, and the fused limit they
    # feed. They were three independent settings until 2026-10-02, at 20/20
    # against a fused 25 -- and because the fused result is drawn from the UNION
    # of the two branches, the pool silently shrank toward 20 exactly when the
    # branches agreed, which is what a working hybrid does. Sized together, the
    # union is >= the limit by construction and the number here is the number
    # retrieved. Read only when the reranker is on; with it off, search_policies
    # passes its own top_k (6), because 25 unranked chunks would not fit the
    # agent's num_ctx of 4096.
    reranker_candidates: int = 20
    # Relevance floor on the reranker's 0.0-1.0 score. 0.0 means OFF.
    # NOT a reuse of min_confidence_score: that one is cosine similarity on the
    # reranker-off path, this is a reranker relevance probability, and one knob
    # for two scales would be a latent bug.
    #
    # Measured 2026-09-25 on chatbot-test-v1, 61 DISTINCT questions through the
    # real remote stack:
    #   right document retrieved (n=59): top_score 0.8478 .. 0.9998
    #   document missed          (n=2) : 0.5519 and 0.9886
    # So 0.2 sits 4.2x below the lowest score that produced a correct retrieval,
    # and it fired on NONE of the 61. It is deliberately inert: it exists to catch
    # obviously-irrelevant questions (an earlier "can I bring penguin into office"
    # scored 0.017), not to adjudicate borderline ones.
    #
    # Do not raise it expecting better precision. One of the two document misses
    # scored 0.9886 — a high score does not mean the right document was found, so
    # no threshold separates hits from misses on this corpus. Raising it toward
    # 0.6-0.7 would make the single 0.5519 case escalate in code instead of by
    # model judgement (saving one LLM call) at the cost of only ~1.2-1.4x margin
    # above the lowest correct retrieval. That trade was judged not worth it.
    #
    # An earlier version of this comment cited "38 production requests, n=3
    # escalated / n=27 answered". That was wrong: those 38 requests were 5
    # distinct questions, one of them repeated 23 times during load testing.
    reranker_min_score: float = 0.2
    reranker_instruction: str = "Given an employee compliance question, retrieve the internal policy clause that answers it"

    # Hybrid search
    bm25_enabled: bool = True
    # BM25 length normalisation, passed to Qdrant in per-document `options`.
    # Qdrant's default is 256; this corpus measured 1602 chunks at mean 49.7
    # tokens (median 38, p90 109, max 350) on 2026-09-29. At 256 the term
    # (1 - b + b*dl/avg_len) stays near 0.25 for every chunk, so `b` goes inert
    # and long chunks are never penalised.
    #
    # WRITE-TIME, unlike bm25_enabled and reranker_candidates: this is
    # baked into every stored sparse vector and is inert at query time (measured:
    # the same text stored at 50 vs 256 gives 1.504788 vs 1.652097; changing it
    # on the query side alone changes nothing). Changing it means re-encoding —
    # scripts/migrate_collection.py into a fresh collection, or a full re-ingest.
    # A partial re-ingest silently mixes two normalisations, and nothing can
    # detect it: Qdrant discards collection-level BM25 config, so a stored vector
    # carries no record of what produced it.
    bm25_avg_len: float = 50.0

    # Agent
    agent_timeout: int = 120

    # Router (pre-retrieval classification)
    router_enabled: bool = True
    router_llm_model: str = ""            # classifier model override; empty -> main LLM
    router_confidence_floor: float = 0.6  # below this -> safe default IN_SCOPE

    # Chunking
    chunk_min_tokens: int = 50
    chunk_max_tokens: int = 400

    # Teams Bot
    teams_tenant_id: str = ""
    teams_client_id: str = ""
    teams_client_secret: SecretStr = SecretStr("")
    teams_refresh_token: SecretStr = SecretStr("")
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
    # stop_grace_period to 30s for exactly this reason, and says so — so the bot
    # finishes draining, saves state and exits on its own rather than being
    # SIGKILLed, which is the hard crash this whole path exists to avoid. Raising
    # this past that grace silently gives the behaviour back, and nothing fails
    # loudly when it does: change both together.
    #
    # 12, not 8: measured on the VM 2026-09-23, a warm answer takes ~6.9s end to
    # end, so an 8s drain left 1.1s of margin and timed out on its very first
    # production restart. 12s clears a warm answer comfortably while still
    # leaving room under the 30s container grace to save state and exit. A COLD
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

    # extra="ignore" is load-bearing: the deployed .env (and its untracked restore
    # backup) carries keys whose fields have been deleted, and pydantic-settings'
    # default extra="forbid" turns any one of them into a hard ValidationError on
    # import — config.py's module-level `settings = get_settings()` would crash
    # the whole app, not just this settings lookup.
    #
    # The cost is that deleting a field silently demotes its .env key to
    # decoration. MIN_CONFIDENCE_SCORE, RERANKER_QUERY_TEMPLATE, BM25_AVG_LEN and
    # RETRIEVAL_TOP_K each survived that way long enough to be tuned by hand.
    #
    # The case that settles it: PIPELINE_MODE sat in the deployed .env for 109
    # days after its field was deleted (73bb90a, 2026-06-19), found by the first
    # run of the check below on 2026-10-06. The removal spec had listed the .env
    # line explicitly and reasoned "a leftover env line is harmless -- but we
    # remove it anyway to keep config honest". Knowing was never the problem. The
    # .env is untracked and lives on a host no commit reaches, so the one step
    # that mattered was the one nothing could verify.
    #
    # unknown_env_keys() below is the counterweight: ignore the key, name it once.
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


def unknown_env_keys(env_text: str) -> list[str]:
    """Keys in `env_text` that match no Settings field, in the order they appear.

    Pure function over the file's text so it can be tested without touching disk
    or the real .env, which holds live secrets. Comparison is case-insensitive
    because pydantic-settings resolves env keys that way — flagging a lowercase
    key would send someone deleting a working line.
    """
    fields = set(Settings.model_fields)
    orphans = []
    for line in env_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key = line.split("=", 1)[0].strip()
        if key.lower() not in fields:
            orphans.append(key)
    return orphans


# Settings that exist, are read by real code, and are read only on a branch this
# deployment does not take. The third shape of dead config, after the field
# nothing reads and the .env key matching no field -- and the one neither of
# those catches, because nothing here is wrong except the pairing of a value
# with a configuration. Four have been found by hand so far; this is the table
# that finds the fifth.
#
# WRITE-TIME settings are deliberately absent. BM25_AVG_LEN is inert at query
# time and live during ingest, so its inertness is a property of which process
# is running rather than of the configuration, and a warning that fires wrongly
# during an ingest is worse than no warning at all.
_INERT_WHEN: tuple[tuple[str, Callable[["Settings"], bool], str], ...] = (
    (
        "RERANKER_QUERY_TEMPLATE",
        lambda s: s.reranker_uses_chat_template,
        "it is read only on the llama-server backend, and this one builds a "
        "Qwen3 chat template in code. RERANKER_INSTRUCTION is the knob that "
        "applies here.",
    ),
    (
        "MIN_CONFIDENCE_SCORE",
        lambda s: not s.cosine_floor_applies,
        "it is a cosine threshold, and the top score is not a cosine similarity "
        "with the reranker or BM25 on. RERANKER_MIN_SCORE is the live floor.",
    ),
)


def inert_env_keys(
    settings: "Settings",
    env_text: str = "",
    environ: Mapping[str, str] | None = None,
) -> list[str]:
    """Keys the operator has set that do nothing on `settings`, each with why.

    Two sources, because the two deployments differ and checking either alone is
    blind in the other. On a dev host pydantic-settings reads `.env` itself and
    the keys never reach os.environ; in the container `.env` does not exist at
    all and compose's `env_file:` has already injected them as real environment
    variables.

    This one works inside the container, where unknown_env_keys() structurally
    cannot. That docstring is right that there is no set of "keys the operator
    meant as settings" to enumerate there -- they arrive indistinguishable from
    PATH. The difference is that this never enumerates: it asks after a fixed,
    short list of keys by name, and nothing outside this project sets any of
    them.

    Takes `settings` rather than reading the global so a test can ask about a
    configuration the machine is not running, and `env_text` rather than a path
    for the same reason unknown_env_keys() does -- the real .env holds live
    secrets.
    """
    environ = os.environ if environ is None else environ

    set_in_file = set()
    for line in env_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        set_in_file.add(line.split("=", 1)[0].strip().upper())

    reports = []
    for key, is_inert, why in _INERT_WHEN:
        if key not in set_in_file and key not in environ:
            continue
        if is_inert(settings):
            reports.append(f"{key} is set but does nothing here: {why}")
    return reports


def _warn_about_inert(settings: "Settings") -> None:
    """Name set-but-inert settings once per process.

    NOT host-side only, unlike _warn_about_orphans -- reading `.env` covers the
    dev host and os.environ covers the container, and the deployed bot is the
    place these keys actually cost something.
    """
    try:
        env_text = Path(".env").read_text(encoding="utf-8")
    except OSError:
        env_text = ""
    for report in inert_env_keys(settings, env_text):
        logger.warning("%s", report)


def _warn_about_orphans() -> None:
    """Name dead .env keys once per process. See the model_config comment.

    HOST-SIDE ONLY, and deliberately so. `.env` is in .dockerignore and
    docker-compose-remote.yml injects it through `env_file:`, so inside the
    deployed container the file does not exist and this returns silently. That
    is the right behaviour, not a gap to patch: in the container the keys arrive
    as environment variables indistinguishable from PATH and HOSTNAME, so there
    is no set of "keys the operator meant as settings" left to compare against.

    It therefore needs both a Python environment with the deps AND `.env` on
    disk. A dev machine has both. `srv-agent-01` has neither in one place --
    everything runs in Docker, so the host has no pydantic -- and there the file
    has to be handed to a container explicitly:

        docker compose -f docker-compose-remote.yml run --rm $EVAL \
          -v /home/sa.ivanov/rag/.env:/tmp/env.check:ro \
          --entrypoint python bot -c \
          "from config import unknown_env_keys; \
           print(unknown_env_keys(open('/tmp/env.check').read()) or 'clean')"

    which is why unknown_env_keys() takes text rather than a path. Do not read
    the deployed bot's silence as a clean .env -- it has never looked.
    """
    try:
        orphans = unknown_env_keys(Path(".env").read_text(encoding="utf-8"))
    except OSError:
        return
    if orphans:
        logger.warning(
            ".env sets %d key(s) that match no setting and are being ignored: %s",
            len(orphans),
            ", ".join(orphans),
        )


@lru_cache
def get_settings() -> Settings:
    """Cached, so the orphan warning is emitted once per process, not per lookup."""
    settings = Settings()
    _warn_about_orphans()
    _warn_about_inert(settings)
    return settings


settings = get_settings()
