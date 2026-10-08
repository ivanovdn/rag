"""Which weights an Ollama tag points at, right now.

A tag is a name, not a model. `qwen3.6:latest` on a shared host moves whenever
anyone there re-pulls it, and nothing on this side changes or logs when that
happens -- the bot keeps answering, against different weights. The digest is
the content address of what the tag currently resolves to, which makes it the
only stable thing to record about the model a measurement or a deploy ran on.
"""

import requests


def ollama_model_digest(base_url: str, model: str, timeout: float = 5.0) -> str:
    """The full digest `model` resolves to on the Ollama server at `base_url`.

    Raises on every failure -- transport, HTTP status, a malformed body, or the
    model not being on the host -- and leaves what that means to the caller. No
    near-match fallback: a digest for the wrong model would be reported with
    exactly the confidence of the right one. Matching follows Ollama's own rule
    that a name with no tag means `:latest`.
    """
    wanted = model if ":" in model.rsplit("/", 1)[-1] else f"{model}:latest"
    resp = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=timeout)
    resp.raise_for_status()
    for entry in resp.json().get("models", []):
        if wanted in (entry.get("name"), entry.get("model")):
            return entry["digest"]
    raise LookupError(f"{wanted} is not among the models on {base_url}")
