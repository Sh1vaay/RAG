"""
providers.py
────────────
Constructs the chat model and the embedding model from an **explicit** config.

Nothing here reads the environment at call time. That is deliberate: the previous
version resolved providers from `os.environ`, which is process-wide, so one signed-in
user changing provider changed it for every other user of the server. Configuration
now travels as a `ProviderConfig` value, which is what makes per-user settings
possible at all.

    LLM providers        ollama | openai | anthropic | gemini | grok | cohere
    Embedding providers  ollama | openai | gemini | cohere

The two are chosen independently because **Anthropic and xAI/Grok publish no
embeddings API** — a RAG pipeline always needs embeddings, so those two are
chat-only and must be paired with another provider for vectors.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field, replace
from typing import Any, Optional

# model: default model id. key: required env var, or None when the provider is local.
CHAT_PROVIDERS = {
    "ollama": {"model": "llama3.2:1b", "key": None},
    "openai": {"model": "gpt-4o-mini", "key": "OPENAI_API_KEY"},
    "anthropic": {"model": "claude-opus-5", "key": "ANTHROPIC_API_KEY"},
    "gemini": {"model": "gemini-2.5-flash", "key": "GOOGLE_API_KEY"},
    "grok": {"model": "grok-4", "key": "XAI_API_KEY"},
    "cohere": {"model": "command-a-03-2025", "key": "COHERE_API_KEY"},
}

EMBEDDING_PROVIDERS = {
    "ollama": {"model": "nomic-embed-text", "key": None},
    "openai": {"model": "text-embedding-3-small", "key": "OPENAI_API_KEY"},
    "gemini": {"model": "text-embedding-004", "key": "GOOGLE_API_KEY"},
    "cohere": {"model": "embed-english-v3.0", "key": "COHERE_API_KEY"},
}

# Chat-only vendors, listed so the error can explain *why* rather than just refusing.
NO_EMBEDDING_API = {"anthropic", "grok"}

# Every credential the app can hold, so callers never hard-code the list.
ALL_KEY_ENVS = sorted(
    {spec["key"] for spec in CHAT_PROVIDERS.values() if spec["key"]}
    | {spec["key"] for spec in EMBEDDING_PROVIDERS.values() if spec["key"]}
)


class ConfigError(ValueError):
    """Raised for an unusable provider selection — surfaced to the user verbatim."""


@dataclass(frozen=True)
class ProviderConfig:
    """A complete, self-contained description of how to build one user's models."""

    llm_provider: str = "ollama"
    llm_model: str = ""
    embedding_provider: str = "ollama"
    embedding_model: str = ""
    routing_method: str = "semantic"
    reranker_provider: str = "flashrank"
    keys: dict[str, str] = field(default_factory=dict)
    ollama_base_url: str = "http://localhost:11434"

    @property
    def resolved_llm_model(self) -> str:
        return self.llm_model or CHAT_PROVIDERS[self.llm_provider]["model"]

    @property
    def resolved_embedding_model(self) -> str:
        return self.embedding_model or EMBEDDING_PROVIDERS[self.embedding_provider]["model"]

    def key_for(self, provider: str, registry: dict) -> Optional[str]:
        env_name = registry[provider]["key"]
        return self.keys.get(env_name) if env_name else None

    def validate(self) -> "ProviderConfig":
        """Reject an unusable selection before anything tries to build a client."""
        if self.llm_provider not in CHAT_PROVIDERS:
            raise ConfigError(
                f"Unknown LLM provider '{self.llm_provider}'. "
                f"Choose one of: {', '.join(sorted(CHAT_PROVIDERS))}."
            )
        if self.embedding_provider not in EMBEDDING_PROVIDERS:
            if self.embedding_provider in NO_EMBEDDING_API:
                raise ConfigError(
                    f"'{self.embedding_provider}' publishes no embeddings API, so it can "
                    f"serve the LLM only. Choose an embedding provider from: "
                    f"{', '.join(sorted(EMBEDDING_PROVIDERS))}."
                )
            raise ConfigError(
                f"Unknown embedding provider '{self.embedding_provider}'. "
                f"Choose one of: {', '.join(sorted(EMBEDDING_PROVIDERS))}."
            )
        if self.routing_method not in ("semantic", "llm"):
            raise ConfigError("routing_method must be 'semantic' or 'llm'.")
        if self.reranker_provider not in ("flashrank", "cohere"):
            raise ConfigError("reranker_provider must be 'flashrank' or 'cohere'.")

        for provider, registry, label in (
            (self.llm_provider, CHAT_PROVIDERS, "LLM"),
            (self.embedding_provider, EMBEDDING_PROVIDERS, "embedding"),
        ):
            env_name = registry[provider]["key"]
            if env_name and not (self.keys.get(env_name) or "").strip():
                raise ConfigError(
                    f"{env_name} is required for the {label} provider '{provider}' "
                    f"but has not been set."
                )
        return self

    def summary(self) -> dict:
        """Non-secret view for /api/status. Never leaks a key."""
        return {
            "llm_provider": self.llm_provider,
            "llm_model": self.resolved_llm_model,
            "llm_key_configured": _has_key(self, self.llm_provider, CHAT_PROVIDERS),
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.resolved_embedding_model,
            "embedding_key_configured": _has_key(
                self, self.embedding_provider, EMBEDDING_PROVIDERS
            ),
            "routing_method": self.routing_method,
            "reranker_provider": self.reranker_provider,
        }

    def describe(self) -> str:
        return (
            f"LLM: {self.llm_provider}/{self.resolved_llm_model}  |  "
            f"Embeddings: {self.embedding_provider}/{self.resolved_embedding_model}"
        )

    def with_updates(self, **changes) -> "ProviderConfig":
        """Copy with non-None changes applied. Keys merge rather than replace, so an
        omitted credential keeps its stored value instead of being wiped."""
        merged_keys = dict(self.keys)
        for env_name, value in (changes.pop("keys", None) or {}).items():
            if value and value.strip():
                merged_keys[env_name] = value.strip()
        clean = {k: v for k, v in changes.items() if v is not None}
        return replace(self, keys=merged_keys, **clean)


def _has_key(config: ProviderConfig, provider: str, registry: dict) -> bool:
    env_name = registry[provider]["key"]
    if env_name is None:
        return True
    return bool((config.keys.get(env_name) or "").strip())


def env_defaults() -> ProviderConfig:
    """Server-wide defaults from .env — the starting point for a user with no saved
    settings, and the whole configuration in single-user mode."""
    keys = {}
    for name in ALL_KEY_ENVS:
        value = (os.getenv(name) or "").strip()
        if value and not value.startswith(("your_", "paste_")):
            keys[name] = value

    return ProviderConfig(
        llm_provider=(os.getenv("LLM_PROVIDER") or "ollama").strip().lower(),
        llm_model=(os.getenv("LLM_MODEL") or "").strip(),
        embedding_provider=(os.getenv("EMBEDDING_PROVIDER") or "ollama").strip().lower(),
        embedding_model=(os.getenv("EMBEDDING_MODEL") or "").strip(),
        routing_method=(os.getenv("ROUTING_METHOD") or "semantic").strip().lower(),
        reranker_provider=(os.getenv("RERANKER_PROVIDER") or "flashrank").strip().lower(),
        keys=keys,
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )


def check_ollama_available(base_url: str, *models: str) -> None:
    """Verify the local Ollama daemon is up and the named models are pulled."""
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=5) as resp:
            installed = {m["name"] for m in json.loads(resp.read()).get("models", [])}
    except (urllib.error.URLError, OSError) as exc:
        raise ConfigError(
            f"Cannot reach the Ollama daemon at {base_url} ({exc}). "
            "Start it with 'ollama serve' (or launch the Ollama app) and try again."
        ) from exc

    for model in models:
        # Ollama reports bare names as 'name:latest', so accept either form.
        if model not in installed and f"{model}:latest" not in installed:
            raise ConfigError(
                f"Ollama model '{model}' is not installed. Pull it with: ollama pull {model}\n"
                f"Currently installed: {', '.join(sorted(installed)) or '(none)'}"
            )


def get_llm(config: ProviderConfig) -> Any:
    """Build the chat model this config describes."""
    config.validate()
    provider, model = config.llm_provider, config.resolved_llm_model
    key = config.key_for(provider, CHAT_PROVIDERS)

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        check_ollama_available(config.ollama_base_url, model)
        return ChatOllama(model=model, temperature=0, base_url=config.ollama_base_url)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, temperature=0, api_key=key)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        # Claude Opus 5 (and its siblings) reject temperature/top_p/top_k with a 400,
        # so it is omitted rather than set to 0. Thinking is on by default and shares
        # the max_tokens budget with the reply, hence the generous ceiling.
        kwargs: dict = {"model": model, "max_tokens": 8192, "api_key": key}
        if not model.startswith(("claude-opus-5", "claude-fable-5", "claude-sonnet-5")):
            kwargs["temperature"] = 0
        return ChatAnthropic(**kwargs)

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model, temperature=0, google_api_key=key)

    if provider == "grok":
        from langchain_xai import ChatXAI

        return ChatXAI(model=model, temperature=0, api_key=key)

    from langchain_cohere import ChatCohere

    return ChatCohere(model=model, temperature=0, cohere_api_key=key)


def get_embeddings(config: ProviderConfig) -> Any:
    """Build the embedding model this config describes."""
    config.validate()
    provider, model = config.embedding_provider, config.resolved_embedding_model
    key = config.key_for(provider, EMBEDDING_PROVIDERS)

    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        check_ollama_available(config.ollama_base_url, model)
        return OllamaEmbeddings(model=model, base_url=config.ollama_base_url)

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=model, api_key=key)

    if provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(model=model, google_api_key=key)

    from langchain_cohere import CohereEmbeddings

    return CohereEmbeddings(model=model, cohere_api_key=key)


_FLASHRANK: Any = None


def _shared_flashrank() -> Any:
    """One Flashrank instance for the whole process.

    The cross-encoder is a ~100MB ONNX model and is stateless per query, so a
    second user costs nothing rather than another copy of the model.
    """
    global _FLASHRANK
    if _FLASHRANK is None:
        from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank

        print("Initializing local Cross-Encoder reranking (Flashrank)...")
        _FLASHRANK = FlashrankRerank(top_n=3)
    return _FLASHRANK


def get_reranker(config: ProviderConfig) -> Any:
    """Reranker for this config, reusing the local cross-encoder across users."""
    if config.reranker_provider == "cohere":
        key = (config.keys.get("COHERE_API_KEY") or "").strip()
        if key:
            from langchain_cohere import CohereRerank

            print("Initializing cloud-based Cohere reranking...")
            return CohereRerank(top_n=3, cohere_api_key=key)
        print(
            "[WARNING] COHERE_API_KEY is not set. Falling back to local Flashrank.",
            file=sys.stderr,
        )
    return _shared_flashrank()


def provider_catalog(config: ProviderConfig) -> dict:
    """Registries shaped for the dashboard's dropdowns, scoped to one user's keys.

    `requires_key` is a property of the provider; `key_configured` is a property of
    this user's saved credentials. The dashboard uses the pair to show only the
    credential fields the current selection actually needs.
    """

    def entries(registry: dict) -> list[dict]:
        return [
            {
                "id": name,
                "default_model": spec["model"],
                "key_env": spec["key"],
                "requires_key": spec["key"] is not None,
                "key_configured": spec["key"] is None
                or bool((config.keys.get(spec["key"]) or "").strip()),
            }
            for name, spec in registry.items()
        ]

    return {
        "chat": entries(CHAT_PROVIDERS),
        "embedding": entries(EMBEDDING_PROVIDERS),
        "chat_only": sorted(NO_EMBEDDING_API),
    }
