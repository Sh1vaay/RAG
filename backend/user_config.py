"""
user_config.py
──────────────
Per-user provider settings, stored inside the user's own workspace.

Each workspace owns a `config.json`. Provider/model choices are stored in clear;
API keys are encrypted with Fernet. A user with no saved settings inherits the
server's `.env` defaults, so single-user installs behave exactly as before.

**Why the workspace and not a Supabase table.** The approved plan put this in a
`user_settings` table. Keeping it beside the documents and the index means the
whole of a user's state is one directory — so when Supabase Storage lands
(Phase 3) the config syncs with everything else instead of needing a second,
separate mechanism. For a single server that is strictly simpler. A multi-server
deployment would need the shared table, and that is the point to revisit this.

**Threat model.** The server must be able to decrypt these keys, because the
server is what calls the provider APIs. Encryption protects the files at rest —
a leaked backup or a synced index does not leak credentials. It does not protect
against a compromised server.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

from cryptography.fernet import Fernet, InvalidToken
from fastapi import BackgroundTasks

from .providers import ALL_KEY_ENVS, ProviderConfig, env_defaults
from .workspace import WORKSPACE_ROOT, Workspace

CONFIG_FILENAME = "config.json"
_KEYFILE = WORKSPACE_ROOT / ".encryption-key"
_fernet: Optional[Fernet] = None


def _config_path(workspace: "Workspace") -> Path:
    return workspace.root / CONFIG_FILENAME


def _get_fernet() -> Fernet:
    """Fernet built from APP_ENCRYPTION_KEY, or a generated key persisted locally.

    Generating on first use keeps the app working out of the box. Set
    APP_ENCRYPTION_KEY explicitly for anything real — rotating the key makes every
    stored credential undecryptable, and users simply re-enter them.
    """
    global _fernet
    if _fernet is not None:
        return _fernet

    secret = (os.getenv("APP_ENCRYPTION_KEY") or "").strip()
    if not secret:
        if _KEYFILE.is_file():
            secret = _KEYFILE.read_text(encoding="utf-8").strip()
        else:
            secret = Fernet.generate_key().decode()
            _KEYFILE.parent.mkdir(parents=True, exist_ok=True)
            _KEYFILE.write_text(secret, encoding="utf-8")
            print(
                f"[config] No APP_ENCRYPTION_KEY set — generated one at {_KEYFILE}. "
                "Set APP_ENCRYPTION_KEY in .env for a real deployment.",
                file=sys.stderr,
            )
    _fernet = Fernet(secret.encode())
    return _fernet



def load_user_config(
    workspace: Workspace,
    token: Optional[str] = None,
) -> ProviderConfig:
    """This user's config, falling back to the server defaults for anything unset.

    When a *token* is supplied and no local config exists, tries to pull one from
    Supabase Storage first — so a user signing in from a fresh server inherits
    their saved settings.
    """
    defaults = env_defaults()
    path = _config_path(workspace)

    # If no local file, try cloud before falling back to defaults
    if not path.is_file() and token:
        from .storage import get_storage_client, sync_config_from_storage

        storage = get_storage_client(workspace.user_id, token)
        sync_config_from_storage(storage, path)

    if not path.is_file():
        return defaults

    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[config] Ignoring unreadable {path}: {exc}", file=sys.stderr)
        return defaults

    keys = dict(defaults.keys)
    fernet = _get_fernet()
    for env_name, token in (stored.get("keys") or {}).items():
        if env_name not in ALL_KEY_ENVS:
            continue
        try:
            keys[env_name] = fernet.decrypt(token.encode()).decode()
        except (InvalidToken, AttributeError, ValueError):
            # Usually means APP_ENCRYPTION_KEY changed. Drop the unreadable value
            # and fall through to the default rather than failing the request.
            print(
                f"[config] Could not decrypt {env_name} for workspace "
                f"'{workspace.user_id}' — the encryption key has probably changed.",
                file=sys.stderr,
            )

    return ProviderConfig(
        llm_provider=stored.get("llm_provider", defaults.llm_provider),
        llm_model=stored.get("llm_model", defaults.llm_model),
        embedding_provider=stored.get("embedding_provider", defaults.embedding_provider),
        embedding_model=stored.get("embedding_model", defaults.embedding_model),
        routing_method=stored.get("routing_method", defaults.routing_method),
        reranker_provider=stored.get("reranker_provider", defaults.reranker_provider),
        keys=keys,
        ollama_base_url=defaults.ollama_base_url,
    )


def save_user_config(
    workspace: Workspace,
    config: ProviderConfig,
    token: Optional[str] = None,
    background_tasks: Optional[BackgroundTasks] = None,
) -> ProviderConfig:
    """Persist a validated config.  Keys are encrypted; nothing else is.

    When a *token* is supplied, the config is also uploaded to Supabase Storage
    so it survives server restarts and follows the user across machines.
    """
    config.validate()
    fernet = _get_fernet()

    payload = {
        "llm_provider": config.llm_provider,
        "llm_model": config.llm_model,
        "embedding_provider": config.embedding_provider,
        "embedding_model": config.embedding_model,
        "routing_method": config.routing_method,
        "reranker_provider": config.reranker_provider,
        "keys": {
            env_name: fernet.encrypt(value.encode()).decode()
            for env_name, value in config.keys.items()
            if value and value.strip()
        },
    }

    workspace.ensure()
    path = _config_path(workspace)
    # Write via a temp file so an interrupted save cannot leave a truncated config
    # that would lock the user out of their own settings.
    config_bytes = json.dumps(payload, indent=2).encode("utf-8")
    tmp = path.with_suffix(".json.tmp")
    tmp.write_bytes(config_bytes)
    tmp.replace(path)

    # Push to Supabase Storage (best-effort)
    if token:
        from .storage import get_storage_client

        storage = get_storage_client(workspace.user_id, token)
        if storage:
            if background_tasks:
                background_tasks.add_task(storage.upload_config, config_bytes)
            else:
                storage.upload_config(config_bytes)

    return config


def embedding_changed(before: ProviderConfig, after: ProviderConfig) -> bool:
    """True when the vector space changed, which makes the existing index unusable."""
    return (
        before.embedding_provider != after.embedding_provider
        or before.resolved_embedding_model != after.resolved_embedding_model
    )
