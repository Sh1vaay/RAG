"""
workspace.py
────────────
Per-user document and index locations.

Every path the pipeline touches is derived here from a user id, so no module has
to know the on-disk layout.  Local disk is a **cache**: the durable copy lives
in Supabase Storage (when configured), and ``workspaces/`` can be deleted at any
time without data loss.

The id must always come from a verified JWT — never from client input. Callers go
through ``Workspace.for_user()``, which rejects anything that is not a plain
identifier, so a crafted id cannot escape the workspace root.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Single-user installs (and every request before auth lands) share this id, which
# keeps the app working unchanged until Phase 1 supplies a real one.
DEFAULT_USER_ID = "local"

# Supabase user ids are UUIDs; allow the same shape plus the local sentinel.
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

WORKSPACE_ROOT = Path(os.getenv("WORKSPACE_ROOT", "workspaces")).resolve()


class InvalidUserIdError(ValueError):
    """Raised when a user id could be used to escape the workspace root."""


@dataclass(frozen=True)
class Workspace:
    """Resolved locations for one user's documents and vector index."""

    user_id: str
    root: Path

    @classmethod
    def for_user(cls, user_id: str | None) -> "Workspace":
        # Only an explicit `None` means "no user supplied, use the shared workspace".
        # An empty or whitespace string is a *malformed* id and must fail loudly —
        # silently falling back would hand a caller with a broken token someone
        # else's documents once auth is wired up.
        if user_id is None:
            uid = DEFAULT_USER_ID
        else:
            uid = user_id.strip()

        if not _SAFE_ID.match(uid):
            # Defence in depth: ids come from a verified token, but a path
            # traversal here would expose every other tenant.
            raise InvalidUserIdError(f"Refusing to build a workspace for user id {uid!r}.")

        root = (WORKSPACE_ROOT / uid).resolve()
        if not str(root).startswith(str(WORKSPACE_ROOT)):
            raise InvalidUserIdError(f"Workspace for {uid!r} would escape {WORKSPACE_ROOT}.")
        return cls(user_id=uid, root=root)

    @property
    def documents_dir(self) -> Path:
        return self.root / "documents"

    @property
    def index_dir(self) -> Path:
        return self.root / "faiss_db"

    @property
    def config_path(self) -> Path:
        return self.root / "config.json"

    @property
    def builds_dir(self) -> Path:
        return self.root / "builds"

    def ensure(self) -> "Workspace":
        """Create the directories. Safe to call repeatedly."""
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.builds_dir.mkdir(parents=True, exist_ok=True)
        return self

    @property
    def has_index(self) -> bool:
        return self.index_dir.is_dir() and any(self.index_dir.iterdir())

    def staged_files(self) -> list[dict]:
        """Files awaiting ingestion, shaped for /api/status."""
        if not self.documents_dir.is_dir():
            return []
        out = []
        for entry in sorted(self.documents_dir.iterdir()):
            if not entry.is_file():
                continue
            size = entry.stat().st_size
            readable = (
                f"{size / (1024 * 1024):.2f} MB" if size > 1024 * 1024 else f"{size / 1024:.1f} KB"
            )
            out.append({"name": entry.name, "size": readable, "status": "ready"})
        return out

    def delete_document(self, filename: str) -> bool:
        """Remove a staged document from local disk.  Returns True if deleted."""
        safe = os.path.basename(filename)
        path = self.documents_dir / safe
        if path.is_file():
            path.unlink()
            return True
        return False

    def clear_index(self) -> None:
        """Drop the index so the next ingest starts clean."""
        if not self.index_dir.is_dir():
            return
        for entry in self.index_dir.iterdir():
            if entry.is_file():
                entry.unlink()

    # ── Supabase Storage sync ─────────────────────────────────────────────

    def sync_index_from_storage(self, token: Optional[str]) -> bool:
        """Pull the FAISS index from Supabase Storage into the local cache.

        Returns True if any files were downloaded.  A no-op when *token* is
        ``None`` (single-user mode) or when the index already exists locally.
        """
        if self.has_index or not token:
            return False
        from .storage import get_storage_client, sync_index_from_storage

        storage = get_storage_client(self.user_id, token)
        return sync_index_from_storage(storage, self.index_dir)

    def sync_documents_from_storage(self, token: Optional[str]) -> bool:
        """Pull documents from Supabase Storage, skipping files already on disk."""
        if not token:
            return False
        from .storage import get_storage_client, sync_documents_from_storage

        storage = get_storage_client(self.user_id, token)
        return sync_documents_from_storage(storage, self.documents_dir)

    def sync_builds_from_storage(self, token: Optional[str]) -> bool:
        """Pull build files from Supabase Storage."""
        if not token:
            return False
        from .storage import get_storage_client, sync_builds_from_storage

        storage = get_storage_client(self.user_id, token)
        return sync_builds_from_storage(storage, self.builds_dir)

    def sync_to_storage(self, token: Optional[str]) -> None:
        """Push local documents + index + config + builds to Supabase Storage."""
        if not token:
            return
        from .storage import (
            get_storage_client,
            sync_documents_to_storage,
            sync_index_to_storage,
            sync_builds_to_storage,
        )

        storage = get_storage_client(self.user_id, token)
        if storage is None:
            return
        ndocs = sync_documents_to_storage(storage, self.documents_dir)
        nidx = sync_index_to_storage(storage, self.index_dir)
        nblds = sync_builds_to_storage(storage, self.builds_dir)
        if ndocs or nidx or nblds:
            print(
                f"☁️  [Storage] Synced {ndocs} doc(s), {nidx} index file(s), and {nblds} build file(s) for "
                f"workspace '{self.user_id}'.",
                file=sys.stderr,
            )
