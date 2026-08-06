"""
auth.py
───────
Supabase authentication for the API.

Tokens are verified **locally** against the project's public JWKS
(`/auth/v1/.well-known/jwks.json`) rather than by calling Supabase on every
request. The project signs with ES256, so verification needs nothing secret —
only the project URL. That is why this integration works with just the
publishable key and no service-role key.

The verified `sub` claim is the only source of user identity in the system:
every workspace path and every Storage prefix derives from it, never from
anything the client sends in a body or query string.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import jwt
from dotenv import load_dotenv
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import PyJWKClient, PyJWKClientConnectionError, PyJWKClientError

# Load .env here rather than relying on another module having done it. This module
# reads its configuration at import time, and import order is decided by the
# linter's alphabetical sort — depending on `main` to have loaded the environment
# first silently disabled authentication.
load_dotenv()

# One .env serves both halves of the app, so accept the NEXT_PUBLIC_ names the
# frontend already uses and fall back to unprefixed ones for a backend-only deploy.
SUPABASE_URL = (
    os.getenv("SUPABASE_URL") or os.getenv("NEXT_PUBLIC_SUPABASE_URL") or ""
).rstrip("/")

SUPABASE_PUBLISHABLE_KEY = (
    os.getenv("SUPABASE_PUBLISHABLE_KEY")
    or os.getenv("NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY")
    or os.getenv("SUPABASE_ANON_KEY")
    or os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY")
    or ""
)

# With no Supabase project configured the API stays single-user: every request
# resolves to the shared local workspace. This keeps the CLI and existing
# single-user installs working rather than locking them out.
AUTH_ENABLED = bool(SUPABASE_URL and SUPABASE_PUBLISHABLE_KEY)

_JWKS_URL = f"{SUPABASE_URL}/auth/v1/.well-known/jwks.json" if SUPABASE_URL else ""
_jwks_client: Optional[PyJWKClient] = None

# Supabase sets `aud: "authenticated"` for signed-in users.
_EXPECTED_AUDIENCE = "authenticated"

_bearer = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class CurrentUser:
    """A verified caller. `id` is the Supabase `sub` claim."""

    id: str
    email: Optional[str] = None
    token: Optional[str] = None
    """The raw access token, so downstream Storage calls can act *as* this user
    and be governed by RLS instead of a privileged service key."""

    @property
    def is_anonymous(self) -> bool:
        return self.token is None


def _get_jwks_client() -> PyJWKClient:
    """Lazily build the JWKS client. PyJWKClient caches keys and refetches on
    an unknown `kid`, so key rotation is handled without a restart."""
    global _jwks_client
    if _jwks_client is None:
        _jwks_client = PyJWKClient(_JWKS_URL, cache_keys=True, lifespan=600)
    return _jwks_client


def verify_token(token: str) -> CurrentUser:
    """Verify a Supabase access token, or raise 401 with the reason."""
    try:
        signing_key = _get_jwks_client().get_signing_key_from_jwt(token)
        claims = jwt.decode(
            token,
            signing_key.key,
            algorithms=["ES256", "RS256"],
            audience=_EXPECTED_AUDIENCE,
            options={"require": ["exp", "sub"]},
        )
    except jwt.ExpiredSignatureError as exc:
        raise HTTPException(status_code=401, detail="Session expired. Sign in again.") from exc
    except PyJWKClientConnectionError as exc:
        # We genuinely could not reach Supabase. That is our problem, not the
        # caller's — a 401 here would bounce a validly signed-in user to the
        # login screen for an outage.
        raise HTTPException(
            status_code=503, detail=f"Could not reach Supabase to verify the session: {exc}"
        ) from exc
    except PyJWKClientError as exc:
        # No signing key matched. The token carries an unknown `kid` or none at
        # all, so it was not issued by this project: a bad credential (401), not
        # a server fault. Checked *after* the connection error above, which is a
        # subclass of this.
        raise HTTPException(
            status_code=401, detail=f"Session token was not issued by this project: {exc}"
        ) from exc
    except jwt.InvalidTokenError as exc:
        raise HTTPException(status_code=401, detail=f"Invalid session token: {exc}") from exc

    subject = claims.get("sub")
    if not subject:
        raise HTTPException(status_code=401, detail="Token carries no subject claim.")

    return CurrentUser(id=subject, email=claims.get("email"), token=token)


async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> CurrentUser:
    """FastAPI dependency resolving the caller.

    When no Supabase project is configured the API runs in single-user mode and
    every request maps to the shared local workspace.
    """
    if not AUTH_ENABLED:
        from .workspace import DEFAULT_USER_ID

        return CurrentUser(id=DEFAULT_USER_ID)

    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=401,
            detail="Not signed in.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = verify_token(credentials.credentials)
    request.state.user = user
    return user


def auth_status() -> dict:
    """Non-secret description of the auth configuration, for /api/status."""
    return {
        "auth_enabled": AUTH_ENABLED,
        "supabase_url": SUPABASE_URL or None,
    }
