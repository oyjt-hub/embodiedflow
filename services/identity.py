"""Request-scoped local user identity for task attribution."""
from __future__ import annotations

from contextvars import ContextVar, Token

_actor: ContextVar[str | None] = ContextVar("embodiedflow_actor", default=None)


def set_actor(name: str | None) -> Token:
    return _actor.set(name)


def reset_actor(token: Token) -> None:
    _actor.reset(token)


def current_actor() -> str | None:
    return _actor.get()
