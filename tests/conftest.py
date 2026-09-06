"""Shared test fixtures and utilities."""

from __future__ import annotations
from typing import Any


class FakeResponse:
    """Universal fake HTTP response for all news collector tests."""

    def __init__(
        self,
        payload: object | None = None,
        *,
        content: bytes | None = None,
        status_code: int = 200,
        error: Any = None,
    ) -> None:
        self.content = content or b""
        self.payload = payload
        self.status_code = status_code
        self.error = error

    def raise_for_status(self) -> None:
        if self.error:
            raise self.error
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> Any:
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload
