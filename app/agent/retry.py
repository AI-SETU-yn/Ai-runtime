"""Orchestrator-level retry handling hooks for tool execution.

Distinct from ``app.mcp_client.utils.retry``, which retries individual MCP
transport calls (connection/timeout errors, below the tool-execution boundary).
This module wraps a full ``ToolExecutorService.execute()`` call at the
orchestrator layer (``GeneralAgent.act()``), so a retry policy can be tuned or
replaced per deployment without touching agent logic.

The default policy performs no retries (``max_attempts=1``): many tools are
non-idempotent (create/update/payment operations), so retrying automatically
just because this hook exists would risk duplicate side effects. A caller must
deliberately opt into more attempts - and should generally only do so for
tools/errors it knows are safe to retry.
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from app.tool_executor.exceptions import ToolExecutionError

logger = logging.getLogger(__name__)

OnRetryHook = Callable[[int, ToolExecutionError], None]


@dataclass(frozen=True)
class ToolRetryPolicy:
    """Bounded retry policy for orchestrator-level tool execution.

    Attributes:
        max_attempts: Total attempts including the first call. ``1`` (the
            default) disables retries entirely.
        base_delay_seconds: Delay before the second attempt; doubles each
            subsequent attempt (capped at ``max_delay_seconds``).
        max_delay_seconds: Upper bound on the backoff delay.
        jitter_seconds: Random extra delay (uniformly distributed, added on
            top of the backoff) to avoid retry storms.
        retryable_codes: ``ToolExecutionError.code`` values worth retrying.
            Errors with any other code are re-raised on the first failure.
    """

    max_attempts: int = 1
    base_delay_seconds: float = 0.2
    max_delay_seconds: float = 2.0
    jitter_seconds: float = 0.1
    retryable_codes: frozenset[str] = field(default_factory=lambda: frozenset({'TIMEOUT', 'SERVICE_UNAVAILABLE'}))

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError('max_attempts must be at least 1.')
        if self.base_delay_seconds < 0:
            raise ValueError('base_delay_seconds must be greater than or equal to 0.')
        if self.max_delay_seconds < self.base_delay_seconds:
            raise ValueError('max_delay_seconds must be greater than or equal to base_delay_seconds.')
        if self.jitter_seconds < 0:
            raise ValueError('jitter_seconds must be greater than or equal to 0.')

    def is_retryable(self, exc: ToolExecutionError) -> bool:
        return exc.code in self.retryable_codes

    def delay_seconds(self, attempt: int) -> float:
        """Backoff delay before retrying after a failed ``attempt`` (1-indexed)."""
        backoff = min(self.max_delay_seconds, self.base_delay_seconds * 2 ** (attempt - 1))
        return backoff + random.uniform(0, self.jitter_seconds)


class ToolRetryHandler:
    """Runs a tool-execution operation under a ``ToolRetryPolicy``.

    Injected into ``GeneralAgent`` as an optional collaborator; the default
    handler (default policy, ``max_attempts=1``) is behaviorally identical to
    calling the operation directly with no retry logic at all.
    """

    def __init__(self, policy: ToolRetryPolicy | None = None) -> None:
        self._policy = policy or ToolRetryPolicy()

    @property
    def policy(self) -> ToolRetryPolicy:
        return self._policy

    async def run(
        self,
        operation: Callable[[], Awaitable[dict[str, object]]],
        *,
        on_retry: OnRetryHook | None = None,
    ) -> dict[str, object]:
        """Execute ``operation``, retrying on retryable ``ToolExecutionError``s.

        Re-raises the last ``ToolExecutionError`` once attempts are exhausted or
        the error's code isn't in ``policy.retryable_codes`` - callers keep their
        existing single-attempt error handling unchanged.
        """
        last_error: ToolExecutionError | None = None
        for attempt in range(1, self._policy.max_attempts + 1):
            try:
                return await operation()
            except ToolExecutionError as exc:
                last_error = exc
                if attempt >= self._policy.max_attempts or not self._policy.is_retryable(exc):
                    raise
                if on_retry is not None:
                    on_retry(attempt, exc)
                await asyncio.sleep(self._policy.delay_seconds(attempt))
        assert last_error is not None  # pragma: no cover - loop always returns or raises
        raise last_error
