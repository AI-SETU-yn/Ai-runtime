"""Bounded retry with exponential backoff and jitter."""
import asyncio
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from ..exceptions import MCPConnectionError, MCPTimeoutError, ServerUnavailableError


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    base_delay_seconds: float = 0.1
    max_delay_seconds: float = 2.0
    jitter_seconds: float = 0.1

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.base_delay_seconds < 0:
            raise ValueError("base_delay_seconds must be greater than or equal to 0")
        if self.max_delay_seconds < self.base_delay_seconds:
            raise ValueError("max_delay_seconds must be greater than or equal to base_delay_seconds")
        if self.jitter_seconds < 0:
            raise ValueError("jitter_seconds must be greater than or equal to 0")


async def retry_async(operation: Callable[[], Awaitable[object]], policy: RetryPolicy) -> object:
    retryable = (MCPConnectionError, MCPTimeoutError, ServerUnavailableError)
    last_error: Exception | None = None
    for attempt in range(1, policy.max_attempts + 1):
        try:
            return await operation()
        except retryable as exc:
            last_error = exc
            if attempt == policy.max_attempts:
                raise
            delay = min(policy.max_delay_seconds, policy.base_delay_seconds * 2 ** (attempt - 1)) + random.uniform(0, policy.jitter_seconds)
            await asyncio.sleep(delay)
    if last_error is not None:
        raise last_error
    raise MCPConnectionError("Retry policy prevented MCP operation execution")
