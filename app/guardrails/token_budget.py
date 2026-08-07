"""Token-aware input budget enforcement.

Complements (does not replace) the existing character-based
``input.max_length`` guardrail rule. Character counts are a poor proxy for
model context usage, especially for non-English text or code; this module
counts the actual components that make up a model call -- the user message,
system prompt, conversation history, and tool context -- using
``app.utils.tokens.estimate_tokens`` and enforces a configurable budget
before any model call is made.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from app.utils.tokens import estimate_tokens


@dataclass(frozen=True)
class TokenUsage:
    token_count: int
    token_limit: int
    remaining_tokens: int

    @property
    def exceeded(self) -> bool:
        return self.token_count > self.token_limit


class TokenBudgetChecker:
    def __init__(self, token_limit: int) -> None:
        self._token_limit = token_limit

    @property
    def token_limit(self) -> int:
        return self._token_limit

    def evaluate(
        self,
        *,
        user_message: str = '',
        system_prompt: str = '',
        conversation_history: Iterable[str] | None = None,
        tool_context: str = '',
    ) -> TokenUsage:
        components = [user_message, system_prompt, tool_context, *(conversation_history or [])]
        token_count = sum(estimate_tokens(component) for component in components if component)
        return TokenUsage(
            token_count=token_count,
            token_limit=self._token_limit,
            remaining_tokens=self._token_limit - token_count,
        )
