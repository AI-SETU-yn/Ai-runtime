"""Streaming-safe output guardrails.

The AI Runtime's `/chat` endpoint is non-streaming today (see
`app.services.chat_service.ChatService.chat`), so nothing currently calls
this module in production. It exists so that when a streaming endpoint is
added, output redaction does not have to be re-designed or bolted on as an
afterthought -- the same `GuardrailEngine.enforce_output` rules (secrets,
emails, phone numbers, ...) apply to streamed text exactly as they do to a
full response.

Design
------
A secret or PII value can be split across two chunks by the upstream model
(e.g. an email address emitted token-by-token). Redacting each chunk in
isolation would miss that. Buffering the entire response defeats the point
of streaming. `StreamingGuardrail` takes the middle path: it keeps a small
trailing "carry" buffer (`tail_chars`) of the most recently seen text,
re-evaluates guardrail rules over `carry + new_chunk` on every `feed()`
call, and only releases text up to the last `tail_chars` characters -- the
tail stays held back in case a pattern is still being written and completes
on the *next* chunk. `flush()` releases whatever tail remains at stream end.

`tail_chars` defaults to 256, comfortably longer than any single pattern
this engine currently matches (bearer tokens, emails, phone numbers, JWTs),
so a match is never split across a release boundary in practice.

SSE framing
-----------
This module operates on plain text deltas, not raw SSE bytes. That is
deliberate: guardrail redaction belongs at the token/text-delta level,
*before* a caller wraps each delta in `data: ...\n\n` framing (this mirrors
how model-gateway's own LiteLLM streaming path yields text deltas that a
transport layer frames separately -- see
`model-gateway/app/services/litellm_service.py`). A future SSE endpoint
should call `guard_text_stream(engine, text_delta_iterator)` and frame each
yielded chunk as its own SSE `data:` event; guardrail logic never needs to
parse or reconstruct SSE control lines that way, so stream framing is
preserved by construction rather than by re-implementing an SSE parser here.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from app.guardrails.audit import GuardrailAuditEvent, metrics, record_audit_event
from app.guardrails.engine import GuardrailEngine

_DEFAULT_TAIL_CHARS = 256


class StreamingGuardrail:
    """Incremental output-guardrail filter for chunked/streamed model responses."""

    def __init__(self, engine: GuardrailEngine, *, tail_chars: int = _DEFAULT_TAIL_CHARS) -> None:
        self._engine = engine
        self._tail_chars = max(tail_chars, 0)
        self._carry = ''

    def feed(self, chunk: str) -> str:
        """Feed the next chunk. Returns the text that is now safe to emit (may be empty)."""

        if not chunk:
            return ''
        window = self._carry + chunk
        result = self._engine.enforce_output(window)
        if result.redacted:
            metrics.increment('streaming_redactions')
            record_audit_event(
                GuardrailAuditEvent(
                    guardrail_name='streaming_output_guardrail',
                    action='redact',
                    severity='medium',
                    detail=f'{len(result.triggered)} rule(s) redacted in stream chunk',
                )
            )
        text = result.final_text
        if len(text) <= self._tail_chars:
            self._carry = text
            return ''
        release, self._carry = text[: -self._tail_chars], text[-self._tail_chars :]
        return release

    def flush(self) -> str:
        """Release and clear whatever tail remains. Call once after the source is exhausted."""

        remainder, self._carry = self._carry, ''
        return remainder

    def reset(self) -> None:
        self._carry = ''


async def guard_text_stream(
    engine: GuardrailEngine,
    source: AsyncIterator[str],
    *,
    tail_chars: int = _DEFAULT_TAIL_CHARS,
) -> AsyncIterator[str]:
    """Wrap an async text-chunk iterator with output guardrails.

    Yields only the text that is safe to emit -- callers apply their own
    transport framing (SSE, chunked HTTP, websocket, ...) around each
    yielded piece. See the module docstring for SSE integration guidance.
    """

    guardrail = StreamingGuardrail(engine, tail_chars=tail_chars)
    async for chunk in source:
        released = guardrail.feed(chunk)
        if released:
            yield released
    tail = guardrail.flush()
    if tail:
        yield tail
