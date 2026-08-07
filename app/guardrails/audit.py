"""Lightweight guardrail observability: structured audit events + counters.

There is no metrics/tracing backend wired into this service today, so this
module provides a minimal, dependency-free implementation:

- ``record_audit_event`` writes one structured JSON log line per guardrail
  decision (never the raw text being evaluated -- callers pass only
  identifiers, actions, and severities).
- ``metrics`` is an in-process counter registry. It is intentionally simple
  (a lock-guarded dict) rather than a full metrics client so it has zero new
  dependencies; swapping in Prometheus/OpenTelemetry later only requires
  changing this module, since call sites depend only on ``metrics.increment``.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

from app.conversation.context import ConversationContextStore
from app.utils.json_logging import pretty_json

logger = logging.getLogger('app.guardrails.audit')

# Fixed set of counters called out by the guardrail audit. Pre-seeding them
# at zero means `snapshot()` always reports every known counter, even before
# it has ever fired.
_COUNTER_NAMES = (
    'input_blocked',
    'input_redacted',
    'output_redacted',
    'classifier_invoked',
    'classifier_failed',
    'grounding_retry',
    'streaming_redactions',
)


class GuardrailMetricsRegistry:
    """Thread-safe in-memory counters for guardrail activity."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, int] = dict.fromkeys(_COUNTER_NAMES, 0)

    def increment(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + amount

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counters)

    def reset(self) -> None:
        """Test-only helper to zero all counters between assertions."""
        with self._lock:
            self._counters = dict.fromkeys(_COUNTER_NAMES, 0)


# Process-wide singleton. Guardrail call sites (engine, security client/service,
# streaming wrapper, grounding validator) share this instance.
metrics = GuardrailMetricsRegistry()


@dataclass(frozen=True)
class GuardrailAuditEvent:
    guardrail_name: str
    action: str  # allow | block | redact | annotate | classify
    severity: str = 'info'  # info | low | medium | high | critical
    latency_ms: float | None = None
    user_id: str | None = None
    tenant_id: str | None = None
    detail: str | None = None


def record_audit_event(event: GuardrailAuditEvent) -> None:
    """Emit a structured audit log line. Never pass raw prompt/response text here."""

    context = ConversationContextStore.get()
    logger.info(
        'guardrail_audit_json\n%s',
        pretty_json(
            {
                'request_id': context.request_id or None,
                'conversation_id': context.conversation_id or None,
                'user_id': event.user_id,
                'tenant_id': event.tenant_id,
                'guardrail_name': event.guardrail_name,
                'action': event.action,
                'severity': event.severity,
                'latency_ms': event.latency_ms,
                'detail': event.detail,
            }
        ),
    )
