from __future__ import annotations

import logging
from time import perf_counter

from app.exceptions.errors import GuardrailViolationError
from app.guardrails import GuardrailEvaluationResult
from app.guardrails.audit import GuardrailAuditEvent, metrics, record_audit_event
from app.security.classifier import SecurityClassifier
from app.security.models import SecurityClassificationResult, SecurityClassifierConfig, SecurityDecision

logger = logging.getLogger(__name__)


class SecurityClassificationService:
    def __init__(self, classifier: SecurityClassifier, config: SecurityClassifierConfig) -> None:
        self._classifier = classifier
        self._config = config

    @property
    def config(self) -> SecurityClassifierConfig:
        return self._config

    def should_defer_input_blocks(self) -> bool:
        return self._config.enabled and self._config.mode == 'hybrid'

    async def classify_if_needed(
        self,
        *,
        message: str,
        guardrail_result: GuardrailEvaluationResult,
    ) -> SecurityClassificationResult:
        if not self._config.enabled:
            return SecurityClassificationResult(executed=False)

        trigger = self._trigger_reason(guardrail_result)
        if trigger is None:
            return SecurityClassificationResult(executed=False)

        logger.info(
            'security_classifier_invoked trigger=%s input_rule_hits=%s',
            trigger,
            len(guardrail_result.triggered),
        )
        metrics.increment('classifier_invoked')
        started = perf_counter()
        decision = await self._classifier.classify(message)
        latency_ms = round((perf_counter() - started) * 1000, 2)
        normalized = self._normalize_decision(decision)
        logger.info(
            'security_classifier_decision safe=%s category=%s confidence=%s trigger=%s',
            normalized.safe,
            normalized.category,
            normalized.confidence,
            trigger,
        )
        record_audit_event(
            GuardrailAuditEvent(
                guardrail_name='security_classifier',
                action='allow' if normalized.safe else 'block',
                severity='info' if normalized.safe else 'high',
                latency_ms=latency_ms,
                detail=f'trigger={trigger} category={normalized.category.value}',
            )
        )
        return SecurityClassificationResult(executed=True, triggered_by=trigger, decision=normalized)

    def ensure_safe(self, result: SecurityClassificationResult) -> None:
        if not result.blocked or result.decision is None:
            return
        raise GuardrailViolationError(
            result.decision.reason or 'Request blocked by AI security classifier.',
            stage='input',
            rule_id=f'security_classifier.{result.decision.category.value.lower()}',
            tags=['security-classifier', result.decision.category.value],
        )

    def _trigger_reason(self, guardrail_result: GuardrailEvaluationResult) -> str | None:
        if self._config.always_run:
            return 'always_run'
        suspicious_tags = set(self._config.suspicious_tags)
        for decision in guardrail_result.triggered:
            if suspicious_tags.intersection(decision.tags):
                return 'suspicious_rule'
        return None

    def _normalize_decision(self, decision: SecurityDecision) -> SecurityDecision:
        if decision.safe:
            return decision
        threshold = self._config.confidence_threshold
        if decision.confidence >= threshold:
            return decision
        return decision.model_copy(update={'safe': True, 'reason': f'Below confidence threshold {threshold:.2f}: {decision.reason}'})