from __future__ import annotations

import logging
import re
from collections.abc import Iterable

from app.exceptions.errors import GuardrailViolationError
from app.guardrails.audit import GuardrailAuditEvent, metrics, record_audit_event
from app.guardrails.models import (
    GuardrailDecision,
    GuardrailEvaluationResult,
    GuardrailRule,
    GuardrailsConfig,
)
from app.guardrails.token_budget import TokenBudgetChecker, TokenUsage

_FLAG_MAP = {
    'IGNORECASE': re.IGNORECASE,
    'MULTILINE': re.MULTILINE,
    'DOTALL': re.DOTALL,
}

logger = logging.getLogger(__name__)

# Rules that carry phone-shaped candidates through the shared context-aware
# assessment in `_apply_phone_candidate_rule` (false-positive suppression for
# dates, reference IDs, academic years, etc.), regardless of whether the
# configured action is `redact` (output) or `annotate` (input).
_PHONE_CANDIDATE_RULE_IDS = frozenset({'output.phone_redaction', 'input.pii_phone'})
_PHONE_CANDIDATE_PATTERN = re.compile(
    r'(?<![A-Za-z0-9])(?:\+?\d(?:[\d\s().-]{8,}\d)|\(\d{3}\)\s*\d{3}[-.\s]?\d{4})(?![A-Za-z0-9-])'
)

_COUNTER_BY_STAGE_ACTION = {
    ('input', 'block'): 'input_blocked',
    ('input', 'redact'): 'input_redacted',
    ('output', 'redact'): 'output_redacted',
}

_SEVERITY_BY_ACTION = {
    'block': 'high',
    'redact': 'medium',
    'annotate': 'low',
    'allow': 'info',
}


class GuardrailEngine:
    def __init__(self, config: GuardrailsConfig) -> None:
        self._config = config
        self._token_budget_checker = TokenBudgetChecker(config.token_budget.limit)

    @property
    def config(self) -> GuardrailsConfig:
        return self._config

    def enforce_input(
        self,
        message: str,
        *,
        defer_block_tags: Iterable[str] | None = None,
    ) -> GuardrailEvaluationResult:
        return self._evaluate('input', message, defer_block_tags=set(defer_block_tags or []))

    def enforce_output(self, answer: str) -> GuardrailEvaluationResult:
        return self._evaluate('output', answer, defer_block_tags=set())

    def check_token_budget(
        self,
        *,
        user_message: str = '',
        system_prompt: str = '',
        conversation_history: Iterable[str] | None = None,
        tool_context: str = '',
    ) -> TokenUsage:
        """Enforce the token budget before any model call. Raises on overflow.

        Runs alongside (not instead of) the existing `input.max_length`
        character-count rule -- see `TokenBudgetSettings` docstring.
        """
        usage = self._token_budget_checker.evaluate(
            user_message=user_message,
            system_prompt=system_prompt,
            conversation_history=conversation_history,
            tool_context=tool_context,
        )
        if self._config.token_budget.enabled and usage.exceeded:
            metrics.increment('input_blocked')
            record_audit_event(
                GuardrailAuditEvent(
                    guardrail_name='input.max_tokens',
                    action='block',
                    severity='high',
                    detail=f'token_count={usage.token_count} token_limit={usage.token_limit}',
                )
            )
            raise GuardrailViolationError(
                f'Input exceeds the maximum allowed token budget ({usage.token_count}/{usage.token_limit} tokens).',
                stage='input',
                rule_id='input.max_tokens',
                tags=['availability', 'abuse-prevention', 'token-budget'],
            )
        return usage

    def _evaluate(self, stage: str, text: str, *, defer_block_tags: set[str]) -> GuardrailEvaluationResult:
        rules = getattr(self._config, stage).rules
        current_text = text
        triggered: list[GuardrailDecision] = []

        for rule in rules:
            if not rule.enabled:
                continue
            current_text, decision = self._apply_rule(rule, current_text, defer_block_tags=defer_block_tags)
            if decision is None:
                continue
            triggered.append(decision)
            self._record_metrics_and_audit(stage, decision)
            if rule.action == 'block' and not decision.deferred:
                raise GuardrailViolationError(
                    decision.message or f'{stage.title()} blocked by guardrail policy.',
                    stage=stage,
                    rule_id=rule.id,
                    tags=rule.tags,
                )

        return GuardrailEvaluationResult(
            stage=stage,
            original_text=text,
            final_text=current_text,
            blocked=False,
            triggered=triggered,
        )

    @staticmethod
    def _record_metrics_and_audit(stage: str, decision: GuardrailDecision) -> None:
        effective_action = 'block' if decision.action == 'block' and not decision.deferred else decision.action
        counter_name = _COUNTER_BY_STAGE_ACTION.get((stage, effective_action))
        if counter_name:
            metrics.increment(counter_name)
        record_audit_event(
            GuardrailAuditEvent(
                guardrail_name=decision.rule_id,
                action=effective_action,
                severity=_SEVERITY_BY_ACTION.get(effective_action, 'info'),
                detail=f'deferred={decision.deferred}' if decision.deferred else None,
            )
        )

    def _apply_rule(
        self,
        rule: GuardrailRule,
        text: str,
        *,
        defer_block_tags: set[str],
    ) -> tuple[str, GuardrailDecision | None]:
        if rule.type == 'max_length':
            return self._apply_max_length(rule, text, defer_block_tags=defer_block_tags)
        if rule.type == 'keyword':
            return self._apply_keyword(rule, text, defer_block_tags=defer_block_tags)
        if rule.type == 'regex':
            return self._apply_regex(rule, text, defer_block_tags=defer_block_tags)
        return text, None

    def _apply_max_length(
        self,
        rule: GuardrailRule,
        text: str,
        *,
        defer_block_tags: set[str],
    ) -> tuple[str, GuardrailDecision | None]:
        if rule.max_length is None or len(text) <= rule.max_length:
            return text, None
        return text, self._decision(rule, defer_block_tags=defer_block_tags)

    def _apply_keyword(
        self,
        rule: GuardrailRule,
        text: str,
        *,
        defer_block_tags: set[str],
    ) -> tuple[str, GuardrailDecision | None]:
        lowered_text = text.lower()
        for keyword in rule.keywords:
            if keyword.lower() in lowered_text:
                if rule.action == 'redact':
                    redacted = re.sub(re.escape(keyword), rule.replacement, text, flags=re.IGNORECASE)
                    return redacted, self._decision(rule, defer_block_tags=defer_block_tags)
                return text, self._decision(rule, defer_block_tags=defer_block_tags)
        return text, None

    def _apply_regex(
        self,
        rule: GuardrailRule,
        text: str,
        *,
        defer_block_tags: set[str],
    ) -> tuple[str, GuardrailDecision | None]:
        if rule.id in _PHONE_CANDIDATE_RULE_IDS:
            return self._apply_phone_candidate_rule(rule, text, defer_block_tags=defer_block_tags)
        flags = self._resolve_flags(rule.flags)
        matched = False
        updated = text
        for pattern in rule.patterns:
            compiled = re.compile(pattern, flags)
            if rule.action == 'redact':
                replaced, count = compiled.subn(rule.replacement, updated)
                if count:
                    updated = replaced
                    matched = True
            elif compiled.search(updated):
                matched = True
                break
        if not matched:
            return text, None
        return updated, self._decision(rule, defer_block_tags=defer_block_tags)

    def _apply_phone_candidate_rule(
        self,
        rule: GuardrailRule,
        text: str,
        *,
        defer_block_tags: set[str],
    ) -> tuple[str, GuardrailDecision | None]:
        """Shared phone-candidate assessment for redact (output) and annotate (input) rules.

        Both rule ids reuse the same context-aware false-positive suppression
        (dates, reference IDs, academic years, ...). Only `action == 'redact'`
        rewrites text; `annotate` flags a decision without altering the text.
        Logging intentionally never includes the matched value itself.
        """
        matches = list(_PHONE_CANDIDATE_PATTERN.finditer(text))
        if not matches:
            return text, None

        should_redact_text = rule.action == 'redact'
        flagged_any = False
        updated_parts: list[str] = []
        last_index = 0

        for match in matches:
            start, end = match.span()
            candidate = match.group(0)
            updated_parts.append(text[last_index:start])
            should_flag, reason, confidence = self._assess_phone_candidate(rule, text, match)
            if should_flag:
                flagged_any = True
                updated_parts.append(rule.replacement if should_redact_text else candidate)
                logger.info(
                    'phone_candidate_flagged rule_id=%s action=%s reason=%s confidence=%.2f match_length=%d',
                    rule.id,
                    rule.action,
                    reason,
                    confidence,
                    len(candidate),
                )
            else:
                updated_parts.append(candidate)
                logger.info(
                    'phone_candidate_skipped rule_id=%s reason=%s confidence=%.2f match_length=%d',
                    rule.id,
                    reason,
                    confidence,
                    len(candidate),
                )
            last_index = end

        updated_parts.append(text[last_index:])
        updated = ''.join(updated_parts)
        if not flagged_any:
            return text, None
        return updated, self._decision(rule, defer_block_tags=defer_block_tags)

    def _assess_phone_candidate(
        self,
        rule: GuardrailRule,
        full_text: str,
        match: re.Match[str],
    ) -> tuple[bool, str, float]:
        candidate = match.group(0).strip()
        normalized = re.sub(r'\s+', ' ', candidate)
        digits_only = re.sub(r'\D', '', candidate)

        if len(digits_only) < 10:
            return False, 'insufficient_digits', 0.05
        if self._has_exempt_context(rule, full_text, match.start()):
            return False, 'context_exempt', 0.10
        if self._matches_exempt_pattern(rule, normalized):
            return False, 'pattern_exempt', 0.10
        if not self._looks_like_phone_number(candidate, digits_only):
            return False, 'invalid_phone_shape', 0.25
        return True, 'valid_phone_number', 0.98

    @staticmethod
    def _normalize_context_key(value: str) -> str:
        return re.sub(r'[^A-Za-z0-9]', '', value).lower()

    @classmethod
    def _has_exempt_context(cls, rule: GuardrailRule, full_text: str, match_start: int) -> bool:
        exempt_keys = {cls._normalize_context_key(key) for key in rule.exempt_context_keys}
        if not exempt_keys:
            return False
        line_start = full_text.rfind('\n', 0, match_start) + 1
        line_end = full_text.find('\n', match_start)
        if line_end == -1:
            line_end = len(full_text)
        line = full_text[line_start:line_end]
        prefix = line[: max(0, match_start - line_start)]
        field_match = re.search(r'([A-Za-z][A-Za-z0-9_]*)\s*:\s*$', prefix)
        if not field_match:
            return False
        field_name = cls._normalize_context_key(field_match.group(1))
        return any(field_name == key or field_name.endswith(key) for key in exempt_keys)

    @staticmethod
    def _matches_exempt_pattern(rule: GuardrailRule, value: str) -> bool:
        return any(re.fullmatch(pattern, value, flags=re.IGNORECASE) for pattern in rule.exempt_patterns)

    @staticmethod
    def _looks_like_phone_number(candidate: str, digits_only: str) -> bool:
        if len(digits_only) < 10 or len(digits_only) > 15:
            return False
        separator_count = sum(candidate.count(sep) for sep in (' ', '-', '.', '(', ')'))
        if separator_count == 0:
            return len(digits_only) == 10
        return bool(re.search(r'[\s().-]', candidate))

    @staticmethod
    def _resolve_flags(flag_names: list[str]) -> int:
        flags = 0
        for flag_name in flag_names:
            flags |= _FLAG_MAP.get(flag_name.upper(), 0)
        return flags

    @staticmethod
    def _should_defer(rule: GuardrailRule, defer_block_tags: set[str]) -> bool:
        return rule.action == 'block' and bool(defer_block_tags.intersection(rule.tags))

    @classmethod
    def _decision(cls, rule: GuardrailRule, *, defer_block_tags: set[str]) -> GuardrailDecision:
        return GuardrailDecision(
            rule_id=rule.id,
            action=rule.action,
            target=rule.target,
            message=rule.message,
            tags=rule.tags,
            deferred=cls._should_defer(rule, defer_block_tags),
        )
