from __future__ import annotations

import re

from app.exceptions.errors import GuardrailViolationError
from app.guardrails.models import (
    GuardrailDecision,
    GuardrailEvaluationResult,
    GuardrailRule,
    GuardrailsConfig,
)

_FLAG_MAP = {
    'IGNORECASE': re.IGNORECASE,
    'MULTILINE': re.MULTILINE,
    'DOTALL': re.DOTALL,
}


class GuardrailEngine:
    def __init__(self, config: GuardrailsConfig) -> None:
        self._config = config

    def enforce_input(self, message: str) -> GuardrailEvaluationResult:
        return self._evaluate('input', message)

    def enforce_output(self, answer: str) -> GuardrailEvaluationResult:
        return self._evaluate('output', answer)

    def _evaluate(self, stage: str, text: str) -> GuardrailEvaluationResult:
        rules = getattr(self._config, stage).rules
        current_text = text
        triggered: list[GuardrailDecision] = []

        for rule in rules:
            if not rule.enabled:
                continue
            current_text, decision = self._apply_rule(rule, current_text)
            if decision is None:
                continue
            triggered.append(decision)
            if rule.action == 'block':
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

    def _apply_rule(self, rule: GuardrailRule, text: str) -> tuple[str, GuardrailDecision | None]:
        if rule.type == 'max_length':
            return self._apply_max_length(rule, text)
        if rule.type == 'keyword':
            return self._apply_keyword(rule, text)
        if rule.type == 'regex':
            return self._apply_regex(rule, text)
        return text, None

    def _apply_max_length(self, rule: GuardrailRule, text: str) -> tuple[str, GuardrailDecision | None]:
        if rule.max_length is None or len(text) <= rule.max_length:
            return text, None
        return text, self._decision(rule)

    def _apply_keyword(self, rule: GuardrailRule, text: str) -> tuple[str, GuardrailDecision | None]:
        lowered_text = text.lower()
        for keyword in rule.keywords:
            if keyword.lower() in lowered_text:
                if rule.action == 'redact':
                    redacted = re.sub(re.escape(keyword), rule.replacement, text, flags=re.IGNORECASE)
                    return redacted, self._decision(rule)
                return text, self._decision(rule)
        return text, None

    def _apply_regex(self, rule: GuardrailRule, text: str) -> tuple[str, GuardrailDecision | None]:
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
        return updated, self._decision(rule)

    @staticmethod
    def _resolve_flags(flag_names: list[str]) -> int:
        flags = 0
        for flag_name in flag_names:
            flags |= _FLAG_MAP.get(flag_name.upper(), 0)
        return flags

    @staticmethod
    def _decision(rule: GuardrailRule) -> GuardrailDecision:
        return GuardrailDecision(
            rule_id=rule.id,
            action=rule.action,
            target=rule.target,
            message=rule.message,
            tags=rule.tags,
        )
