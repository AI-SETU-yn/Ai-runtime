from __future__ import annotations

import logging
import re
from collections.abc import Iterable

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

logger = logging.getLogger(__name__)

_PHONE_REDACTION_RULE_ID = 'output.phone_redaction'
_PHONE_CANDIDATE_PATTERN = re.compile(
    r'(?<![A-Za-z0-9])(?:\+?\d(?:[\d\s().-]{8,}\d)|\(\d{3}\)\s*\d{3}[-.\s]?\d{4})(?![A-Za-z0-9-])'
)
_ACADEMIC_YEAR_PATTERN = re.compile(r'^\s*(?:academic\s*year\s*[:\-]?\s*)?(19|20)\d{2}\s*-\s*(19|20)\d{2}\s*$', re.IGNORECASE)
_ISO_DATE_PATTERN = re.compile(r'^\s*\d{4}[-/]\d{2}[-/]\d{2}\s*$')
_MONTH_YEAR_PATTERN = re.compile(r'^\s*(?:0?[1-9]|1[0-2])[-/](19|20)\d{2}\s*$')
_FINANCIAL_YEAR_PATTERN = re.compile(r'^\s*(?:FY\s*)?(19|20)\d{2}(?:\s*-\s*(19|20)\d{2})?\s*$', re.IGNORECASE)
_QUARTER_PATTERN = re.compile(r'^\s*(?:Q[1-4]|Quarter\s*[1-4])(?:\s*[-/]?\s*(19|20)\d{2})?\s*$', re.IGNORECASE)
_UUID_PATTERN = re.compile(r'^\s*[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\s*$', re.IGNORECASE)
_BUSINESS_ID_PATTERN = re.compile(r'^[A-Za-z]{2,}\d{2,}[A-Za-z0-9-]*$|^\d{4,}-[A-Za-z0-9-]{2,}$')
_PHONE_CONTEXT_WHITELIST = {
    'academicyear',
    'financialyear',
    'startdate',
    'enddate',
    'month',
    'year',
    'referenceid',
}


class GuardrailEngine:
    def __init__(self, config: GuardrailsConfig) -> None:
        self._config = config

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
        if rule.id == _PHONE_REDACTION_RULE_ID and rule.action == 'redact':
            return self._apply_phone_redaction(rule, text, defer_block_tags=defer_block_tags)
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

    def _apply_phone_redaction(
        self,
        rule: GuardrailRule,
        text: str,
        *,
        defer_block_tags: set[str],
    ) -> tuple[str, GuardrailDecision | None]:
        matches = list(_PHONE_CANDIDATE_PATTERN.finditer(text))
        if not matches:
            return text, None

        redacted_any = False
        updated_parts: list[str] = []
        last_index = 0

        for match in matches:
            start, end = match.span()
            candidate = match.group(0)
            updated_parts.append(text[last_index:start])
            should_redact, reason, confidence = self._assess_phone_candidate(text, match)
            if should_redact:
                updated_parts.append(rule.replacement)
                redacted_any = True
                logger.info(
                    'Phone redaction applied rule_id=%s matched_value=%r reason=%s confidence=%.2f',
                    rule.id,
                    candidate,
                    reason,
                    confidence,
                )
            else:
                updated_parts.append(candidate)
                logger.info(
                    'Skipped Phone Redaction rule_id=%s reason=%s value=%r confidence=%.2f',
                    rule.id,
                    reason,
                    candidate,
                    confidence,
                )
            last_index = end

        updated_parts.append(text[last_index:])
        updated = ''.join(updated_parts)
        if not redacted_any:
            return text, None
        return updated, self._decision(rule, defer_block_tags=defer_block_tags)

    def _assess_phone_candidate(self, full_text: str, match: re.Match[str]) -> tuple[bool, str, float]:
        candidate = match.group(0).strip()
        normalized = re.sub(r'\s+', ' ', candidate)
        digits_only = re.sub(r'\D', '', candidate)

        if len(digits_only) < 10:
            return False, 'insufficient_digits', 0.05
        if self._has_whitelisted_context(full_text, match.start()):
            return False, 'context_whitelist', 0.10
        if self._is_non_phone_identifier(normalized):
            return False, 'non_phone_identifier', 0.10
        if not self._looks_like_phone_number(candidate, digits_only):
            return False, 'invalid_phone_shape', 0.25
        return True, 'valid_phone_number', 0.98

    @staticmethod
    def _has_whitelisted_context(full_text: str, match_start: int) -> bool:
        line_start = full_text.rfind('\n', 0, match_start) + 1
        line_end = full_text.find('\n', match_start)
        if line_end == -1:
            line_end = len(full_text)
        line = full_text[line_start:line_end]
        prefix = line[: max(0, match_start - line_start)]
        field_match = re.search(r'([A-Za-z][A-Za-z0-9_]*)\s*:\s*$', prefix)
        if not field_match:
            return False
        field_name = re.sub(r'[^A-Za-z0-9]', '', field_match.group(1)).lower()
        return field_name in _PHONE_CONTEXT_WHITELIST

    @staticmethod
    def _is_non_phone_identifier(value: str) -> bool:
        if _ACADEMIC_YEAR_PATTERN.fullmatch(value):
            return True
        if _ISO_DATE_PATTERN.fullmatch(value):
            return True
        if _MONTH_YEAR_PATTERN.fullmatch(value):
            return True
        if _FINANCIAL_YEAR_PATTERN.fullmatch(value):
            return True
        if _QUARTER_PATTERN.fullmatch(value):
            return True
        if _UUID_PATTERN.fullmatch(value):
            return True
        if _BUSINESS_ID_PATTERN.fullmatch(value):
            return True
        return False

    @staticmethod
    def _looks_like_phone_number(candidate: str, digits_only: str) -> bool:
        if len(digits_only) < 10 or len(digits_only) > 15:
            return False
        if re.fullmatch(r'(19|20)\d{2}-(19|20)\d{2}', candidate.strip()):
            return False
        if re.fullmatch(r'\d{4}[-/]\d{2}[-/]\d{2}', candidate.strip()):
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
