from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SecurityClassifierSettings(BaseModel):
    model_config = ConfigDict(extra='forbid')

    enabled: bool = True
    mode: Literal['hybrid'] = 'hybrid'
    always_run: bool = False
    confidence_threshold: float = Field(default=0.9, ge=0.0, le=1.0)
    suspicious_tags: list[str] = Field(default_factory=lambda: ['suspicious'])


class TokenBudgetSettings(BaseModel):
    model_config = ConfigDict(extra='forbid')

    enabled: bool = True
    # Generous default (well above the ~1000-token equivalent of the existing
    # 4000-character limit) so this acts as a safety net alongside, not a
    # replacement for, input.max_length while token counting is verified.
    limit: int = Field(default=6000, gt=0)


class GuardrailRule(BaseModel):
    model_config = ConfigDict(extra='forbid')

    id: str
    description: str | None = None
    enabled: bool = True
    type: Literal['regex', 'keyword', 'max_length']
    action: Literal['allow', 'block', 'redact', 'annotate']
    target: str = 'message'
    message: str | None = None
    tags: list[str] = Field(default_factory=list)
    patterns: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    max_length: int | None = None
    replacement: str = '[redacted]'
    flags: list[str] = Field(default_factory=list)
    exempt_patterns: list[str] = Field(default_factory=list)
    exempt_context_keys: list[str] = Field(default_factory=list)


class GuardrailStageConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')

    rules: list[GuardrailRule] = Field(default_factory=list)


class GuardrailsConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')

    input: GuardrailStageConfig = Field(default_factory=GuardrailStageConfig)
    output: GuardrailStageConfig = Field(default_factory=GuardrailStageConfig)
    security_classifier: SecurityClassifierSettings = Field(default_factory=SecurityClassifierSettings)
    token_budget: TokenBudgetSettings = Field(default_factory=TokenBudgetSettings)


class GuardrailDecision(BaseModel):
    model_config = ConfigDict(extra='forbid')

    rule_id: str
    action: Literal['allow', 'block', 'redact', 'annotate']
    target: str
    message: str | None = None
    tags: list[str] = Field(default_factory=list)
    deferred: bool = False


class GuardrailEvaluationResult(BaseModel):
    model_config = ConfigDict(extra='forbid')

    stage: Literal['input', 'output']
    original_text: str
    final_text: str
    blocked: bool = False
    triggered: list[GuardrailDecision] = Field(default_factory=list)

    @property
    def redacted(self) -> bool:
        return self.original_text != self.final_text

    @property
    def suspicious(self) -> bool:
        return any('suspicious' in decision.tags for decision in self.triggered)

    @property
    def pii_detected(self) -> bool:
        return any('pii' in decision.tags for decision in self.triggered)
