from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class GuardrailRule(BaseModel):
    model_config = ConfigDict(extra='forbid')

    id: str
    description: str | None = None
    enabled: bool = True
    type: Literal['regex', 'keyword', 'max_length']
    action: Literal['allow', 'block', 'redact']
    target: str = 'message'
    message: str | None = None
    tags: list[str] = Field(default_factory=list)
    patterns: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    max_length: int | None = None
    replacement: str = '[redacted]'
    flags: list[str] = Field(default_factory=list)


class GuardrailStageConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')

    rules: list[GuardrailRule] = Field(default_factory=list)


class GuardrailsConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')

    input: GuardrailStageConfig = Field(default_factory=GuardrailStageConfig)
    output: GuardrailStageConfig = Field(default_factory=GuardrailStageConfig)


class GuardrailDecision(BaseModel):
    model_config = ConfigDict(extra='forbid')

    rule_id: str
    action: Literal['allow', 'block', 'redact']
    target: str
    message: str | None = None
    tags: list[str] = Field(default_factory=list)


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
