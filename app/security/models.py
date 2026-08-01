from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class SecurityCategory(StrEnum):
    SAFE = 'SAFE'
    PROMPT_INJECTION = 'PROMPT_INJECTION'
    ROLE_ESCALATION = 'ROLE_ESCALATION'
    SYSTEM_PROMPT_EXTRACTION = 'SYSTEM_PROMPT_EXTRACTION'
    DATA_EXFILTRATION = 'DATA_EXFILTRATION'
    UNKNOWN = 'UNKNOWN'


class SecurityDecision(BaseModel):
    model_config = ConfigDict(extra='forbid')

    safe: bool
    category: SecurityCategory
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class SecurityClassifierConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')

    enabled: bool = True
    mode: str = 'hybrid'
    always_run: bool = False
    confidence_threshold: float = Field(default=0.9, ge=0.0, le=1.0)
    suspicious_tags: list[str] = Field(default_factory=lambda: ['suspicious'])


class SecurityClassificationResult(BaseModel):
    model_config = ConfigDict(extra='forbid')

    executed: bool = False
    triggered_by: str | None = None
    decision: SecurityDecision | None = None

    @property
    def blocked(self) -> bool:
        return self.executed and self.decision is not None and not self.decision.safe