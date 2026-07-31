from pydantic import BaseModel, Field


class GuardrailOutcome(BaseModel):
    stage: str
    action: str
    rule_id: str
    tags: list[str] = Field(default_factory=list)


class GuardrailMetadata(BaseModel):
    input: list[GuardrailOutcome] = Field(default_factory=list)
    output: list[GuardrailOutcome] = Field(default_factory=list)


class SecurityMetadata(BaseModel):
    executed: bool = False
    triggered_by: str | None = None
    safe: bool | None = None
    category: str | None = None
    confidence: float | None = None
    reason: str | None = None


class ConversationMetadata(BaseModel):
    conversation_id: str
    request_id: str
    correlation_id: str
    execution_time_ms: float
    planner_intent: str | None = None
    requires_tool: bool = False
    trace_id: str | None = None
    guardrails: GuardrailMetadata | None = None
    security: SecurityMetadata | None = None


class ChatResponse(BaseModel):
    answer: str
    metadata: ConversationMetadata