from pydantic import BaseModel


class ConversationMetadata(BaseModel):
    conversation_id: str
    request_id: str
    correlation_id: str
    execution_time_ms: float
    planner_intent: str | None = None
    requires_tool: bool = False
    trace_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    metadata: ConversationMetadata
