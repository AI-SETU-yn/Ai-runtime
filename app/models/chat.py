from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    message: str = Field(min_length=1)
    conversation_id: str | None = None


class HealthResponse(BaseModel):
    status: str
