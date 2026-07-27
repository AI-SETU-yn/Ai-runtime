from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class InvocationStatus(StrEnum):
    SUCCESS = 'success'
    ERROR = 'error'


class HealthState(StrEnum):
    HEALTHY = 'healthy'
    UNHEALTHY = 'unhealthy'


class ToolError(BaseModel):
    code: str
    message: str
    data: Any = None


class ToolResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    success: bool
    status: InvocationStatus
    tool_results: Any = Field(default=None, serialization_alias='toolResults')
    execution_time: float = Field(serialization_alias='executionTime')
    error: ToolError | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class HealthStatus(BaseModel):
    server: str
    status: HealthState
    latency: float
    details: dict[str, Any] = Field(default_factory=dict)
