from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class PlannerOutput(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    intent: str
    requires_tool: bool = Field(default=False, validation_alias=AliasChoices('requires_tool', 'requiresTool'))
    domain: str | None = None
    service: str | None = None
    entity: str | None = None
    operation: str | None = None
    tool: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    rationale: str | None = None
    raw_response: str | None = Field(default=None, validation_alias=AliasChoices('raw_response', 'rawResponse'))
    adapter: str | None = None
    model: str | None = None
