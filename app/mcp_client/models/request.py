from typing import Any
from uuid import uuid4

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class ToolRequestParams(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    meta: dict[str, Any] = Field(default_factory=dict, validation_alias=AliasChoices('meta', '_meta'), serialization_alias='_meta')


class ToolRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    jsonrpc: str = '2.0'
    id: str = Field(default_factory=lambda: str(uuid4()))
    method: str = 'tools/call'
    params: ToolRequestParams

    @property
    def trace_id(self) -> str | None:
        value = self.params.meta.get('trace_id')
        return value if isinstance(value, str) else None

    def wire_payload(self) -> dict[str, Any]:
        return self.model_dump(by_alias=True, exclude_none=True)
