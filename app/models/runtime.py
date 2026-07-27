from typing import Any

from pydantic import BaseModel, Field


class RuntimeContext(BaseModel):
    subject: str
    user_id: str
    organization_id: str | None = None
    branch_id: str | None = None
    app_id: str | None = None
    tenant_id: str | None = None
    locale: str | None = None
    application_ids: list[str] = Field(default_factory=list)
    roles: list[str] = Field(default_factory=list)
    permissions: list[str] = Field(default_factory=list)
    session_id: str | None = None
    request_id: str | None = None
    correlation_id: str | None = None
    trace_id: str | None = None
    jwt: str | None = None
    raw_claims: dict[str, Any] = Field(default_factory=dict)
