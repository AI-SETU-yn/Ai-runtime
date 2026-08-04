from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AuthorizationStep(BaseModel):
    step_id: str
    tool: str
    capability: str
    domain: str
    service: str
    entity: str
    operation: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class AuthorizationRequest(BaseModel):
    user_id: str
    organization_id: str | None = None
    branch_id: str | None = None
    app_id: str | None = None
    tenant_id: str | None = None
    session_id: str | None = None
    request_id: str | None = None
    correlation_id: str | None = None
    trace_id: str | None = None
    steps: list[AuthorizationStep] = Field(default_factory=list)


class AuthorizationDecision(BaseModel):
    allowed: bool
    step_id: str | None = None
    tool: str | None = None
    capability: str | None = None
    domain: str | None = None
    service: str | None = None
    entity: str | None = None
    operation: str | None = None
    reason: str = ''
    source: str = 'rbac'
    raw_response: dict[str, Any] = Field(default_factory=dict)
