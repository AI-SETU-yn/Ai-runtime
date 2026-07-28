from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class ExecutionPlanStep(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    step_id: str | None = Field(default=None, validation_alias=AliasChoices('step_id', 'stepId', 'id'))
    intent: str | None = None
    domain: str | None = None
    service: str | None = None
    entity: str | None = None
    operation: str | None = None
    tool: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    parameter_bindings: dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices('parameter_bindings', 'parameterBindings', 'bindings'),
    )
    depends_on: list[str] = Field(default_factory=list, validation_alias=AliasChoices('depends_on', 'dependsOn'))


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
    execution_plan: list[ExecutionPlanStep] = Field(
        default_factory=list,
        validation_alias=AliasChoices('execution_plan', 'executionPlan'),
    )
