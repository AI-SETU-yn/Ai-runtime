from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.models.planner import ExecutionPlanStep, PlannerOutput


class ClarificationOption(BaseModel):
    label: str
    value: Any


class ClarificationRequest(BaseModel):
    step_id: str
    tool_name: str | None = None
    missing_parameters: list[str] = Field(default_factory=list)
    question: str
    options: list[ClarificationOption] = Field(default_factory=list)
    resolved_parameters: dict[str, Any] = Field(default_factory=dict)


class ExecutionState(BaseModel):
    original_question: str | None = None
    planner_output: PlannerOutput | None = None
    execution_plan: list[ExecutionPlanStep] = Field(default_factory=list)
    completed_steps: list[dict[str, Any]] = Field(default_factory=list)
    pending_step_id: str | None = None
    resolved_parameters: dict[str, dict[str, Any]] = Field(default_factory=dict)
    collected_inputs: dict[str, Any] = Field(default_factory=dict)
    awaiting_clarification: ClarificationRequest | None = None

    @property
    def awaiting_input(self) -> bool:
        return self.awaiting_clarification is not None
