from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from app.graph.state import RuntimeState
from app.models.planner import PlannerOutput


class AgentStatus(StrEnum):
    RUNNING = 'running'
    COMPLETED = 'completed'
    ABORTED = 'aborted'


class AgentDecisionAction(StrEnum):
    EXECUTE = 'execute'
    REPLAN = 'replan'
    FINALIZE = 'finalize'
    ABORT = 'abort'


class Observation(BaseModel):
    observation_id: str = Field(default_factory=lambda: str(uuid4()))
    iteration: int
    tool_name: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    execution_status: str | None = None
    success: bool | None = None
    latency_ms: float | None = None
    result_summary: str | None = None
    structured_output: Any = None
    error: Any = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AgentDecision(BaseModel):
    action: AgentDecisionAction
    reason: str
    should_continue: bool = False


class ReasoningContext(BaseModel):
    original_request: str
    observations: list[Observation] = Field(default_factory=list)
    failures: list[Observation] = Field(default_factory=list)
    completed_steps: list[dict[str, Any]] = Field(default_factory=list)
    available_tools: list[str] = Field(default_factory=list)


class AgentState(RuntimeState):
    observations: list[Observation] = Field(default_factory=list)
    iteration_count: int = 0
    max_iterations: int = 3
    reasoning_history: list[str] = Field(default_factory=list)
    tool_history: list[dict[str, Any]] = Field(default_factory=list)
    current_plan: PlannerOutput | None = None
    current_goal: str | None = None
    memory_context: dict[str, Any] = Field(default_factory=dict)
    agent_status: AgentStatus = AgentStatus.RUNNING
    decision: AgentDecision | None = None

    @classmethod
    def from_runtime_state(cls, state: RuntimeState) -> 'AgentState':
        if isinstance(state, AgentState):
            return state
        return cls(**state.model_dump())
