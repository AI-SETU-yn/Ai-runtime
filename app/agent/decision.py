from __future__ import annotations

from app.agent.models import AgentDecision, AgentDecisionAction, AgentState


class StopPolicy:
    def should_stop(self, state: AgentState) -> bool:
        return state.iteration_count >= state.max_iterations


class DecisionEngine:
    """Deterministic first-pass decision engine, shaped for later LLM policies."""

    def __init__(self, stop_policy: StopPolicy | None = None) -> None:
        self._stop_policy = stop_policy or StopPolicy()

    def decide(self, state: AgentState) -> AgentDecision:
        if self._stop_policy.should_stop(state) and self._needs_more_work(state):
            return AgentDecision(
                action=AgentDecisionAction.ABORT,
                reason='Maximum agent iterations reached before the goal was complete.',
            )
        if state.planner_output is None:
            return AgentDecision(
                action=AgentDecisionAction.REPLAN,
                reason='No planner output is available.',
                should_continue=True,
            )
        if not state.planner_output.requires_tool:
            return AgentDecision(
                action=AgentDecisionAction.FINALIZE,
                reason='Planner determined that no enterprise tool is required.',
            )
        if state.tool_execution_result is None:
            return AgentDecision(
                action=AgentDecisionAction.EXECUTE,
                reason='Planner selected a tool but it has not been executed yet.',
                should_continue=True,
            )
        return AgentDecision(
            action=AgentDecisionAction.FINALIZE,
            reason='Tool execution produced an observation for response generation.',
        )

    @staticmethod
    def _needs_more_work(state: AgentState) -> bool:
        return state.planner_output is None or (
            state.planner_output.requires_tool and state.tool_execution_result is None
        )
