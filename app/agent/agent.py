from __future__ import annotations

import logging

from app.agent.decision import DecisionEngine
from app.agent.memory import MemoryInterface, RequestScopedMemory
from app.agent.models import AgentDecisionAction, AgentState, AgentStatus, ReasoningContext
from app.agent.observation import ObservationManager
from app.agent.tool_discovery import ToolDiscovery
from app.graph.nodes.current_info_router import CurrentInfoRouterNode
from app.graph.nodes.response_generator import ResponseGeneratorNode
from app.models.execution import ExecutionState
from app.planner.planner import PlannerService
from app.tool_executor.service import ToolExecutorService
from app.utils.json_logging import pretty_json
from app.utils.redaction import redact_sensitive

logger = logging.getLogger(__name__)


class GeneralAgent:
    """Owns the enterprise reason-act-observe-decide loop."""

    def __init__(
        self,
        planner_service: PlannerService,
        tool_executor_service: ToolExecutorService,
        response_generator_node: ResponseGeneratorNode,
        *,
        decision_engine: DecisionEngine | None = None,
        observation_manager: ObservationManager | None = None,
        memory: MemoryInterface | None = None,
        tool_discovery: ToolDiscovery | None = None,
        current_info_router: CurrentInfoRouterNode | None = None,
    ) -> None:
        self._planner_service = planner_service
        self._tool_executor_service = tool_executor_service
        self._response_generator_node = response_generator_node
        self._decision_engine = decision_engine or DecisionEngine()
        self._observation_manager = observation_manager or ObservationManager()
        self._memory = memory or RequestScopedMemory()
        self._tool_discovery = tool_discovery or ToolDiscovery()
        self._current_info_router = current_info_router or CurrentInfoRouterNode()

    async def reason(self, state: AgentState) -> dict[str, object]:
        if state.resume_execution and state.planner_output is not None:
            logger.info('agent_reasoning_resumed_existing_plan_json\n%s', pretty_json({
                'intent': state.planner_output.intent,
                'pending_step_id': state.execution_state.pending_step_id if state.execution_state else None,
                'completed_step_count': len(state.execution_state.completed_steps) if state.execution_state else 0,
            }))
            return {
                'current_plan': state.planner_output,
                'current_goal': state.current_goal or state.user_question,
                'agent_status': AgentStatus.RUNNING,
            }
        memory_context = await self._memory.load(state)
        reasoning_context = self._reasoning_context(state)
        prompt_message = self._planner_message(state, reasoning_context)
        planner_output = await self._planner_service.plan(prompt_message)
        reasoning_history = [
            *state.reasoning_history,
            planner_output.intent,
        ]
        logger.info('agent_reasoning_completed_json\n%s', pretty_json({
            'iteration': state.iteration_count + 1,
            'intent': planner_output.intent,
            'requires_tool': planner_output.requires_tool,
            'execution_plan_steps': len(planner_output.execution_plan),
            'observation_count': len(state.observations),
        }))
        return {
            'planner_output': planner_output,
            'current_plan': planner_output,
            'current_goal': state.current_goal or state.user_question,
            'memory_context': memory_context,
            'reasoning_history': reasoning_history,
            'agent_status': AgentStatus.RUNNING,
        }

    async def act(self, state: AgentState) -> dict[str, object]:
        planner_output = state.planner_output
        assert planner_output is not None
        result = await self._tool_executor_service.execute(
            planner_output=planner_output,
            runtime_context=state.runtime_context,
            request_id=state.request_id,
            correlation_id=state.correlation_id,
            trace_id=state.trace_id or '',
            execution_state=state.execution_state,
            clarification_answer=state.clarification_answer,
            allow_clarification=True,
        )
        tool_history = [
            *state.tool_history,
            {
                'iteration': state.iteration_count + 1,
                'intent': planner_output.intent,
                'requires_tool': planner_output.requires_tool,
                'tool_name': result.get('tool_name') if isinstance(result, dict) else None,
                'success': result.get('success') if isinstance(result, dict) else None,
            },
        ]
        updates: dict[str, object] = {
            'tool_execution_result': result,
            'iteration_count': state.iteration_count + 1,
            'tool_history': tool_history,
        }
        if isinstance(result, dict) and isinstance(result.get('execution_state'), dict):
            updates['execution_state'] = ExecutionState.model_validate(result['execution_state'])
        return updates

    async def observe(self, state: AgentState) -> dict[str, object]:
        observation = self._observation_manager.from_tool_result(state)
        if observation is None:
            logger.info('agent_observation_skipped_no_tool_result')
            return {}
        await self._memory.save_observation(state, observation)
        observations = [*state.observations, observation]
        logger.info('agent_observation_recorded_json\n%s', pretty_json({
            'iteration': observation.iteration,
            'tool_name': observation.tool_name,
            'execution_status': observation.execution_status,
            'success': observation.success,
            'latency_ms': observation.latency_ms,
        }))
        return {
            'observations': observations,
            'memory_context': dict(state.memory_context),
        }

    async def decide(self, state: AgentState) -> dict[str, object]:
        decision = self._decision_engine.decide(state)
        status = AgentStatus.RUNNING if decision.should_continue else AgentStatus.COMPLETED
        if decision.action == AgentDecisionAction.ABORT:
            status = AgentStatus.ABORTED
        logger.info('agent_decision_json\n%s', pretty_json({
            'iteration': state.iteration_count,
            'action': decision.action.value,
            'reason': decision.reason,
            'should_continue': decision.should_continue,
        }))
        return {
            'decision': decision,
            'agent_status': status,
        }

    async def generate_final_response(self, state: AgentState) -> dict[str, object]:
        if state.decision and state.decision.action == AgentDecisionAction.ABORT:
            abort_state = state.model_copy(
                update={
                    'tool_execution_result': {
                        'status': 'error',
                        'success': False,
                        'error': {
                            'code': 'AGENT_ABORTED',
                            'message': state.decision.reason,
                        },
                    }
                }
            )
            return await self._response_generator_node(abort_state)
        web_updates = await self._current_info_router(state)
        response_state = state.model_copy(update=web_updates) if web_updates else state
        return await self._response_generator_node(response_state)

    def route_after_decision(self, state: AgentState) -> str:
        if state.decision and state.decision.action == AgentDecisionAction.REPLAN:
            return 'continue'
        return 'finalize'

    def _reasoning_context(self, state: AgentState) -> ReasoningContext:
        failures = [
            observation
            for observation in state.observations
            if observation.success is False or observation.execution_status == 'error'
        ]
        return ReasoningContext(
            original_request=state.user_question,
            observations=state.observations,
            failures=failures,
            completed_steps=state.tool_history,
            available_tools=self._tool_discovery.available_tools(),
        )

    @staticmethod
    def _planner_message(state: AgentState, reasoning_context: ReasoningContext) -> str:
        if not reasoning_context.observations:
            return state.user_question
        return (
            f'Original user request:\n{reasoning_context.original_request}\n\n'
            f'Previous observations:\n{pretty_json(redact_sensitive([item.model_dump() for item in reasoning_context.observations]))}\n\n'
            'Continue planning only if more enterprise tool work is required. '
            'Otherwise return a finalizable plan with requires_tool=false.'
        )
