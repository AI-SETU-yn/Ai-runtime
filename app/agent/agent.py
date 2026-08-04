from __future__ import annotations

import logging
from typing import Any

from app.agent.decision import DecisionEngine
from app.agent.memory import MemoryInterface, RequestScopedMemory
from app.agent.models import AgentDecisionAction, AgentState, AgentStatus, ReasoningContext
from app.agent.observation import ObservationManager
from app.agent.tool_discovery import ToolDiscovery
from app.graph.nodes.current_info_router import CurrentInfoRouterNode
from app.graph.nodes.response_generator import ResponseGeneratorNode
from app.models.execution import ExecutionState
from app.models.planner import ExecutionPlanStep, PlannerOutput, PlannerTask
from app.planner.planner import PlannerService
from app.rbac.service import RBACAuthorizationService
from app.tool_executor.exceptions import ToolExecutionError
from app.tool_executor.service import ToolExecutorService
from app.utils.json_logging import pretty_json
from app.utils.redaction import redact_sensitive

logger = logging.getLogger(__name__)

SUPPORTED_TASK_TYPES = frozenset({'enterprise'})


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
        rbac_authorization_service: RBACAuthorizationService | None = None,
    ) -> None:
        self._planner_service = planner_service
        self._tool_executor_service = tool_executor_service
        self._response_generator_node = response_generator_node
        self._decision_engine = decision_engine or DecisionEngine()
        self._observation_manager = observation_manager or ObservationManager()
        self._memory = memory or RequestScopedMemory()
        self._tool_discovery = tool_discovery or ToolDiscovery()
        self._current_info_router = current_info_router or CurrentInfoRouterNode()
        self._rbac_authorization_service = rbac_authorization_service

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

        routed = self._route_tasks(planner_output)
        if isinstance(routed, dict):
            logger.warning('agent_task_routing_unsupported_task_type_json\n%s', pretty_json({
                'intent': planner_output.intent,
                'task_type': routed['error'].get('task_type'),
            }))
            tool_history = [
                *state.tool_history,
                {
                    'iteration': state.iteration_count + 1,
                    'intent': planner_output.intent,
                    'requires_tool': planner_output.requires_tool,
                    'tool_name': None,
                    'success': False,
                },
            ]
            return {
                'tool_execution_result': routed,
                'iteration_count': state.iteration_count + 1,
                'tool_history': tool_history,
            }
        execution_target = routed

        if self._rbac_authorization_service is not None:
            decision = await self._rbac_authorization_service.authorize_plan(
                planner_output=execution_target,
                runtime_context=state.runtime_context,
                request_id=state.request_id,
                correlation_id=state.correlation_id,
                trace_id=state.trace_id or '',
            )
            if not decision.allowed:
                result = self._rbac_authorization_service.authorization_failure_result(decision)
                tool_history = [
                    *state.tool_history,
                    {
                        'iteration': state.iteration_count + 1,
                        'intent': planner_output.intent,
                        'requires_tool': planner_output.requires_tool,
                        'tool_name': result.get('tool_name'),
                        'success': False,
                        'authorization': 'denied',
                    },
                ]
                logger.warning('agent_tool_authorization_denied_json\n%s', pretty_json(redact_sensitive({
                    'intent': planner_output.intent,
                    'tool_name': result.get('tool_name'),
                    'error': result.get('error'),
                })))
                return {
                    'tool_execution_result': result,
                    'iteration_count': state.iteration_count + 1,
                    'tool_history': tool_history,
                }
        try:
            result = await self._tool_executor_service.execute(
                planner_output=execution_target,
                runtime_context=state.runtime_context,
                request_id=state.request_id,
                correlation_id=state.correlation_id,
                trace_id=state.trace_id or '',
                execution_state=state.execution_state,
                clarification_answer=state.clarification_answer,
                allow_clarification=True,
            )
        except ToolExecutionError as exc:
            result = {
                'status': 'error',
                'success': False,
                'tool_name': execution_target.tool,
                'data': None,
                'error': {
                    'code': exc.code,
                    'message': exc.message,
                    'status_code': exc.status_code,
                },
            }
            logger.warning('agent_tool_execution_failed_json\n%s', pretty_json(redact_sensitive({
                'intent': planner_output.intent,
                'tool_name': execution_target.tool,
                'error': result['error'],
            })))
        tool_history = [
            *state.tool_history,
            {
                'iteration': state.iteration_count + 1,
                'intent': planner_output.intent,
                'requires_tool': execution_target.requires_tool,
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

    def _route_tasks(self, planner_output: PlannerOutput) -> PlannerOutput | dict[str, Any]:
        """Runtime Task Router: dispatches PlannerOutput.tasks, preserving legacy behavior when absent."""
        if not planner_output.tasks:
            return planner_output

        unsupported_task = next(
            (task for task in planner_output.tasks if task.type.strip().casefold() not in SUPPORTED_TASK_TYPES),
            None,
        )
        if unsupported_task is not None:
            return self._unsupported_task_type_result(unsupported_task)

        execution_plan = self._execution_plan_from_enterprise_tasks(planner_output.tasks)
        return planner_output.model_copy(update={'requires_tool': True, 'execution_plan': execution_plan})

    @staticmethod
    def _execution_plan_from_enterprise_tasks(tasks: list[PlannerTask]) -> list[ExecutionPlanStep]:
        return [
            ExecutionPlanStep(
                step_id=f'step_{index}',
                intent=(
                    f'{task.service}.{task.entity}.{task.operation}'
                    if all((task.service, task.entity, task.operation))
                    else None
                ),
                domain=task.domain,
                service=task.service,
                entity=task.entity,
                operation=task.operation,
                parameters=dict(task.parameters),
                visible_in_response=True,
            )
            for index, task in enumerate(tasks, start=1)
        ]

    @staticmethod
    def _unsupported_task_type_result(task: PlannerTask) -> dict[str, Any]:
        return {
            'status': 'unsupported_task_type',
            'success': False,
            'response_type': 'unsupported_task_type',
            'tool_name': None,
            'data': None,
            'error': {
                'code': 'UNSUPPORTED_TASK_TYPE',
                'message': f"Runtime does not support task type '{task.type}' in this phase.",
                'task_type': task.type,
            },
        }

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
