import pytest

from app.agent.agent import GeneralAgent
from app.agent.models import AgentState
from app.models.planner import PlannerOutput, PlannerTask
from app.models.runtime import RuntimeContext


class RecordingToolExecutor:
    def __init__(self, result: dict[str, object] | None = None) -> None:
        self.calls: list[dict[str, object]] = []
        self._result = result or {
            'status': 'success',
            'success': True,
            'tool_name': 'stub.tool',
            'data': {'ok': True},
        }

    async def execute(self, **kwargs):
        self.calls.append(kwargs)
        return self._result


def make_agent(executor: RecordingToolExecutor) -> GeneralAgent:
    return GeneralAgent(
        planner_service=object(),
        tool_executor_service=executor,
        response_generator_node=object(),
    )


def make_state(planner_output: PlannerOutput) -> AgentState:
    return AgentState(
        conversation_id='conv-1',
        request_id='req-1',
        correlation_id='corr-1',
        runtime_context=RuntimeContext(subject='user-1', user_id='user-1'),
        user_question='test question',
        planner_output=planner_output,
        trace_id='trace-1',
    )


@pytest.mark.asyncio
async def test_legacy_planner_output_without_tasks_executes_unchanged():
    executor = RecordingToolExecutor()
    agent = make_agent(executor)
    planner_output = PlannerOutput(
        intent='academic.holiday.list',
        requires_tool=True,
        domain='vidhya',
        service='academic',
        entity='holiday',
        operation='list',
        parameters={'academic_year_id': 'ay-1'},
    )

    updates = await agent.act(make_state(planner_output))

    assert len(executor.calls) == 1
    assert executor.calls[0]['planner_output'] is planner_output
    assert updates['tool_execution_result']['success'] is True


@pytest.mark.asyncio
async def test_single_enterprise_task_is_converted_to_execution_plan():
    executor = RecordingToolExecutor()
    agent = make_agent(executor)
    planner_output = PlannerOutput(
        tasks=[
            PlannerTask(
                type='enterprise',
                domain='vidhya',
                service='academic',
                entity='holiday',
                operation='list',
                parameters={'academic_year_id': 'ay-1'},
            )
        ],
    )

    updates = await agent.act(make_state(planner_output))

    assert len(executor.calls) == 1
    executed_output = executor.calls[0]['planner_output']
    assert executed_output.requires_tool is True
    assert len(executed_output.execution_plan) == 1
    step = executed_output.execution_plan[0]
    assert step.step_id == 'step_1'
    assert step.domain == 'vidhya'
    assert step.service == 'academic'
    assert step.entity == 'holiday'
    assert step.operation == 'list'
    assert step.parameters == {'academic_year_id': 'ay-1'}
    assert updates['tool_execution_result']['success'] is True


@pytest.mark.asyncio
async def test_multiple_enterprise_tasks_execute_in_declared_order():
    executor = RecordingToolExecutor()
    agent = make_agent(executor)
    planner_output = PlannerOutput(
        tasks=[
            PlannerTask(
                type='enterprise',
                domain='vidhya',
                service='academic',
                entity='academic_year',
                operation='list',
            ),
            PlannerTask(
                type='enterprise',
                domain='vidhya',
                service='academic',
                entity='holiday',
                operation='list',
                parameters={'academic_year_id': 'ay-1'},
            ),
        ],
    )

    updates = await agent.act(make_state(planner_output))

    assert len(executor.calls) == 1
    executed_output = executor.calls[0]['planner_output']
    assert [step.step_id for step in executed_output.execution_plan] == ['step_1', 'step_2']
    assert [step.entity for step in executed_output.execution_plan] == ['academic_year', 'holiday']
    assert executed_output.execution_plan[1].parameters == {'academic_year_id': 'ay-1'}
    assert updates['tool_execution_result']['success'] is True


@pytest.mark.asyncio
async def test_unsupported_task_type_returns_placeholder_without_executing():
    executor = RecordingToolExecutor()
    agent = make_agent(executor)
    planner_output = PlannerOutput(
        tasks=[
            PlannerTask(type='web_search', parameters={'query': 'today weather'}),
        ],
    )

    updates = await agent.act(make_state(planner_output))

    assert executor.calls == []
    result = updates['tool_execution_result']
    assert result['status'] == 'unsupported_task_type'
    assert result['success'] is False
    assert result['error']['code'] == 'UNSUPPORTED_TASK_TYPE'
    assert result['error']['task_type'] == 'web_search'
    assert updates['tool_history'][-1]['success'] is False


@pytest.mark.asyncio
async def test_unsupported_task_type_short_circuits_even_when_mixed_with_enterprise_tasks():
    executor = RecordingToolExecutor()
    agent = make_agent(executor)
    planner_output = PlannerOutput(
        tasks=[
            PlannerTask(type='enterprise', domain='vidhya', service='academic', entity='holiday', operation='list'),
            PlannerTask(type='general', parameters={'topic': 'fees'}),
        ],
    )

    updates = await agent.act(make_state(planner_output))

    assert executor.calls == []
    result = updates['tool_execution_result']
    assert result['status'] == 'unsupported_task_type'
    assert result['error']['task_type'] == 'general'
