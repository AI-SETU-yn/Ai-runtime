import pytest

from app.agent.agent import GeneralAgent
from app.agent.models import AgentState
from app.models.planner import PlannerOutput
from app.models.runtime import RuntimeContext
from app.tool_executor.exceptions import ToolExecutionError


class FailingToolExecutor:
    async def execute(self, **kwargs):
        raise ToolExecutionError('MCP server request timed out.', code='TIMEOUT', status_code=504)


@pytest.mark.asyncio
async def test_agent_maps_tool_execution_error_to_structured_tool_failure():
    agent = GeneralAgent(
        planner_service=object(),
        tool_executor_service=FailingToolExecutor(),
        response_generator_node=object(),
    )
    state = AgentState(
        conversation_id='conv-1',
        request_id='req-1',
        correlation_id='corr-1',
        runtime_context=RuntimeContext(subject='user-1', user_id='user-1'),
        user_question='List records',
        planner_output=PlannerOutput(
            intent='generic.record.list',
            requires_tool=True,
            domain='generic',
            service='records',
            entity='record',
            operation='list',
            tool='records.list',
        ),
    )

    updates = await agent.act(state)

    result = updates['tool_execution_result']
    assert result['status'] == 'error'
    assert result['success'] is False
    assert result['tool_name'] == 'records.list'
    assert result['error'] == {
        'code': 'TIMEOUT',
        'message': 'MCP server request timed out.',
        'status_code': 504,
    }
    assert updates['tool_history'][0]['success'] is False
