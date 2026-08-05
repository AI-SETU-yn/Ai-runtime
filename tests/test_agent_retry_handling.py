import pytest

from app.agent.agent import GeneralAgent
from app.agent.models import AgentState
from app.agent.retry import ToolRetryHandler, ToolRetryPolicy
from app.models.planner import PlannerOutput
from app.models.runtime import RuntimeContext
from app.tool_executor.exceptions import ToolExecutionError


class FailThenSucceedToolExecutor:
    """Fails with a retryable error N times, then succeeds."""

    def __init__(self, failures_before_success: int, code: str = 'TIMEOUT') -> None:
        self._remaining_failures = failures_before_success
        self._code = code
        self.call_count = 0

    async def execute(self, **kwargs):
        self.call_count += 1
        if self._remaining_failures > 0:
            self._remaining_failures -= 1
            raise ToolExecutionError('Transient upstream failure.', code=self._code, status_code=504)
        return {'status': 'success', 'success': True, 'tool_name': 'records.list', 'data': {'ok': True}}


class AlwaysFailingToolExecutor:
    def __init__(self, code: str = 'TIMEOUT') -> None:
        self._code = code
        self.call_count = 0

    async def execute(self, **kwargs):
        self.call_count += 1
        raise ToolExecutionError('MCP server request timed out.', code=self._code, status_code=504)


def make_state() -> AgentState:
    return AgentState(
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


# --- ToolRetryPolicy --------------------------------------------------------


def test_policy_rejects_max_attempts_below_one() -> None:
    with pytest.raises(ValueError, match='max_attempts'):
        ToolRetryPolicy(max_attempts=0)


def test_policy_rejects_negative_base_delay() -> None:
    with pytest.raises(ValueError, match='base_delay_seconds'):
        ToolRetryPolicy(base_delay_seconds=-1)


def test_policy_rejects_max_delay_below_base_delay() -> None:
    with pytest.raises(ValueError, match='max_delay_seconds'):
        ToolRetryPolicy(base_delay_seconds=1.0, max_delay_seconds=0.5)


def test_policy_rejects_negative_jitter() -> None:
    with pytest.raises(ValueError, match='jitter_seconds'):
        ToolRetryPolicy(jitter_seconds=-0.1)


def test_policy_is_retryable_checks_error_code() -> None:
    policy = ToolRetryPolicy(retryable_codes=frozenset({'TIMEOUT'}))
    assert policy.is_retryable(ToolExecutionError('x', code='TIMEOUT', status_code=504)) is True
    assert policy.is_retryable(ToolExecutionError('x', code='VALIDATION_ERROR', status_code=400)) is False


def test_default_policy_disables_retries() -> None:
    assert ToolRetryPolicy().max_attempts == 1


# --- ToolRetryHandler --------------------------------------------------------


@pytest.mark.asyncio
async def test_retry_handler_default_policy_does_not_retry() -> None:
    handler = ToolRetryHandler()
    executor = AlwaysFailingToolExecutor()

    with pytest.raises(ToolExecutionError):
        await handler.run(lambda: executor.execute())

    assert executor.call_count == 1


@pytest.mark.asyncio
async def test_retry_handler_retries_retryable_errors_until_success() -> None:
    handler = ToolRetryHandler(ToolRetryPolicy(max_attempts=3, base_delay_seconds=0, jitter_seconds=0))
    executor = FailThenSucceedToolExecutor(failures_before_success=2)

    result = await handler.run(lambda: executor.execute())

    assert result['success'] is True
    assert executor.call_count == 3


@pytest.mark.asyncio
async def test_retry_handler_stops_at_max_attempts() -> None:
    handler = ToolRetryHandler(ToolRetryPolicy(max_attempts=3, base_delay_seconds=0, jitter_seconds=0))
    executor = AlwaysFailingToolExecutor()

    with pytest.raises(ToolExecutionError):
        await handler.run(lambda: executor.execute())

    assert executor.call_count == 3


@pytest.mark.asyncio
async def test_retry_handler_does_not_retry_non_retryable_codes() -> None:
    handler = ToolRetryHandler(
        ToolRetryPolicy(max_attempts=5, base_delay_seconds=0, jitter_seconds=0, retryable_codes=frozenset({'TIMEOUT'}))
    )
    executor = AlwaysFailingToolExecutor(code='VALIDATION_ERROR')

    with pytest.raises(ToolExecutionError):
        await handler.run(lambda: executor.execute())

    assert executor.call_count == 1


@pytest.mark.asyncio
async def test_retry_handler_invokes_on_retry_hook_with_attempt_and_error() -> None:
    handler = ToolRetryHandler(ToolRetryPolicy(max_attempts=3, base_delay_seconds=0, jitter_seconds=0))
    executor = FailThenSucceedToolExecutor(failures_before_success=2)
    calls: list[tuple[int, str]] = []

    await handler.run(lambda: executor.execute(), on_retry=lambda attempt, exc: calls.append((attempt, exc.code)))

    assert calls == [(1, 'TIMEOUT'), (2, 'TIMEOUT')]


# --- GeneralAgent integration -------------------------------------------------


@pytest.mark.asyncio
async def test_agent_uses_default_retry_handler_and_does_not_retry() -> None:
    executor = AlwaysFailingToolExecutor()
    agent = GeneralAgent(planner_service=object(), tool_executor_service=executor, response_generator_node=object())

    updates = await agent.act(make_state())

    assert executor.call_count == 1
    result = updates['tool_execution_result']
    assert result['status'] == 'error'
    assert result['error']['code'] == 'TIMEOUT'


@pytest.mark.asyncio
async def test_agent_recovers_via_injected_retry_handler() -> None:
    executor = FailThenSucceedToolExecutor(failures_before_success=1)
    retry_handler = ToolRetryHandler(ToolRetryPolicy(max_attempts=2, base_delay_seconds=0, jitter_seconds=0))
    agent = GeneralAgent(
        planner_service=object(),
        tool_executor_service=executor,
        response_generator_node=object(),
        retry_handler=retry_handler,
    )

    updates = await agent.act(make_state())

    assert executor.call_count == 2
    result = updates['tool_execution_result']
    assert result['status'] == 'success'
    assert updates['tool_history'][-1]['success'] is True


@pytest.mark.asyncio
async def test_agent_reports_structured_failure_after_exhausting_retries() -> None:
    executor = AlwaysFailingToolExecutor()
    retry_handler = ToolRetryHandler(ToolRetryPolicy(max_attempts=3, base_delay_seconds=0, jitter_seconds=0))
    agent = GeneralAgent(
        planner_service=object(),
        tool_executor_service=executor,
        response_generator_node=object(),
        retry_handler=retry_handler,
    )

    updates = await agent.act(make_state())

    assert executor.call_count == 3
    result = updates['tool_execution_result']
    assert result['status'] == 'error'
    assert result['success'] is False
    assert result['tool_name'] == 'records.list'
    assert updates['tool_history'][-1]['success'] is False
