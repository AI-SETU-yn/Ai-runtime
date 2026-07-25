from app.graph.state import RuntimeState
from app.tool_executor.service import ToolExecutorService


class ToolExecutorNode:
    def __init__(self, tool_executor_service: ToolExecutorService) -> None:
        self._tool_executor_service = tool_executor_service

    async def __call__(self, state: RuntimeState):
        planner_output = state.planner_output
        assert planner_output is not None
        result = await self._tool_executor_service.execute(
            planner_output=planner_output,
            runtime_context=state.runtime_context,
            request_id=state.request_id,
            correlation_id=state.correlation_id,
            trace_id=state.trace_id or '',
        )
        return {'tool_execution_result': result}
