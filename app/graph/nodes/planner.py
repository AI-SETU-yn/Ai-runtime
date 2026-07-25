import uuid

from app.graph.state import RuntimeState
from app.planner.planner import PlannerService


class PlannerNode:
    def __init__(self, planner_service: PlannerService) -> None:
        self._planner_service = planner_service

    async def __call__(self, state: RuntimeState):
        planner_output = await self._planner_service.plan(state.user_question)
        return {
            'planner_output': planner_output,
            'trace_id': state.trace_id or str(uuid.uuid4()),
        }
