from pydantic import BaseModel, Field

from app.models.planner import PlannerOutput
from app.models.runtime import RuntimeContext


class RuntimeState(BaseModel):
    conversation_id: str
    request_id: str
    correlation_id: str
    runtime_context: RuntimeContext
    user_question: str
    planner_output: PlannerOutput | None = None
    tool_execution_result: dict[str, object] | None = None
    model_response: str | None = None
    final_response: str | None = None
    trace_id: str | None = None
