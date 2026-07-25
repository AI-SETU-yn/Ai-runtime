import logging
import time
import uuid

from app.graph.graph import WorkflowManager
from app.graph.state import RuntimeState
from app.models.chat import ChatRequest
from app.models.response import ChatResponse, ConversationMetadata
from app.models.runtime import RuntimeContext
from app.utils.json_logging import pretty_json

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(self, workflow_manager: WorkflowManager) -> None:
        self._workflow_manager = workflow_manager

    async def chat(self, request: ChatRequest, runtime_context: RuntimeContext, request_id: str, correlation_id: str) -> ChatResponse:
        started_at = time.perf_counter()
        conversation_id = request.conversation_id or str(uuid.uuid4())
        trace_id = str(uuid.uuid4())

        logger.info('chat_request_json\n%s', pretty_json({
            'message': request.message,
            'conversation_id': conversation_id,
            'request_id': request_id,
            'correlation_id': correlation_id,
            'trace_id': trace_id,
            'runtime_context': runtime_context.model_dump(exclude={'jwt'}),
        }))

        state = RuntimeState(
            conversation_id=conversation_id,
            request_id=request_id,
            correlation_id=correlation_id,
            runtime_context=runtime_context,
            user_question=request.message,
            trace_id=trace_id,
        )
        final_state = await self._workflow_manager.run(state)
        elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)

        planner_output = final_state.planner_output
        logger.info(
            'chat_completed execution_time_ms=%s trace_id=%s planner_intent=%s conversation_id=%s request_id=%s correlation_id=%s',
            elapsed_ms,
            trace_id,
            planner_output.intent if planner_output else None,
            conversation_id,
            request_id,
            correlation_id,
        )

        return ChatResponse(
            answer=final_state.final_response or '',
            metadata=ConversationMetadata(
                conversation_id=conversation_id,
                request_id=request_id,
                correlation_id=correlation_id,
                execution_time_ms=elapsed_ms,
                planner_intent=planner_output.intent if planner_output else None,
                requires_tool=planner_output.requires_tool if planner_output else False,
                trace_id=trace_id,
            ),
        )
