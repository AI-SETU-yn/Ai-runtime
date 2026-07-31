import logging
import time
import uuid

from app.graph.graph import WorkflowManager
from app.graph.state import RuntimeState
from app.guardrails import GuardrailEngine
from app.models.chat import ChatRequest
from app.models.response import ChatResponse, ConversationMetadata, GuardrailMetadata, GuardrailOutcome, SecurityMetadata
from app.models.runtime import RuntimeContext
from app.security.service import SecurityClassificationService
from app.utils.json_logging import pretty_json

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(
        self,
        workflow_manager: WorkflowManager,
        guardrail_engine: GuardrailEngine,
        security_classifier_service: SecurityClassificationService,
    ) -> None:
        self._workflow_manager = workflow_manager
        self._guardrail_engine = guardrail_engine
        self._security_classifier_service = security_classifier_service

    async def chat(self, request: ChatRequest, runtime_context: RuntimeContext, request_id: str, correlation_id: str) -> ChatResponse:
        started_at = time.perf_counter()
        conversation_id = request.conversation_id or str(uuid.uuid4())
        trace_id = runtime_context.trace_id or str(uuid.uuid4())
        defer_tags = self._security_classifier_service.config.suspicious_tags if self._security_classifier_service.should_defer_input_blocks() else []
        input_guardrail = self._guardrail_engine.enforce_input(request.message, defer_block_tags=defer_tags)
        security_result = await self._security_classifier_service.classify_if_needed(
            message=input_guardrail.final_text,
            guardrail_result=input_guardrail,
        )
        self._security_classifier_service.ensure_safe(security_result)
        runtime_context = runtime_context.model_copy(
            update={
                'request_id': request_id,
                'correlation_id': correlation_id,
                'trace_id': trace_id,
            }
        )

        logger.info('chat_request_json\n%s', pretty_json({
            'message': input_guardrail.final_text,
            'conversation_id': conversation_id,
            'request_id': request_id,
            'correlation_id': correlation_id,
            'trace_id': trace_id,
            'guardrails': self._serialize_guardrails(input_guardrail),
            'security': self._serialize_security(security_result),
            'runtime_context': runtime_context.model_dump(exclude={'jwt'}),
        }))

        state = RuntimeState(
            conversation_id=conversation_id,
            request_id=request_id,
            correlation_id=correlation_id,
            runtime_context=runtime_context,
            user_question=input_guardrail.final_text,
            trace_id=trace_id,
        )
        final_state = await self._workflow_manager.run(state)
        output_guardrail = self._guardrail_engine.enforce_output(final_state.final_response or '')
        elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)

        planner_output = final_state.planner_output
        logger.info(
            'chat_completed execution_time_ms=%s trace_id=%s planner_intent=%s conversation_id=%s request_id=%s correlation_id=%s input_guardrail_hits=%s output_guardrail_hits=%s security_executed=%s',
            elapsed_ms,
            trace_id,
            planner_output.intent if planner_output else None,
            conversation_id,
            request_id,
            correlation_id,
            len(input_guardrail.triggered),
            len(output_guardrail.triggered),
            security_result.executed,
        )

        return ChatResponse(
            answer=output_guardrail.final_text,
            metadata=ConversationMetadata(
                conversation_id=conversation_id,
                request_id=request_id,
                correlation_id=correlation_id,
                execution_time_ms=elapsed_ms,
                planner_intent=planner_output.intent if planner_output else None,
                requires_tool=planner_output.requires_tool if planner_output else False,
                trace_id=trace_id,
                guardrails=GuardrailMetadata(
                    input=self._to_outcomes(input_guardrail),
                    output=self._to_outcomes(output_guardrail),
                ),
                security=self._to_security_metadata(security_result),
            ),
        )

    @staticmethod
    def _to_outcomes(result) -> list[GuardrailOutcome]:
        return [
            GuardrailOutcome(
                stage=result.stage,
                action=decision.action,
                rule_id=decision.rule_id,
                tags=decision.tags,
            )
            for decision in result.triggered
        ]

    @staticmethod
    def _serialize_guardrails(result) -> dict[str, object]:
        return {
            'stage': result.stage,
            'redacted': result.redacted,
            'triggered': [decision.model_dump() for decision in result.triggered],
        }

    @staticmethod
    def _serialize_security(result) -> dict[str, object]:
        return ChatService._to_security_metadata(result).model_dump() if result.executed else {'executed': False}

    @staticmethod
    def _to_security_metadata(result) -> SecurityMetadata:
        decision = result.decision
        return SecurityMetadata(
            executed=result.executed,
            triggered_by=result.triggered_by,
            safe=decision.safe if decision else None,
            category=decision.category.value if decision else None,
            confidence=decision.confidence if decision else None,
            reason=decision.reason if decision else None,
        )