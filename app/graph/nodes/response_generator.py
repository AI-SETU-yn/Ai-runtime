import logging
from typing import Any

from app.graph.state import RuntimeState
from app.model_gateway.client import ModelGatewayClient
from app.model_gateway.exceptions import ModelGatewayError
from app.prompts.builder import PromptBuilder
from app.utils.json_logging import pretty_json
from app.utils.redaction import redact_sensitive

logger = logging.getLogger(__name__)


class ResponseGeneratorNode:
    def __init__(self, prompt_builder: PromptBuilder, model_gateway_client: ModelGatewayClient) -> None:
        self._prompt_builder = prompt_builder
        self._model_gateway_client = model_gateway_client

    async def __call__(self, state: RuntimeState):
        planner_output = state.planner_output
        assert planner_output is not None
        if self._tool_failed(state.tool_execution_result):
            answer = self._tool_failure_answer(state.tool_execution_result)
            logger.warning('response_generation_skipped_failed_tool_json\n%s', pretty_json(redact_sensitive({
                'planner_intent': planner_output.intent,
                'tool_execution_result': state.tool_execution_result,
                'final_response': answer,
            })))
            return {
                'model_response': answer,
                'final_response': answer,
            }

        prompt = self._prompt_builder.build_response_prompt(
            state.user_question,
            planner_output.intent,
            planner_output.requires_tool,
            state.tool_execution_result,
        )
        logger.info('response_generator_prompt_json\n%s', pretty_json(redact_sensitive({
            'user_question': state.user_question,
            'planner_intent': planner_output.intent,
            'requires_tool': planner_output.requires_tool,
            'domain': planner_output.domain,
            'service': planner_output.service,
            'entity': planner_output.entity,
            'operation': planner_output.operation,
            'parameters': planner_output.parameters,
            'tool_execution_result': state.tool_execution_result,
            'grounded_prompt': prompt,
        })))
        try:
            answer = await self._model_gateway_client.generate(
                prompt,
                metadata={
                    'intent': planner_output.intent,
                    'requires_tool': planner_output.requires_tool,
                    'domain': planner_output.domain,
                    'service': planner_output.service,
                    'entity': planner_output.entity,
                    'operation': planner_output.operation,
                    'parameters': planner_output.parameters,
                    'tool_execution_result': state.tool_execution_result,
                    'conversation_id': state.conversation_id,
                    'trace_id': state.trace_id,
                },
            )
        except ModelGatewayError:
            answer = self._general_answer_failure_answer()
            logger.warning('response_generator_gateway_failed_using_fallback')

        if not isinstance(answer, str) or not answer.strip():
            answer = self._general_answer_failure_answer()
            logger.warning('response_generator_empty_answer_using_fallback')
        logger.info('final_response_json\n%s', pretty_json(redact_sensitive({'final_response': answer})))
        return {
            'model_response': answer,
            'final_response': answer,
        }

    @staticmethod
    def _tool_failed(tool_execution_result: dict[str, object] | None) -> bool:
        if not isinstance(tool_execution_result, dict):
            return False
        if tool_execution_result.get('success') is False or tool_execution_result.get('status') == 'error':
            return True
        data = tool_execution_result.get('data')
        return isinstance(data, dict) and data.get('isError') is True

    @classmethod
    def _tool_failure_answer(cls, tool_execution_result: dict[str, object] | None) -> str:
        detail = cls._extract_tool_error_detail(tool_execution_result)
        if detail:
            return f"I couldn't fetch the requested enterprise data. {detail}"
        return "I couldn't fetch the requested enterprise data because the tool call failed. Please try again after refreshing your session."

    @staticmethod
    def _general_answer_failure_answer() -> str:
        return (
            "I couldn't come up with an answer to that right now. "
            "Please try rephrasing your question or try again in a moment."
        )

    @classmethod
    def _extract_tool_error_detail(cls, value: Any) -> str | None:
        if isinstance(value, dict):
            content = value.get('content')
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and isinstance(item.get('text'), str):
                        return item['text']
            for key in ('content', 'data', 'error', 'message', 'text'):
                detail = cls._extract_tool_error_detail(value.get(key))
                if detail:
                    return detail
        if isinstance(value, list):
            for item in value:
                detail = cls._extract_tool_error_detail(item)
                if detail:
                    return detail
        return value if isinstance(value, str) and value else None
