import logging
import re
from typing import Any

from app.graph.state import RuntimeState
from app.model_gateway.adapter_resolver import ModelGatewayAdapterResolver
from app.model_gateway.client import ModelGatewayClient
from app.model_gateway.exceptions import ModelGatewayError
from app.models.planner import PlannerOutput
from app.utils.json_logging import pretty_json

logger = logging.getLogger(__name__)

_MODEL_GATEWAY_UNAVAILABLE = 'The AI service is temporarily unavailable.'


class ResponseGeneratorNode:
    def __init__(
        self,
        model_gateway_client: ModelGatewayClient,
        adapter_resolver: ModelGatewayAdapterResolver | None = None,
    ) -> None:
        self._model_gateway_client = model_gateway_client
        self._adapter_resolver = adapter_resolver or ModelGatewayAdapterResolver()
        self._grounding_validator = GroundingValidator()

    async def __call__(self, state: RuntimeState):
        planner_output = state.planner_output
        assert planner_output is not None

        response_tool_result = state.tool_execution_result
        response_type = self._response_type(planner_output, response_tool_result)
        request = self._build_generate_request(
            state,
            planner_output,
            response_tool_result,
            response_type=response_type,
        )
        logger.info('response_generator_dispatch_json\n%s', pretty_json({
            'response_type': response_type,
            'conversation': request['conversation'],
            'metadata': request['metadata'],
            'adapter': request['adapter'],
            'tool_result': response_tool_result,
        }))

        try:
            answer = await self._dispatch_generate(request)
        except ModelGatewayError:
            logger.warning('response_generator_gateway_unavailable')
            return {'model_response': _MODEL_GATEWAY_UNAVAILABLE, 'final_response': _MODEL_GATEWAY_UNAVAILABLE}

        if not answer.strip():
            logger.warning('response_generator_empty_gateway_response')
            return {'model_response': _MODEL_GATEWAY_UNAVAILABLE, 'final_response': _MODEL_GATEWAY_UNAVAILABLE}

        if not self._grounding_validator.validate(answer, planner_output, response_tool_result, response_type):
            logger.warning('response_generator_grounding_validation_failed_json\n%s', pretty_json({
                'response_type': response_type,
                'planner_intent': planner_output.intent,
                'tool_result': response_tool_result,
                'model_response': answer,
            }))
            retry_request = self._with_validation_retry(request, answer)
            try:
                retry_answer = await self._dispatch_generate(retry_request)
            except ModelGatewayError:
                logger.warning('response_generator_validation_retry_gateway_unavailable')
                return {'model_response': _MODEL_GATEWAY_UNAVAILABLE, 'final_response': _MODEL_GATEWAY_UNAVAILABLE}
            if retry_answer.strip():
                answer = retry_answer

        logger.info('final_response_json\n%s', pretty_json({'final_response': answer}))
        return {'model_response': answer, 'final_response': answer}

    async def _dispatch_generate(self, request: dict[str, Any]) -> str:
        return await self._model_gateway_client.generate(
            messages=request['messages'],
            tool_result=request.get('toolResult'),
            response_type=request.get('responseType'),
            generation_policy=request.get('generationPolicy'),
            conversation=request.get('conversation'),
            metadata=request.get('metadata'),
            missing_parameters=request.get('missingParameters'),
            adapter=request.get('adapter'),
        )

    def _build_generate_request(
        self,
        state: RuntimeState,
        planner_output: PlannerOutput,
        tool_result: dict[str, object] | None,
        *,
        response_type: str,
    ) -> dict[str, Any]:
        missing_parameters = self._missing_parameters(tool_result)
        adapter = self._adapter_resolver.resolve_for_response(planner_output, response_type)
        return {
            'adapter': adapter,
            'messages': [{'role': 'user', 'content': state.user_question}],
            'toolResult': tool_result,
            'responseType': response_type,
            'generationPolicy': {
                'grounded': planner_output.requires_tool or tool_result is not None,
                'hallucination': 'forbid',
                'format': 'markdown',
                'useToolResultsOnly': planner_output.requires_tool or tool_result is not None,
                'neverInventBusinessData': True,
                'neverAskForPresentData': True,
                'ignoreOrchestrationMetadata': True,
                'concise': True,
            },
            'conversation': {
                'conversationId': state.conversation_id,
                'requestId': state.request_id,
                'correlationId': state.correlation_id,
                'traceId': state.trace_id,
                'userQuestion': state.user_question,
                'plannerIntent': planner_output.intent,
                'requiresTool': planner_output.requires_tool,
                'domain': planner_output.domain,
                'service': planner_output.service,
                'entity': planner_output.entity,
                'operation': planner_output.operation,
                'parameters': planner_output.parameters,
                'executionPlan': [step.model_dump(by_alias=True, exclude_none=True) for step in planner_output.execution_plan],
            },
            'metadata': {
                'runtime': 'ai-runtime-v2',
                'response_type': response_type,
                'awaiting_input': self._tool_awaiting_input(tool_result),
                'tool_failed': self._tool_failed(tool_result),
            },
            'missingParameters': missing_parameters,
        }

    @staticmethod
    def _with_validation_retry(request: dict[str, Any], previous_answer: str) -> dict[str, Any]:
        retry_request = dict(request)
        retry_request['responseType'] = 'validation_retry'
        retry_request['metadata'] = {
            **dict(request.get('metadata') or {}),
            'previous_response': previous_answer,
            'validation_retry': True,
        }
        retry_request['generationPolicy'] = {
            **dict(request.get('generationPolicy') or {}),
            'grounded': True,
            'hallucination': 'forbid',
        }
        return retry_request

    def _response_type(
        self,
        planner_output: PlannerOutput,
        tool_result: dict[str, object] | None,
    ) -> str:
        if self._is_planner_failure(planner_output):
            return 'planner_failure'
        if self._is_web_search_result(tool_result):
            return 'current_info'
        if self._authorization_failed(tool_result):
            return 'authorization_failure'
        if self._tool_awaiting_input(tool_result):
            return 'clarification'
        if self._tool_failed(tool_result):
            return 'tool_failure'
        if planner_output.requires_tool and not self._has_tool_data(tool_result):
            return 'empty_result'
        if planner_output.requires_tool and len(planner_output.execution_plan) > 1:
            return 'multi_tool'
        if planner_output.requires_tool:
            return 'enterprise'
        return 'general'

    @staticmethod
    def _is_planner_failure(planner_output: PlannerOutput) -> bool:
        intent = planner_output.intent.strip().casefold()
        raw_response = (planner_output.raw_response or '').strip().casefold()
        return intent in {'planner.failure', 'planner_failure'} or raw_response.startswith('planner_failure:')

    @staticmethod
    def _tool_awaiting_input(tool_execution_result: dict[str, object] | None) -> bool:
        return isinstance(tool_execution_result, dict) and (
            tool_execution_result.get('awaiting_input') is True
            or tool_execution_result.get('status') == 'requires_input'
        )

    @staticmethod
    def _tool_failed(tool_execution_result: dict[str, object] | None) -> bool:
        if not isinstance(tool_execution_result, dict):
            return False
        if tool_execution_result.get('success') is False or tool_execution_result.get('status') == 'error':
            return True
        data = tool_execution_result.get('data')
        return isinstance(data, dict) and data.get('isError') is True

    @staticmethod
    def _authorization_failed(tool_execution_result: dict[str, object] | None) -> bool:
        if not isinstance(tool_execution_result, dict):
            return False
        if tool_execution_result.get('response_type') == 'authorization_failure':
            return True
        error = tool_execution_result.get('error')
        return isinstance(error, dict) and error.get('code') == 'AUTHORIZATION_FAILED'

    @staticmethod
    def _is_web_search_result(tool_execution_result: dict[str, object] | None) -> bool:
        if not isinstance(tool_execution_result, dict):
            return False
        data = tool_execution_result.get('data')
        return isinstance(data, dict) and data.get('source') == 'web_search'

    @staticmethod
    def _has_tool_data(tool_execution_result: dict[str, object] | None) -> bool:
        if not isinstance(tool_execution_result, dict):
            return False
        if tool_execution_result.get('steps'):
            return True
        data = tool_execution_result.get('data')
        return data not in (None, '', {}, [])

    @staticmethod
    def _missing_parameters(tool_execution_result: dict[str, object] | None) -> list[str]:
        if not isinstance(tool_execution_result, dict):
            return []
        clarification = tool_execution_result.get('clarification')
        if not isinstance(clarification, dict):
            return []
        missing = clarification.get('missing_parameters') or clarification.get('missingParameters')
        return [str(item) for item in missing] if isinstance(missing, list) else []


class GroundingValidator:
    def validate(
        self,
        answer: str,
        planner_output: PlannerOutput,
        enterprise_data: Any,
        response_type: str,
    ) -> bool:
        if response_type in {'general', 'clarification', 'tool_failure', 'planner_failure', 'authorization_failure'}:
            return True
        enterprise_data = self._extract_factual_data(enterprise_data)
        if not planner_output.requires_tool or enterprise_data in (None, '', {}, []):
            return True
        scalar_values = self._scalar_values(enterprise_data)
        if not scalar_values:
            return True
        normalized_answer = self._normalize_text(answer)
        if any(self._normalize_text(value) in normalized_answer for value in scalar_values):
            return True
        record_count = self._record_count(enterprise_data)
        return record_count is not None and str(record_count) in answer

    @classmethod
    def _extract_factual_data(cls, value: Any) -> Any:
        parsed = cls._parse_json_string(value)
        if parsed is not value:
            return cls._extract_factual_data(parsed)
        if isinstance(value, dict):
            content = value.get('content')
            if isinstance(content, list):
                extracted = [cls._extract_content_item(item) for item in content]
                extracted = [item for item in extracted if item not in (None, '')]
                if len(extracted) == 1:
                    return extracted[0]
                if extracted:
                    return extracted
            for key in ('toolResults', 'tool_results'):
                if key in value:
                    return cls._extract_factual_data(value[key])
            if 'steps' in value and isinstance(value['steps'], list):
                return [cls._extract_factual_data(step) for step in value['steps']]
            if 'result' in value:
                return cls._extract_factual_data(value['result'])
            if 'data' in value:
                return cls._extract_factual_data(value['data'])
            return {
                key: cls._extract_factual_data(item)
                for key, item in value.items()
                if not cls._is_internal_field(str(key))
            }
        if isinstance(value, list):
            return [cls._extract_factual_data(item) for item in value]
        return value

    @classmethod
    def _extract_content_item(cls, item: Any) -> Any:
        if isinstance(item, dict) and isinstance(item.get('text'), str):
            return cls._extract_factual_data(item['text'])
        return cls._extract_factual_data(item)

    @classmethod
    def _scalar_values(cls, value: Any) -> set[str]:
        values: set[str] = set()
        if isinstance(value, dict):
            for key, item in value.items():
                if cls._is_internal_field(str(key)):
                    continue
                values.update(cls._scalar_values(item))
            return values
        if isinstance(value, list):
            for item in value:
                values.update(cls._scalar_values(item))
            return values
        if isinstance(value, (str, int, float)) and not isinstance(value, bool):
            text = str(value).strip()
            if len(text) >= 2:
                values.add(text)
        return values

    @classmethod
    def _record_count(cls, value: Any) -> int | None:
        if isinstance(value, list):
            return len(value)
        if isinstance(value, dict):
            for item in value.values():
                if isinstance(item, list):
                    return len(item)
        return None

    @staticmethod
    def _is_internal_field(key: str) -> bool:
        lowered = key.casefold()
        return (
            lowered in {'metadata', 'meta', 'internal'}
            or any(token in lowered for token in ('token', 'secret', 'password', 'authorization'))
        )

    @staticmethod
    def _normalize_text(value: str) -> str:
        return re.sub(r'\s+', ' ', value.casefold()).strip()

    @staticmethod
    def _parse_json_string(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        stripped = value.strip()
        if not stripped or stripped[0] not in '[{':
            return value
        try:
            import json

            return json.loads(stripped)
        except ValueError:
            return value
