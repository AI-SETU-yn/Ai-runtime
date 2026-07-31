import json
from typing import Any

from app.prompts.templates import PLANNER_PROMPT, RESPONSE_SYSTEM_PROMPT


class PromptBuilder:
    _METADATA_KEYS = {
        'tool_name',
        'server',
        'success',
        'response_type',
        'error',
        'registry_lookup_latency_ms',
        'tool_execution_latency_ms',
        'executionTime',
        'metadata',
        'isError',
        'statusCode',
        'errors',
    }

    def build_planner_prompt(self, message: str, *, registry_context: str | None = None) -> str:
        registry_section = ''
        if registry_context:
            registry_section = (
                '\n\nRegistered tool targets. If requires_tool is true, choose only one of these exact '
                f'domain/service/entity/operation combinations and derive intent as service.entity.operation:\n'
                f'{registry_context}\n\n'
                'If one user request needs multiple tools, return execution_plan as an ordered list of steps. '
                'If the user asks for multiple independent datasets, include one step per requested dataset even when no step depends on another. '
                'Each step must include step_id, domain, service, entity, operation, intent, parameters, '
                'depends_on, and optional parameter_bindings. Use parameter_bindings when a parameter must be '
                'read from a previous step, for example: '
                '{"targetParameter":{"from_step":"step_1","path":"$.data[?(@.field==true)].referenceId"}}. '
                'The path must point to factual data returned by a previous step.'
            )
        return f"{PLANNER_PROMPT}{registry_section}\n\nUser message:\n{message}"

    def build_response_prompt(
        self,
        message: str,
        planner_intent: str,
        requires_tool: bool,
        tool_execution_result: dict[str, object] | None = None,
    ) -> str:
        enterprise_context = self._build_enterprise_context(tool_execution_result)
        return (
            f"{RESPONSE_SYSTEM_PROMPT}\n\n"
            f"Planner intent: {planner_intent}\n"
            f"Requires tool: {requires_tool}\n"
            f"Enterprise data: {enterprise_context}\n\n"
            f"User message:\n{message}"
        )

    def extract_enterprise_data(self, tool_execution_result: dict[str, object] | None) -> Any:
        if not tool_execution_result:
            return None
        business_data = self._extract_business_data(tool_execution_result)
        return self._strip_metadata(business_data)

    def _build_enterprise_context(self, tool_execution_result: dict[str, object] | None) -> str:
        enterprise_data = self.extract_enterprise_data(tool_execution_result)
        if enterprise_data in (None, {}):
            return '{}'
        return json.dumps(enterprise_data, ensure_ascii=True, default=str, separators=(',', ':'))

    @classmethod
    def _extract_business_data(cls, value: Any) -> Any:
        parsed = cls._parse_json_string(value)
        if parsed is not value:
            return cls._extract_business_data(parsed)

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
                    return cls._extract_business_data(value[key])

            if 'data' in value:
                data = cls._extract_business_data(value['data'])
                if data not in (None, {}, []):
                    return data
                message = value.get('message')
                if isinstance(message, str) and message:
                    return {'message': message}

            return {
                key: cls._extract_business_data(item)
                for key, item in value.items()
                if key not in cls._METADATA_KEYS
            }

        if isinstance(value, list):
            return [cls._extract_business_data(item) for item in value]

        return value

    @classmethod
    def _extract_content_item(cls, item: Any) -> Any:
        if isinstance(item, dict) and isinstance(item.get('text'), str):
            return cls._extract_business_data(item['text'])
        return cls._extract_business_data(item)

    @classmethod
    def _strip_metadata(cls, value: Any) -> Any:
        if isinstance(value, dict):
            metadata_keys = set(cls._METADATA_KEYS)
            if any(key in value for key in ('tool_name', 'response_type', 'registry_lookup_latency_ms')):
                metadata_keys.add('status')
            return {
                key: cls._strip_metadata(item)
                for key, item in value.items()
                if key not in metadata_keys
            }
        if isinstance(value, list):
            return [cls._strip_metadata(item) for item in value]
        return value

    @staticmethod
    def _parse_json_string(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        stripped = value.strip()
        if not stripped or stripped[0] not in '[{':
            return value
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return value
