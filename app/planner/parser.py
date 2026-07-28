import json
from typing import Any

from app.models.planner import PlannerOutput


class PlannerOutputParser:
    def parse(
        self,
        *,
        intent: str | None,
        domain: str | None,
        service: str | None,
        entity: str | None,
        operation: str | None,
        tool: str | None,
        parameters: dict[str, object] | None,
        execution_plan: list[dict[str, object]] | None = None,
        requires_tool: bool | None,
        raw_response: str | None,
        adapter: str | None,
        model: str | None,
    ) -> PlannerOutput:
        raw_payload = self._parse_raw_response(raw_response)
        normalized_parameters = parameters or self._dict_value(raw_payload.get('parameters')) or {}
        normalized_requires_tool = requires_tool
        if normalized_requires_tool is None:
            normalized_requires_tool = self._bool_value(raw_payload.get('requiresTool'))
        if normalized_requires_tool is None:
            normalized_requires_tool = self._bool_value(raw_payload.get('requires_tool'))
        if normalized_requires_tool is None:
            normalized_requires_tool = bool(tool or raw_payload.get('tool'))
        normalized_execution_plan = (
            execution_plan
            or self._list_value(raw_payload.get('execution_plan'))
            or self._list_value(raw_payload.get('executionPlan'))
            or []
        )
        return PlannerOutput(
            intent=intent or self._str_value(raw_payload.get('intent')) or '',
            requires_tool=normalized_requires_tool,
            domain=domain or self._str_value(raw_payload.get('domain')),
            service=service or self._str_value(raw_payload.get('service')),
            entity=entity or self._str_value(raw_payload.get('entity')),
            operation=operation or self._str_value(raw_payload.get('operation')),
            parameters=normalized_parameters,
            tool=tool or self._str_value(raw_payload.get('tool')),
            rationale=None,
            raw_response=raw_response,
            adapter=adapter or self._str_value(raw_payload.get('adapter')),
            model=model or self._str_value(raw_payload.get('model')),
            execution_plan=normalized_execution_plan,
        )

    @classmethod
    def _parse_raw_response(cls, raw_response: str | None) -> dict[str, Any]:
        if not raw_response:
            return {}
        decoder = json.JSONDecoder()
        for index, char in enumerate(raw_response):
            if char != '{':
                continue
            try:
                payload, _ = decoder.raw_decode(raw_response[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload
        return {}

    @staticmethod
    def _str_value(value: Any) -> str | None:
        return value if isinstance(value, str) and value else None

    @staticmethod
    def _dict_value(value: Any) -> dict[str, object] | None:
        return value if isinstance(value, dict) else None

    @staticmethod
    def _list_value(value: Any) -> list[dict[str, object]] | None:
        if not isinstance(value, list):
            return None
        if not all(isinstance(item, dict) for item in value):
            return None
        return value

    @staticmethod
    def _bool_value(value: Any) -> bool | None:
        return value if isinstance(value, bool) else None
