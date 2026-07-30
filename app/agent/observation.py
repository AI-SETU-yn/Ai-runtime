from __future__ import annotations

import json
from typing import Any

from app.agent.models import AgentState, Observation


class ObservationManager:
    def from_tool_result(self, state: AgentState) -> Observation | None:
        result = state.tool_execution_result
        if result is None:
            return None
        return Observation(
            iteration=state.iteration_count,
            tool_name=self._tool_name(result),
            arguments=self._arguments(state),
            execution_status=self._string_value(result.get('status')),
            success=result.get('success') if isinstance(result.get('success'), bool) else None,
            latency_ms=self._latency(result),
            result_summary=self._summary(result),
            structured_output=result.get('data'),
            error=result.get('error'),
        )

    @staticmethod
    def _tool_name(result: dict[str, Any]) -> str | None:
        value = result.get('tool_name')
        return value if isinstance(value, str) else None

    @staticmethod
    def _arguments(state: AgentState) -> dict[str, Any]:
        if state.planner_output is None:
            return {}
        return dict(state.planner_output.parameters)

    @staticmethod
    def _latency(result: dict[str, Any]) -> float | None:
        value = result.get('tool_execution_latency_ms')
        if isinstance(value, (int, float)):
            return float(value)
        return None

    @classmethod
    def _summary(cls, result: dict[str, Any]) -> str:
        error = result.get('error')
        if error:
            return cls._compact(error)
        data = result.get('data')
        if data in (None, '', [], {}):
            return 'Tool returned no data.'
        return cls._compact(data)

    @staticmethod
    def _compact(value: Any) -> str:
        text = json.dumps(value, ensure_ascii=True, default=str, separators=(',', ':'))
        return text if len(text) <= 500 else f'{text[:497]}...'

    @staticmethod
    def _string_value(value: Any) -> str | None:
        return value if isinstance(value, str) else None
