from __future__ import annotations

from app.models.planner import PlannerOutput
from app.tool_registry.exceptions import DuplicateToolException, ToolNotFoundException
from app.tool_registry.models import ResolvedTool
from app.tool_registry.service import ToolRegistryService


class ModelGatewayAdapterResolver:
    """Resolve model-gateway adapter hints from registry metadata."""

    def __init__(self, tool_registry_service: ToolRegistryService | None = None) -> None:
        self._tool_registry_service = tool_registry_service

    def resolve_for_planning(self) -> str:
        adapters = self._registered_adapters()
        return next(iter(adapters)) if len(adapters) == 1 else ''

    def resolve_for_response(self, planner_output: PlannerOutput, response_type: str) -> str:
        normalized_type = response_type.strip().casefold()
        if normalized_type in {'general', 'current_info', 'planner_failure'}:
            return ''
        resolved_tool = self._resolved_tool_for_output(planner_output)
        if resolved_tool is not None:
            return resolved_tool.adapter or ''
        return planner_output.adapter or ''

    def _registered_adapters(self) -> set[str]:
        if self._tool_registry_service is None:
            return set()
        adapters: set[str] = set()
        for registry in self._tool_registry_service.repository.registries:
            if registry.adapter:
                adapters.add(registry.adapter)
            for tool in registry.tools:
                if tool.adapter:
                    adapters.add(tool.adapter)
        return adapters

    def _resolved_tool_for_output(self, planner_output: PlannerOutput) -> ResolvedTool | None:
        if self._tool_registry_service is None:
            return None
        target = self._target(planner_output)
        if target is None:
            return None
        domain, service, entity, operation = target
        try:
            return self._tool_registry_service.find_tool(domain, service, entity, operation)
        except (ToolNotFoundException, DuplicateToolException):
            return None

    @staticmethod
    def _target(planner_output: PlannerOutput) -> tuple[str, str, str, str] | None:
        step = planner_output.execution_plan[-1] if planner_output.execution_plan else None
        domain = (step.domain if step else planner_output.domain) or ''
        service = (step.service if step else planner_output.service) or ''
        entity = (step.entity if step else planner_output.entity) or ''
        operation = (step.operation if step else planner_output.operation) or ''
        if all((domain, service, entity, operation)):
            return domain, service, entity, operation
        return None
