from __future__ import annotations

from app.tool_registry.service import ToolRegistryService


class ToolDiscovery:
    def __init__(self, tool_registry_service: ToolRegistryService | None = None) -> None:
        self._tool_registry_service = tool_registry_service

    def available_tools(self) -> list[str]:
        if self._tool_registry_service is None:
            return []
        tools: list[str] = []
        for registry in self._tool_registry_service.repository.registries:
            for tool in registry.tools:
                tools.append(
                    f'{registry.domain}.{registry.service}.{tool.entity}.{tool.operation.value}:{tool.name}'
                )
        return tools
