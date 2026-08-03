"""Validate Tool Registry declarations against live MCP tools/list metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.mcp_client.client.client import MCPClient
from app.tool_registry.models import ToolDefinition
from app.tool_registry.service import ToolRegistryService


@dataclass(frozen=True)
class ToolContractMismatch:
    server: str
    tool: str
    issue: str
    details: dict[str, Any] = field(default_factory=dict)


class ToolContractValidationError(RuntimeError):
    def __init__(self, mismatches: list[ToolContractMismatch]) -> None:
        self.mismatches = mismatches
        super().__init__(f'Tool Registry and MCP contract validation failed with {len(mismatches)} mismatch(es).')


class ToolContractValidator:
    """Compare registry tool names and input parameters with MCP tools/list."""

    def __init__(self, registry_service: ToolRegistryService, mcp_client: MCPClient) -> None:
        self._registry_service = registry_service
        self._mcp_client = mcp_client

    async def validate(self) -> list[ToolContractMismatch]:
        mismatches: list[ToolContractMismatch] = []
        expected_by_server = self._expected_tools_by_server()
        for server, expected_tools in expected_by_server.items():
            remote_tools = await self._mcp_client.list_tools(server)
            remote_by_name = {
                str(tool.get('name')): tool
                for tool in remote_tools
                if isinstance(tool.get('name'), str) and str(tool.get('name')).strip()
            }
            mismatches.extend(self._missing_tools(server, expected_tools, remote_by_name))
            mismatches.extend(self._unexpected_tools(server, expected_tools, remote_by_name))
            mismatches.extend(self._parameter_mismatches(server, expected_tools, remote_by_name))
        return mismatches

    async def validate_or_raise_async(self) -> None:
        mismatches = await self.validate()
        if mismatches:
            raise ToolContractValidationError(mismatches)

    def _expected_tools_by_server(self) -> dict[str, dict[str, ToolDefinition]]:
        expected: dict[str, dict[str, ToolDefinition]] = {}
        for registry in self._registry_service.repository.registries:
            tools = expected.setdefault(registry.server, {})
            for tool in registry.tools:
                tools[tool.name] = tool
        return expected

    @staticmethod
    def _missing_tools(
        server: str,
        expected_tools: dict[str, ToolDefinition],
        remote_by_name: dict[str, dict[str, Any]],
    ) -> list[ToolContractMismatch]:
        return [
            ToolContractMismatch(
                server=server,
                tool=tool_name,
                issue='missing_mcp_tool',
                details={'registry_tool': tool_name},
            )
            for tool_name in sorted(set(expected_tools) - set(remote_by_name))
        ]

    @staticmethod
    def _unexpected_tools(
        server: str,
        expected_tools: dict[str, ToolDefinition],
        remote_by_name: dict[str, dict[str, Any]],
    ) -> list[ToolContractMismatch]:
        return [
            ToolContractMismatch(
                server=server,
                tool=tool_name,
                issue='unexpected_mcp_tool',
                details={'mcp_tool': tool_name},
            )
            for tool_name in sorted(set(remote_by_name) - set(expected_tools))
        ]

    @classmethod
    def _parameter_mismatches(
        cls,
        server: str,
        expected_tools: dict[str, ToolDefinition],
        remote_by_name: dict[str, dict[str, Any]],
    ) -> list[ToolContractMismatch]:
        mismatches: list[ToolContractMismatch] = []
        for tool_name in sorted(set(expected_tools).intersection(remote_by_name)):
            expected_tool = expected_tools[tool_name]
            remote_schema = cls._input_schema(remote_by_name[tool_name])
            remote_properties = cls._schema_properties(remote_schema)
            if not remote_properties:
                continue
            expected_parameters = set(expected_tool.required_parameters + expected_tool.optional_parameters)
            remote_parameters = set(remote_properties) - {'context'}
            missing_parameters = sorted(expected_parameters - remote_parameters)
            if missing_parameters:
                mismatches.append(
                    ToolContractMismatch(
                        server=server,
                        tool=tool_name,
                        issue='parameter_missing_in_mcp_schema',
                        details={
                            'missing_parameters': missing_parameters,
                            'registry_parameters': sorted(expected_parameters),
                            'mcp_parameters': sorted(remote_parameters),
                        },
                    )
                )
            required_remote = cls._schema_required(remote_schema)
            missing_required = sorted(set(expected_tool.required_parameters) - required_remote - {'context'})
            if missing_required:
                mismatches.append(
                    ToolContractMismatch(
                        server=server,
                        tool=tool_name,
                        issue='required_parameter_not_marked_required_in_mcp_schema',
                        details={
                            'missing_required_parameters': missing_required,
                            'registry_required_parameters': sorted(expected_tool.required_parameters),
                            'mcp_required_parameters': sorted(required_remote),
                        },
                    )
                )
        return mismatches

    @staticmethod
    def _input_schema(tool: dict[str, Any]) -> dict[str, Any]:
        schema = tool.get('inputSchema') or tool.get('input_schema') or {}
        return schema if isinstance(schema, dict) else {}

    @staticmethod
    def _schema_properties(schema: dict[str, Any]) -> dict[str, Any]:
        properties = schema.get('properties')
        return properties if isinstance(properties, dict) else {}

    @staticmethod
    def _schema_required(schema: dict[str, Any]) -> set[str]:
        required = schema.get('required')
        if not isinstance(required, list):
            return set()
        return {item for item in required if isinstance(item, str)}
