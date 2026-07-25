from __future__ import annotations

import logging
from time import perf_counter
from typing import Any

from app.mcp_client import MCPClient
from app.mcp_client.exceptions import (
    AuthenticationError,
    MCPConnectionError,
    MCPTimeoutError,
    ResponseParseError,
    ServerNotFoundError,
    ServerUnavailableError,
)
from app.models.planner import PlannerOutput
from app.models.runtime import RuntimeContext
from app.tool_executor.exceptions import ToolExecutionError, ToolResolutionError
from app.tool_registry.exceptions import DuplicateToolException, ToolNotFoundException
from app.tool_registry.models import ResolvedTool
from app.tool_registry.service import ToolRegistryService
from app.utils.json_logging import pretty_json

logger = logging.getLogger(__name__)


class ToolExecutorService:
    def __init__(self, tool_registry_service: ToolRegistryService, mcp_client: MCPClient) -> None:
        self._tool_registry_service = tool_registry_service
        self._mcp_client = mcp_client

    async def execute(
        self,
        *,
        planner_output: PlannerOutput,
        runtime_context: RuntimeContext,
        request_id: str,
        correlation_id: str,
        trace_id: str,
    ) -> dict[str, Any] | None:
        if not planner_output.requires_tool:
            return None
        if not all((planner_output.domain, planner_output.service, planner_output.entity, planner_output.operation)):
            raise ToolResolutionError(
                'Planner output is missing one or more resolution fields: domain, service, entity, operation.'
            )

        lookup_started = perf_counter()
        resolved_tool = self._resolve_tool(planner_output)
        lookup_latency_ms = round((perf_counter() - lookup_started) * 1000, 2)
        logger.info('tool_registry_lookup_json\n%s', pretty_json({
            'requested_lookup': {
                'domain': planner_output.domain,
                'service': planner_output.service,
                'entity': planner_output.entity,
                'operation': planner_output.operation,
            },
            'resolved_tool': resolved_tool.tool.name,
            'resolved_mcp_server': resolved_tool.server,
            'lookup_latency_ms': lookup_latency_ms,
        }))

        mcp_arguments = self._build_arguments(planner_output.parameters, runtime_context, correlation_id)
        mcp_context = self._build_context(runtime_context, request_id, correlation_id, trace_id)
        logger.info('tool_execution_request_json\n%s', pretty_json({
            'tool': resolved_tool.tool.name,
            'server': resolved_tool.server,
            'arguments': mcp_arguments,
            'context': mcp_context,
        }))

        tool_started = perf_counter()
        try:
            response = await self._mcp_client.invoke_tool(
                resolved_tool.server,
                resolved_tool.tool.name,
                mcp_arguments,
                context=mcp_context,
                trace_id=trace_id,
                request_id=request_id,
            )
        except AuthenticationError as exc:
            raise ToolExecutionError('Authentication failed while calling MCP server.', code='AUTHENTICATION_ERROR', status_code=401) from exc
        except ServerNotFoundError as exc:
            raise ToolExecutionError(str(exc), code='SERVER_ERROR', status_code=502) from exc
        except MCPTimeoutError as exc:
            raise ToolExecutionError('MCP server request timed out.', code='TIMEOUT', status_code=504) from exc
        except (MCPConnectionError, ServerUnavailableError) as exc:
            raise ToolExecutionError('Transport error while calling MCP server.', code='TRANSPORT_ERROR', status_code=502) from exc
        except ResponseParseError as exc:
            raise ToolExecutionError('Invalid MCP server response.', code='SERVER_ERROR', status_code=502) from exc

        execution_latency_ms = round((perf_counter() - tool_started) * 1000, 2)
        normalized = {
            'tool_name': resolved_tool.tool.name,
            'server': resolved_tool.server,
            'status': response.status.value,
            'success': response.success,
            'response_type': resolved_tool.tool.response_type.value,
            'data': response.tool_results,
            'error': response.error.model_dump() if response.error else None,
            'registry_lookup_latency_ms': lookup_latency_ms,
            'tool_execution_latency_ms': execution_latency_ms,
        }
        logger.info('tool_execution_response_json\n%s', pretty_json(normalized))
        logger.info(
            'tool_execution_completed planner_latency_ms=%s registry_lookup_latency_ms=%s tool_execution_latency_ms=%s mcp_server=%s tool_name=%s status=%s duration=%s',
            0.0,
            lookup_latency_ms,
            execution_latency_ms,
            resolved_tool.server,
            resolved_tool.tool.name,
            response.status.value,
            execution_latency_ms,
        )
        return normalized

    def _resolve_tool(self, planner_output: PlannerOutput) -> ResolvedTool:
        domain = planner_output.domain or ''
        service = planner_output.service or ''
        entity = planner_output.entity or ''
        operation = planner_output.operation or ''
        try:
            return self._tool_registry_service.find_tool(domain, service, entity, operation)
        except ToolNotFoundException as exc:
            raise ToolResolutionError(
                f'No tool found for domain={domain} service={service} entity={entity} operation={operation}'
            ) from exc
        except DuplicateToolException as exc:
            raise ToolResolutionError(
                f'Multiple tools found for domain={domain} service={service} entity={entity} operation={operation}'
            ) from exc

    @staticmethod
    def _build_arguments(parameters: dict[str, object], runtime_context: RuntimeContext, correlation_id: str) -> dict[str, Any]:
        arguments = dict(parameters)
        arguments['context'] = {
            'organization_id': runtime_context.organization_id or '',
            'branch_id': runtime_context.branch_id or '',
            'user_id': runtime_context.user_id,
            'correlation_id': correlation_id,
        }
        return arguments

    @staticmethod
    def _build_context(
        runtime_context: RuntimeContext,
        request_id: str,
        correlation_id: str,
        trace_id: str,
    ) -> dict[str, Any]:
        return {
            'request_id': request_id,
            'correlation_id': correlation_id,
            'trace_id': trace_id,
            'user_id': runtime_context.user_id,
            'org_id': runtime_context.organization_id,
            'branch_id': runtime_context.branch_id,
            'roles': runtime_context.roles,
            'jwt': runtime_context.jwt,
            'locale': runtime_context.locale,
            'tenant_id': runtime_context.tenant_id,
        }
