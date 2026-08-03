"""Primary high-level MCP client API."""
import logging
from typing import Any
from ..connection.health import HealthMonitor
from ..connection.session import ConnectionSession
from ..executor.tool_executor import ToolExecutor
from ..exceptions import ResponseParseError
from ..models.response import HealthStatus, ToolResponse
from ..models.server import ServerConfig
from ..utils.retry import RetryPolicy, retry_async
from ...utils.json_logging import pretty_json
from ...utils.redaction import redact_sensitive
from .factory import MCPClientFactory
from .manager import MCPClientManager

logger = logging.getLogger(__name__)


class MCPClient:
    """Async, typed gateway to pre-existing enterprise MCP servers."""
    def __init__(self, manager: MCPClientManager | None = None, executor: ToolExecutor | None = None, retry_policy: RetryPolicy | None = None) -> None:
        self._manager = manager or MCPClientManager()
        self._executor, self._retry = executor or ToolExecutor(), retry_policy or RetryPolicy()
        self._factory = MCPClientFactory(self._manager.pool)
        self._health = HealthMonitor()

    def register_server(self, server: ServerConfig) -> None:
        self._manager.register(server)

    async def invoke_tool(self, server: str, tool: str, arguments: dict[str, Any] | None = None, *, context: dict[str, Any] | None = None, trace_id: str | None = None, request_id: str | None = None) -> ToolResponse:
        config = self._manager.registry.get(server)
        logger.info('mcp_client_selection_json\n%s', pretty_json(redact_sensitive({
            'server_selected': config.name,
            'base_url': config.base_url,
            'endpoint_path': config.endpoint_path,
            'tool': tool,
            'request_payload': arguments or {},
            'context': context or {},
            'timeout_policy': self._retry.max_attempts,
        })))
        transport = self._factory.create_transport(config)
        async def operation() -> object:
            return await self._executor.execute(transport, tool, arguments, context=context, trace_id=trace_id, request_id=request_id)
        response = await retry_async(operation, self._retry)
        if not isinstance(response, ToolResponse):
            raise ResponseParseError('MCP executor returned an unexpected response type')
        logger.info('mcp_client_response_json\n%s', pretty_json(redact_sensitive(response.model_dump(by_alias=True))))
        return response

    async def list_tools(self, server: str) -> list[dict[str, Any]]:
        config = self._manager.registry.get(server)
        logger.info('mcp_client_tools_list_started server=%s endpoint_path=%s', config.name, config.endpoint_path)
        transport = self._factory.create_transport(config)

        async def operation() -> object:
            return await transport.list_tools()

        response = await retry_async(operation, self._retry)
        if not isinstance(response, list):
            raise ResponseParseError('MCP tools/list returned an unexpected response type')
        logger.info(
            'mcp_client_tools_list_completed server=%s tool_count=%s',
            config.name,
            len(response),
        )
        return response

    async def health_check(self, server: str) -> HealthStatus:
        config = self._manager.registry.get(server)
        return await self._health.check(ConnectionSession(config, self._manager.pool))

    async def close(self) -> None:
        await self._manager.shutdown()

    async def __aenter__(self) -> "MCPClient":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()
