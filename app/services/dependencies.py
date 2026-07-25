from fastapi import Depends

from app.config.settings import Settings, get_settings
from app.graph.graph import WorkflowManager
from app.mcp_client import MCPClient
from app.mcp_client.client.manager import MCPClientManager
from app.mcp_client.connection.pool import ConnectionPool
from app.mcp_client.discovery.config_loader import ServerConfigLoader
from app.mcp_client.models.connection import ConnectionSettings
from app.model_gateway.client import ModelGatewayClient
from app.planner.parser import PlannerOutputParser
from app.planner.planner import PlannerService
from app.planner.prompts import PlannerPromptProvider
from app.prompts.builder import PromptBuilder
from app.services.chat_service import ChatService
from app.tool_executor.service import ToolExecutorService
from app.tool_registry.repository import ToolRegistryRepository
from app.tool_registry.service import ToolRegistryService

_tool_registry_repository = ToolRegistryRepository()
_tool_registry_service = ToolRegistryService(_tool_registry_repository)
_mcp_client: MCPClient | None = None


def get_prompt_builder() -> PromptBuilder:
    return PromptBuilder()


def get_planner_prompt_provider() -> PlannerPromptProvider:
    return PlannerPromptProvider()


def get_planner_output_parser() -> PlannerOutputParser:
    return PlannerOutputParser()


def get_model_gateway_client(settings: Settings = Depends(get_settings)) -> ModelGatewayClient:
    return ModelGatewayClient(settings)


def get_planner_service(
    prompt_builder: PromptBuilder = Depends(get_prompt_builder),
    prompt_provider: PlannerPromptProvider = Depends(get_planner_prompt_provider),
    output_parser: PlannerOutputParser = Depends(get_planner_output_parser),
    model_gateway_client: ModelGatewayClient = Depends(get_model_gateway_client),
) -> PlannerService:
    return PlannerService(prompt_builder, prompt_provider, output_parser, model_gateway_client)


def build_mcp_client(settings: Settings) -> MCPClient:
    connection_settings = ConnectionSettings(
        connect_timeout_seconds=settings.mcp_connect_timeout_seconds,
        read_timeout_seconds=settings.mcp_read_timeout_seconds,
        write_timeout_seconds=settings.mcp_write_timeout_seconds,
        pool_timeout_seconds=settings.mcp_pool_timeout_seconds,
        verify_tls=settings.mcp_verify_tls,
    )
    manager = MCPClientManager(pool=ConnectionPool(connection_settings))
    return MCPClient(manager=manager)


def get_mcp_client(settings: Settings = Depends(get_settings)) -> MCPClient:
    global _mcp_client
    if _mcp_client is None:
        client = build_mcp_client(settings)
        loader = ServerConfigLoader()
        for server in loader.load(settings.mcp_servers_config_path):
            client.register_server(server)
        _mcp_client = client
    return _mcp_client


def get_tool_registry_service() -> ToolRegistryService:
    return _tool_registry_service


def get_tool_executor_service(
    tool_registry_service: ToolRegistryService = Depends(get_tool_registry_service),
    mcp_client: MCPClient = Depends(get_mcp_client),
) -> ToolExecutorService:
    return ToolExecutorService(tool_registry_service, mcp_client)


def get_workflow_manager(
    planner_service: PlannerService = Depends(get_planner_service),
    model_gateway_client: ModelGatewayClient = Depends(get_model_gateway_client),
    tool_executor_service: ToolExecutorService = Depends(get_tool_executor_service),
) -> WorkflowManager:
    return WorkflowManager(planner_service, model_gateway_client, tool_executor_service)


def get_chat_service(workflow_manager: WorkflowManager = Depends(get_workflow_manager)) -> ChatService:
    return ChatService(workflow_manager)
