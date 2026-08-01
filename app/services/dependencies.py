import asyncio
import logging

from fastapi import Depends

from app.config.settings import Settings, get_settings
from app.graph.graph import WorkflowManager
from app.guardrails import GuardrailEngine, GuardrailsLoader
from app.guardrails.models import GuardrailsConfig
from app.mcp_client import MCPClient
from app.mcp_client.client.manager import MCPClientManager
from app.mcp_client.connection.pool import ConnectionPool
from app.mcp_client.discovery.config_loader import ServerConfigLoader
from app.mcp_client.models.connection import ConnectionSettings
from app.mcp_client.utils.retry import RetryPolicy
from app.model_gateway.client import ModelGatewayClient
from app.planner.parser import PlannerOutputParser
from app.planner.planner import PlannerService
from app.planner.prompts import PlannerPromptProvider
from app.planner.registry_validator import PlannerRegistryValidator
from app.prompts.builder import PromptBuilder
from app.security.client import SecurityClassifierClient
from app.security.models import SecurityClassifierConfig
from app.security.service import SecurityClassificationService
from app.services.chat_service import ChatService
from app.tool_executor.service import ToolExecutorService
from app.tool_registry.repository import ToolRegistryRepository
from app.tool_registry.service import ToolRegistryService
from app.web_search.client import WebSearchClient

logger = logging.getLogger(__name__)

_tool_registry_repository = ToolRegistryRepository()
_tool_registry_service = ToolRegistryService(_tool_registry_repository)
_mcp_client: MCPClient | None = None
_guardrail_engine: GuardrailEngine | None = None
_guardrails_config: GuardrailsConfig | None = None
_security_classifier_service: SecurityClassificationService | None = None
_model_gateway_client: ModelGatewayClient | None = None
_planner_service: PlannerService | None = None
_tool_executor_service: ToolExecutorService | None = None
_workflow_manager: WorkflowManager | None = None


def get_prompt_builder() -> PromptBuilder:
    return PromptBuilder()


def get_planner_prompt_provider() -> PlannerPromptProvider:
    return PlannerPromptProvider()


def get_planner_output_parser() -> PlannerOutputParser:
    return PlannerOutputParser()


def get_model_gateway_client(settings: Settings = Depends(get_settings)) -> ModelGatewayClient:
    global _model_gateway_client
    if _model_gateway_client is None:
        _model_gateway_client = ModelGatewayClient(settings)
    return _model_gateway_client


def get_web_search_client(settings: Settings = Depends(get_settings)) -> WebSearchClient:
    return WebSearchClient(settings)


def get_tool_registry_service() -> ToolRegistryService:
    return _tool_registry_service


def get_guardrails_config(settings: Settings = Depends(get_settings)) -> GuardrailsConfig:
    global _guardrails_config
    if _guardrails_config is None:
        _guardrails_config = GuardrailsLoader().load(settings.guardrails_config_path)
    return _guardrails_config


def get_guardrail_engine(config: GuardrailsConfig = Depends(get_guardrails_config)) -> GuardrailEngine:
    global _guardrail_engine
    if _guardrail_engine is None:
        _guardrail_engine = GuardrailEngine(config)
    return _guardrail_engine


def get_security_classifier_service(
    settings: Settings = Depends(get_settings),
    guardrails_config: GuardrailsConfig = Depends(get_guardrails_config),
) -> SecurityClassificationService:
    global _security_classifier_service
    if _security_classifier_service is None:
        classifier_config = SecurityClassifierConfig.model_validate(guardrails_config.security_classifier.model_dump())
        _security_classifier_service = SecurityClassificationService(
            SecurityClassifierClient(settings),
            classifier_config,
        )
    return _security_classifier_service


def get_planner_service(
    prompt_builder: PromptBuilder = Depends(get_prompt_builder),
    prompt_provider: PlannerPromptProvider = Depends(get_planner_prompt_provider),
    output_parser: PlannerOutputParser = Depends(get_planner_output_parser),
    model_gateway_client: ModelGatewayClient = Depends(get_model_gateway_client),
    tool_registry_service: ToolRegistryService = Depends(get_tool_registry_service),
) -> PlannerService:
    global _planner_service
    if _planner_service is None:
        _planner_service = PlannerService(
            prompt_builder,
            prompt_provider,
            output_parser,
            model_gateway_client,
            PlannerRegistryValidator(tool_registry_service),
        )
    return _planner_service


def build_mcp_client(settings: Settings) -> MCPClient:
    connection_settings = ConnectionSettings(
        connect_timeout_seconds=settings.mcp_connect_timeout_seconds,
        read_timeout_seconds=settings.mcp_read_timeout_seconds,
        write_timeout_seconds=settings.mcp_write_timeout_seconds,
        pool_timeout_seconds=settings.mcp_pool_timeout_seconds,
        verify_tls=settings.mcp_verify_tls,
    )
    manager = MCPClientManager(pool=ConnectionPool(connection_settings))
    retry_policy = RetryPolicy(max_attempts=settings.mcp_max_retries + 1)
    return MCPClient(manager=manager, retry_policy=retry_policy)


def get_mcp_client(settings: Settings = Depends(get_settings)) -> MCPClient:
    global _mcp_client
    if _mcp_client is None:
        client = build_mcp_client(settings)
        loader = ServerConfigLoader()
        for server in loader.load(settings.mcp_servers_config_path):
            client.register_server(server)
        _mcp_client = client
    return _mcp_client


def get_tool_executor_service(
    tool_registry_service: ToolRegistryService = Depends(get_tool_registry_service),
    mcp_client: MCPClient = Depends(get_mcp_client),
) -> ToolExecutorService:
    global _tool_executor_service
    if _tool_executor_service is None:
        _tool_executor_service = ToolExecutorService(tool_registry_service, mcp_client)
    return _tool_executor_service


def get_workflow_manager(
    planner_service: PlannerService = Depends(get_planner_service),
    model_gateway_client: ModelGatewayClient = Depends(get_model_gateway_client),
    tool_executor_service: ToolExecutorService = Depends(get_tool_executor_service),
    web_search_client: WebSearchClient = Depends(get_web_search_client),
) -> WorkflowManager:
    global _workflow_manager
    if _workflow_manager is None:
        _workflow_manager = WorkflowManager(
            planner_service,
            model_gateway_client,
            tool_executor_service,
            get_tool_registry_service(),
            web_search_client,
        )
    return _workflow_manager


def get_chat_service(
    workflow_manager: WorkflowManager = Depends(get_workflow_manager),
    guardrail_engine: GuardrailEngine = Depends(get_guardrail_engine),
    security_classifier_service: SecurityClassificationService = Depends(get_security_classifier_service),
) -> ChatService:
    return ChatService(workflow_manager, guardrail_engine, security_classifier_service)


async def close_runtime_clients() -> None:
    global _mcp_client, _model_gateway_client, _planner_service, _tool_executor_service, _workflow_manager
    global _guardrail_engine, _guardrails_config, _security_classifier_service
    close_operations = []
    if _mcp_client is not None:
        close_operations.append(_mcp_client.close())
    if _model_gateway_client is not None:
        close_operations.append(_model_gateway_client.close())
    if close_operations:
        results = await asyncio.gather(*close_operations, return_exceptions=True)
        failures = [result for result in results if isinstance(result, Exception)]
        if failures:
            logger.warning('runtime_client_shutdown_completed_with_errors error_count=%s', len(failures))
            for failure in failures:
                logger.warning('runtime_client_shutdown_error error_type=%s error=%s', type(failure).__name__, failure)
    _mcp_client = None
    _model_gateway_client = None
    _planner_service = None
    _tool_executor_service = None
    _workflow_manager = None
    _guardrail_engine = None
    _guardrails_config = None
    _security_classifier_service = None
