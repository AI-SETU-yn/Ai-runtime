import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.runtime import router as runtime_router
from app.config.settings import get_settings
from app.exceptions.errors import AppException
from app.exceptions.handlers import (
    app_exception_handler,
    request_validation_exception_handler,
    unhandled_exception_handler,
)
from app.graph.graph import WorkflowManager
from app.middleware.request_context import RequestContextMiddleware
from app.model_gateway.client import ModelGatewayClient
from app.planner.parser import PlannerOutputParser
from app.planner.planner import PlannerService
from app.planner.prompts import PlannerPromptProvider
from app.prompts.builder import PromptBuilder
from app.services.dependencies import get_mcp_client, get_tool_registry_service
from app.tool_executor.service import ToolExecutorService
from app.utils.logging import configure_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings)
    if settings.bypass_auth:
        logger.warning('Running in LOCAL DEVELOPMENT MODE - JWT authentication is bypassed.')

    logger.info(
        'model_gateway_configuration_loaded',
        extra={
            'model_gateway_url': settings.model_gateway_url,
            'planner_path': settings.model_gateway_planner_path,
            'generate_path': settings.model_gateway_chat_path,
            'adapter': settings.model_gateway_adapter,
        },
    )

    tool_registry_service = get_tool_registry_service()
    tool_registry_service.initialize(settings.tool_registry_path)
    app.state.tool_registry_service = tool_registry_service
    logger.info(
        'tool_registry_loaded',
        extra={
            'registry_path': str(settings.tool_registry_path),
            'registry_count': len(tool_registry_service.repository.registries),
            'ambiguous_key_count': len(tool_registry_service.repository.ambiguous_keys),
        },
    )

    mcp_client = get_mcp_client(settings)
    app.state.mcp_client = mcp_client
    logger.info(
        'mcp_client_initialized',
        extra={
            'mcp_servers_config_path': str(settings.mcp_servers_config_path),
        },
    )

    model_gateway_client = ModelGatewayClient(settings)
    planner_service = PlannerService(
        PromptBuilder(),
        PlannerPromptProvider(),
        PlannerOutputParser(),
        model_gateway_client,
    )
    tool_executor_service = ToolExecutorService(tool_registry_service, mcp_client)
    workflow_manager = WorkflowManager(planner_service, model_gateway_client, tool_executor_service)
    app.state.workflow_manager = workflow_manager
    logger.info('workflow_initialized')

    app.state.is_ready = settings.ready_on_startup
    try:
        yield
    finally:
        app.state.is_ready = False
        await mcp_client.close()



def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=['*'],
        allow_headers=['*'],
    )
    app.add_middleware(RequestContextMiddleware)

    app.add_exception_handler(AppException, app_exception_handler)
    app.add_exception_handler(RequestValidationError, request_validation_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)

    app.include_router(runtime_router)
    return app


app = create_app()
