from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        env_prefix='AI_RUNTIME_',
        case_sensitive=False,
        extra='ignore',
    )

    app_name: str = 'yn-ai-setu-ai-runtime'
    environment: Literal['local', 'dev', 'stage', 'prod', 'test'] = 'local'
    host: str = '0.0.0.0'
    port: int = 8000
    debug: bool = False
    bypass_auth: bool = False
    local_subject: str = 'developer'
    local_user_id: str = 'developer'
    local_organization_id: str = 'yntec'
    local_branch_id: str = 'main'
    local_app_id: str = 'vidhya'
    local_tenant_id: str = 'yn'
    local_locale: str = 'en-IN'
    local_session_id: str = 'local-session'
    local_application_ids: list[str] = Field(default_factory=lambda: ['hrms', 'vidhya'])
    local_roles: list[str] = Field(default_factory=lambda: ['SUPER_ADMIN'])
    local_permissions: list[str] = Field(default_factory=lambda: ['*'])

    jwt_secret: str = Field(default='', validation_alias=AliasChoices('JWT_SECRET', 'AI_RUNTIME_JWT_SECRET'))
    jwt_algorithm: str = Field(default='HS256', validation_alias=AliasChoices('JWT_ALGORITHM', 'AI_RUNTIME_JWT_ALGORITHM'))

    model_gateway_url: str = 'http://localhost:9000'
    model_gateway_chat_path: str = '/generate'
    model_gateway_planner_path: str = '/planner'
    model_gateway_adapter: str = ''
    model_gateway_timeout_seconds: float = 30.0
    model_gateway_connect_timeout_seconds: float = 5.0
    model_gateway_read_timeout_seconds: float = 30.0
    model_gateway_send_planner_prompt: bool = False
    model_gateway_planner_timeout_seconds: float = Field(
        default=30.0,
        validation_alias=AliasChoices(
            'AI_RUNTIME_MODEL_GATEWAY_PLANNER_TIMEOUT_SECONDS',
            'AI_RUNTIME_MODEL_GATEWAY_TIMEOUT_SECONDS',
        ),
    )
    model_gateway_planner_connect_timeout_seconds: float = Field(
        default=5.0,
        validation_alias=AliasChoices(
            'AI_RUNTIME_MODEL_GATEWAY_PLANNER_CONNECT_TIMEOUT_SECONDS',
            'AI_RUNTIME_MODEL_GATEWAY_CONNECT_TIMEOUT_SECONDS',
        ),
    )
    model_gateway_planner_read_timeout_seconds: float = Field(
        default=30.0,
        validation_alias=AliasChoices(
            'AI_RUNTIME_MODEL_GATEWAY_PLANNER_READ_TIMEOUT_SECONDS',
            'AI_RUNTIME_MODEL_GATEWAY_READ_TIMEOUT_SECONDS',
        ),
    )
    model_gateway_generate_timeout_seconds: float = 90.0
    model_gateway_generate_connect_timeout_seconds: float = 5.0
    model_gateway_generate_read_timeout_seconds: float = 90.0
    model_gateway_max_retries: int = 2

    mcp_servers_config_path: Path = Path('app/mcp_client/config/servers.yaml')
    mcp_connect_timeout_seconds: float = 5.0
    mcp_read_timeout_seconds: float = 30.0
    mcp_write_timeout_seconds: float = 10.0
    mcp_pool_timeout_seconds: float = 5.0
    mcp_max_retries: int = 2
    mcp_verify_tls: bool = True

    allowed_origins: list[str] = Field(default_factory=lambda: ['*'])
    log_level: str = 'INFO'
    ready_on_startup: bool = True
    tool_registry_path: Path = Path('tool-registry')
    guardrails_config_path: Path = Path('app/config/guardrails.yaml')


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

