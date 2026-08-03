from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, model_validator
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
    local_app_id: str = 'local-app'
    local_tenant_id: str = 'local-tenant'
    local_locale: str = 'en-US'
    local_session_id: str = 'local-session'
    local_application_ids: list[str] = Field(default_factory=list)
    local_roles: list[str] = Field(default_factory=lambda: ['SUPER_ADMIN'])
    local_permissions: list[str] = Field(default_factory=lambda: ['*'])

    jwt_secret: str = Field(default='', validation_alias=AliasChoices('JWT_SECRET', 'AI_RUNTIME_JWT_SECRET'))
    jwt_algorithm: str = Field(default='HS256', validation_alias=AliasChoices('JWT_ALGORITHM', 'AI_RUNTIME_JWT_ALGORITHM'))
    jwt_issuer: str | None = None
    jwt_audience: str | None = None

    model_gateway_url: str = 'http://localhost:9000'
    model_gateway_chat_path: str = '/generate'
    model_gateway_planner_path: str = '/planner'
    model_gateway_security_path: str = '/security/classify'
    model_gateway_timeout_seconds: float = 30.0
    model_gateway_connect_timeout_seconds: float = 5.0
    model_gateway_read_timeout_seconds: float = 30.0
    model_gateway_send_planner_prompt: bool = True
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
    model_gateway_security_timeout_seconds: float = 30.0
    model_gateway_security_connect_timeout_seconds: float = 5.0
    model_gateway_security_read_timeout_seconds: float = 30.0
    model_gateway_max_retries: int = 2

    mcp_servers_config_path: Path = Path('app/mcp_client/config/servers.yaml')
    mcp_connect_timeout_seconds: float = 5.0
    mcp_read_timeout_seconds: float = 30.0
    mcp_write_timeout_seconds: float = 10.0
    mcp_pool_timeout_seconds: float = 5.0
    mcp_max_retries: int = 2
    mcp_verify_tls: bool = True
    mcp_validate_contracts_on_startup: bool = False

    allowed_origins: list[str] = Field(default_factory=lambda: ['*'])
    log_level: str = 'INFO'
    ready_on_startup: bool = True
    tool_registry_path: Path = Path('tool-registry')
    guardrails_config_path: Path = Path('app/config/guardrails.yaml')
    current_info_config_path: Path = Path('app/config/current_info.yaml')

    # Disabled unless both this flag and a provider URL are configured.
    web_search_enabled: bool = False
    web_search_url: str = ''
    web_search_api_key: str = Field(
        default='',
        validation_alias=AliasChoices('WEB_SEARCH_API_KEY', 'AI_RUNTIME_WEB_SEARCH_API_KEY'),
    )
    web_search_max_results: int = 5
    web_search_timeout_seconds: float = 10.0
    web_search_connect_timeout_seconds: float = 3.0
    web_search_read_timeout_seconds: float = 10.0
    web_search_max_retries: int = 1
    web_search_max_result_characters: int = 1_000
    web_search_max_context_characters: int = 4_000

    @model_validator(mode='after')
    def validate_production_security(self) -> 'Settings':
        if self.environment != 'prod':
            return self
        if self.bypass_auth:
            raise ValueError('AI_RUNTIME_BYPASS_AUTH must be false in prod.')
        if not self.jwt_secret:
            raise ValueError('AI_RUNTIME_JWT_SECRET must be configured in prod.')
        if '*' in self.allowed_origins:
            raise ValueError('AI_RUNTIME_ALLOWED_ORIGINS must not contain "*" in prod.')
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
