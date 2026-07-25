import logging
from typing import Any

import jwt
from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config.settings import Settings, get_settings
from app.conversation.context import ConversationContextStore
from app.exceptions.errors import UnauthorizedError
from app.models.runtime import RuntimeContext

security = HTTPBearer(auto_error=False)
logger = logging.getLogger(__name__)


class JwtService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def validate_token(self, token: str) -> dict[str, Any]:
        try:
            return jwt.decode(token, self._settings.jwt_secret, algorithms=[self._settings.jwt_algorithm])
        except jwt.InvalidTokenError as exc:
            raise UnauthorizedError('Invalid or expired token.') from exc


class RuntimeContextFactory:
    @staticmethod
    def from_claims(claims: dict[str, Any], token: str | None = None) -> RuntimeContext:
        application_ids = claims.get('applicationIds') or claims.get('appIds') or claims.get('app_ids') or []
        roles = claims.get('roles') or []
        permissions = claims.get('permissions') or []
        return RuntimeContext(
            subject=str(claims.get('sub') or ''),
            user_id=str(claims.get('userId') or claims.get('sub') or ''),
            organization_id=claims.get('organizationId') or claims.get('orgId'),
            branch_id=claims.get('branchId'),
            tenant_id=claims.get('tenantId'),
            locale=claims.get('locale'),
            application_ids=[str(item) for item in application_ids],
            roles=[str(item) for item in roles],
            permissions=[str(item) for item in permissions],
            session_id=claims.get('sessionId'),
            jwt=token,
            raw_claims=claims,
        )

    @staticmethod
    def for_local_development() -> RuntimeContext:
        return RuntimeContext(
            subject='developer',
            user_id='developer',
            organization_id='yntec',
            branch_id='main',
            tenant_id='yn',
            locale='en-IN',
            application_ids=['hrms', 'vidhya'],
            roles=['SUPER_ADMIN'],
            permissions=['*'],
            session_id='local-session',
            jwt='local-development-token',
            raw_claims={'mode': 'local-development'},
        )


def get_jwt_service(settings: Settings = Depends(get_settings)) -> JwtService:
    return JwtService(settings)


async def get_runtime_context(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
    jwt_service: JwtService = Depends(get_jwt_service),
    settings: Settings = Depends(get_settings),
) -> RuntimeContext:
    if settings.bypass_auth:
        runtime_context = RuntimeContextFactory.for_local_development()
        request.state.runtime_context = runtime_context
        request.state.conversation_context = ConversationContextStore.get()
        return runtime_context

    if credentials is None or credentials.scheme.lower() != 'bearer':
        raise UnauthorizedError('Missing bearer token.')

    claims = jwt_service.validate_token(credentials.credentials)
    runtime_context = RuntimeContextFactory.from_claims(claims, credentials.credentials)
    if not runtime_context.subject:
        raise UnauthorizedError('JWT subject is missing.')

    request.state.runtime_context = runtime_context
    request.state.conversation_context = ConversationContextStore.get()
    return runtime_context
