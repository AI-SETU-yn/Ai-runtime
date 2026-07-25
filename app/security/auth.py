import logging
import uuid
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

GATEWAY_USER_HEADER = 'X-USER-Id'
GATEWAY_ORG_HEADER = 'X-ORG-Id'
GATEWAY_BRANCH_HEADER = 'X-BRANCH-Id'
GATEWAY_APP_HEADER = 'X-APP-Id'


class JwtService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def validate_token(self, token: str) -> dict[str, Any]:
        if not self._settings.jwt_secret:
            raise UnauthorizedError('JWT validation secret is not configured.')
        try:
            return jwt.decode(token, self._settings.jwt_secret, algorithms=[self._settings.jwt_algorithm])
        except jwt.ExpiredSignatureError as exc:
            raise UnauthorizedError('JWT token has expired.') from exc
        except jwt.InvalidSignatureError as exc:
            raise UnauthorizedError('JWT signature is invalid.') from exc
        except jwt.InvalidTokenError as exc:
            raise UnauthorizedError('Invalid JWT token.') from exc


class JwtClaimsMapper:
    """Map gateway-compatible JWT claims into one request-scoped runtime context."""

    REQUIRED_CLAIMS = ('sub', 'organizationId', 'branchId')

    def from_request(self, claims: dict[str, Any], token: str, request: Request) -> RuntimeContext:
        missing = [claim for claim in self.REQUIRED_CLAIMS if not self._claim_value(claims, claim)]
        if missing:
            raise UnauthorizedError(f"JWT is missing required claim(s): {', '.join(missing)}")

        subject = str(claims['sub'])
        user_id = str(claims.get('userId') or subject)
        organization_id = str(claims['organizationId'])
        branch_id = str(claims['branchId'])
        app_id = self._optional_string(claims.get('appId'))

        self._validate_gateway_header(request, GATEWAY_USER_HEADER, user_id, subject)
        self._validate_gateway_header(request, GATEWAY_ORG_HEADER, organization_id)
        self._validate_gateway_header(request, GATEWAY_BRANCH_HEADER, branch_id)
        if app_id is not None:
            self._validate_gateway_header(request, GATEWAY_APP_HEADER, app_id)

        application_ids = self._as_string_list(claims.get('applicationIds') or claims.get('appIds') or [])
        if app_id and app_id not in application_ids:
            application_ids = [app_id, *application_ids]

        return RuntimeContext(
            subject=subject,
            user_id=user_id,
            organization_id=organization_id,
            branch_id=branch_id,
            app_id=app_id,
            tenant_id=self._optional_string(claims.get('tenantId')),
            locale=self._optional_string(claims.get('locale')),
            application_ids=application_ids,
            roles=self._as_string_list(claims.get('roles') or []),
            permissions=self._as_string_list(claims.get('permissions') or []),
            session_id=self._optional_string(claims.get('sessionId')),
            request_id=getattr(request.state, 'request_id', None),
            correlation_id=getattr(request.state, 'correlation_id', None),
            trace_id=request.headers.get('X-Trace-Id') or str(uuid.uuid4()),
            jwt=token,
            raw_claims=claims,
        )

    @staticmethod
    def _claim_value(claims: dict[str, Any], claim: str) -> Any:
        value = claims.get(claim)
        return value if value is not None and str(value).strip() else None

    @staticmethod
    def _optional_string(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _as_string_list(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value] if value else []
        if isinstance(value, (list, tuple, set)):
            return [str(item) for item in value if item is not None and str(item)]
        return [str(value)]

    @staticmethod
    def _validate_gateway_header(request: Request, header_name: str, expected: str, *accepted: str) -> None:
        actual = request.headers.get(header_name)
        if actual is None:
            return
        allowed = {expected, *accepted}
        if actual not in allowed:
            logger.warning(
                'gateway_header_claim_mismatch header=%s request_id=%s correlation_id=%s',
                header_name,
                getattr(request.state, 'request_id', None),
                getattr(request.state, 'correlation_id', None),
            )
            raise UnauthorizedError(f'{header_name} does not match JWT claims.')


class RuntimeContextFactory:
    _mapper = JwtClaimsMapper()

    @staticmethod
    def from_claims(claims: dict[str, Any], token: str | None = None) -> RuntimeContext:
        application_ids = claims.get('applicationIds') or claims.get('appIds') or claims.get('app_ids') or []
        roles = claims.get('roles') or []
        permissions = claims.get('permissions') or []
        app_id = claims.get('appId')
        normalized_application_ids = [str(item) for item in application_ids]
        if app_id and str(app_id) not in normalized_application_ids:
            normalized_application_ids = [str(app_id), *normalized_application_ids]
        return RuntimeContext(
            subject=str(claims.get('sub') or ''),
            user_id=str(claims.get('userId') or claims.get('sub') or ''),
            organization_id=claims.get('organizationId') or claims.get('orgId'),
            branch_id=claims.get('branchId'),
            app_id=app_id,
            tenant_id=claims.get('tenantId'),
            locale=claims.get('locale'),
            application_ids=normalized_application_ids,
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
            app_id='vidhya',
            tenant_id='yn',
            locale='en-IN',
            application_ids=['hrms', 'vidhya'],
            roles=['SUPER_ADMIN'],
            permissions=['*'],
            session_id='local-session',
            jwt='local-development-token',
            raw_claims={'mode': 'local-development'},
        )

    @classmethod
    def from_request(cls, claims: dict[str, Any], token: str, request: Request) -> RuntimeContext:
        return cls._mapper.from_request(claims, token, request)


def get_jwt_service(settings: Settings = Depends(get_settings)) -> JwtService:
    return JwtService(settings)


async def get_runtime_context(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
    jwt_service: JwtService = Depends(get_jwt_service),
    settings: Settings = Depends(get_settings),
) -> RuntimeContext:
    existing_context = getattr(request.state, 'runtime_context', None)
    if isinstance(existing_context, RuntimeContext):
        return existing_context

    if settings.bypass_auth:
        runtime_context = RuntimeContextFactory.for_local_development().model_copy(
            update={
                'request_id': getattr(request.state, 'request_id', None),
                'correlation_id': getattr(request.state, 'correlation_id', None),
                'trace_id': request.headers.get('X-Trace-Id') or str(uuid.uuid4()),
            }
        )
        request.state.runtime_context = runtime_context
        request.state.conversation_context = ConversationContextStore.get()
        return runtime_context

    if credentials is None or credentials.scheme.lower() != 'bearer':
        raise UnauthorizedError('Missing bearer token.')

    claims = jwt_service.validate_token(credentials.credentials)
    runtime_context = RuntimeContextFactory.from_request(claims, credentials.credentials, request)

    request.state.runtime_context = runtime_context
    request.state.conversation_context = ConversationContextStore.get()
    logger.info(
        'runtime_context_created user_id=%s organization_id=%s branch_id=%s app_id=%s request_id=%s correlation_id=%s trace_id=%s',
        runtime_context.user_id,
        runtime_context.organization_id,
        runtime_context.branch_id,
        runtime_context.app_id,
        runtime_context.request_id,
        runtime_context.correlation_id,
        runtime_context.trace_id,
    )
    return runtime_context


