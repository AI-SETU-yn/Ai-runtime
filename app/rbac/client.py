from __future__ import annotations

import asyncio
import logging
import re
from time import perf_counter
from typing import Any

import httpx

from app.config.settings import Settings
from app.rbac.models import AuthorizationDecision, AuthorizationRequest, AuthorizationStep
from app.utils.json_logging import pretty_json
from app.utils.redaction import redact_sensitive

logger = logging.getLogger(__name__)


class RBACClientError(RuntimeError):
    pass


class RBACClient:
    def __init__(self, settings: Settings) -> None:
        self._base_url = settings.rbac_url.rstrip('/')
        self._authorization_path = settings.rbac_authorization_path.strip()
        self._menu_items_path = settings.rbac_menu_items_path
        self._timeout = httpx.Timeout(
            timeout=settings.rbac_timeout_seconds,
            connect=settings.rbac_connect_timeout_seconds,
            read=settings.rbac_read_timeout_seconds,
        )
        self._max_retries = settings.rbac_max_retries
        self._fail_open = settings.rbac_fail_open
        self._client: httpx.AsyncClient | None = None

    async def authorize(self, request: AuthorizationRequest, jwt: str | None = None) -> AuthorizationDecision:
        if self._authorization_path:
            return await self._authorize_via_decision_endpoint(request, jwt)
        return await self._authorize_via_menu_items(request, jwt)

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()

    async def _authorize_via_decision_endpoint(
        self,
        request: AuthorizationRequest,
        jwt: str | None,
    ) -> AuthorizationDecision:
        body = await self._post(
            self._authorization_path,
            request.model_dump(exclude_none=True),
            jwt,
            operation='authorize',
        )
        return self._decision_from_response(body, request.steps[0] if request.steps else None)

    async def _authorize_via_menu_items(
        self,
        request: AuthorizationRequest,
        jwt: str | None,
    ) -> AuthorizationDecision:
        if not request.user_id:
            return AuthorizationDecision(allowed=False, reason='Missing user id for RBAC authorization.')
        if not request.branch_id:
            return AuthorizationDecision(allowed=False, reason='Missing branch id for RBAC authorization.')

        body = await self._post(
            self._menu_items_path,
            {'userId': request.user_id},
            jwt,
            operation='menu_items',
            headers={'X-BRANCH-Id': request.branch_id},
        )
        menu_items = self._extract_menu_items(body)
        for step in request.steps:
            if not self._is_step_allowed_by_menu(step, menu_items):
                return AuthorizationDecision(
                    allowed=False,
                    step_id=step.step_id,
                    tool=step.tool,
                    capability=step.capability,
                    domain=step.domain,
                    service=step.service,
                    entity=step.entity,
                    operation=step.operation,
                    reason='RBAC menu access does not include the requested tool capability.',
                    source='rbac_menu_items',
                    raw_response=self._compact_menu_response(body),
                )
        return AuthorizationDecision(
            allowed=True,
            reason='RBAC menu access includes all requested tool capabilities.',
            source='rbac_menu_items',
            raw_response=self._compact_menu_response(body),
        )

    async def _post(
        self,
        path: str,
        payload: dict[str, Any],
        jwt: str | None,
        *,
        operation: str,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        last_error: Exception | None = None
        url = f'{self._base_url}{path if path.startswith("/") else "/" + path}'
        request_headers = dict(headers or {})
        authorization = self._authorization_header(jwt)
        if authorization:
            request_headers['Authorization'] = authorization
        logger.info('rbac_%s_request_json\n%s', operation, pretty_json(redact_sensitive({
            'url': url,
            'payload': payload,
            'headers': request_headers,
        })))
        for attempt in range(self._max_retries + 1):
            started = perf_counter()
            try:
                client = await self._get_client()
                response = await client.post(url, json=payload, headers=request_headers, timeout=self._timeout)
                response.raise_for_status()
                body = response.json()
                latency_ms = round((perf_counter() - started) * 1000, 2)
                logger.info('rbac_%s_response latency_ms=%s attempt=%s', operation, latency_ms, attempt + 1)
                logger.info('rbac_%s_response_json\n%s', operation, pretty_json(redact_sensitive(body)))
                return body
            except (httpx.HTTPError, ValueError) as exc:
                last_error = exc
                latency_ms = round((perf_counter() - started) * 1000, 2)
                logger.warning(
                    'rbac_%s_failure attempt=%s url=%s elapsed_ms=%s error=%s',
                    operation,
                    attempt + 1,
                    url,
                    latency_ms,
                    str(exc),
                )
            if attempt < self._max_retries:
                await asyncio.sleep(0.2 * (attempt + 1))
        if self._fail_open:
            logger.error('rbac_%s_failed_open', operation)
            return {'allowed': True, 'reason': 'RBAC unavailable; fail-open is enabled.'}
        raise RBACClientError('RBAC authorization request failed.') from last_error

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient()
        return self._client

    @classmethod
    def _decision_from_response(
        cls,
        body: dict[str, Any],
        first_step: AuthorizationStep | None,
    ) -> AuthorizationDecision:
        data = body.get('data') if isinstance(body.get('data'), dict) else body
        allowed = data.get('allowed', data.get('authorized', data.get('success')))
        if not isinstance(allowed, bool):
            allowed = not bool(body.get('error'))
        denied_step = data.get('deniedStep') if isinstance(data.get('deniedStep'), dict) else {}
        step_id = data.get('stepId') or data.get('step_id') or denied_step.get('stepId') or denied_step.get('step_id')
        tool = data.get('tool') or data.get('toolName') or denied_step.get('tool') or denied_step.get('toolName')
        reason = str(data.get('reason') or data.get('message') or body.get('message') or '')
        return AuthorizationDecision(
            allowed=allowed,
            step_id=step_id or (first_step.step_id if first_step and not allowed else None),
            tool=tool or (first_step.tool if first_step and not allowed else None),
            capability=data.get('capability') or (first_step.capability if first_step and not allowed else None),
            domain=data.get('domain') or (first_step.domain if first_step and not allowed else None),
            service=data.get('service') or (first_step.service if first_step and not allowed else None),
            entity=data.get('entity') or (first_step.entity if first_step and not allowed else None),
            operation=data.get('operation') or (first_step.operation if first_step and not allowed else None),
            reason=reason or ('Authorized by RBAC service.' if allowed else 'Denied by RBAC service.'),
            raw_response=body,
        )

    @classmethod
    def _extract_menu_items(cls, body: dict[str, Any]) -> list[dict[str, Any]]:
        data = body.get('data') if isinstance(body.get('data'), dict) else body
        modules = data.get('modules') if isinstance(data, dict) else None
        if not isinstance(modules, list):
            return []
        items: list[dict[str, Any]] = []
        for module in modules:
            if not isinstance(module, dict):
                continue
            items.append(cls._menu_record(module, 'module'))
            for sub_module in module.get('subModules') or []:
                if not isinstance(sub_module, dict):
                    continue
                items.append(cls._menu_record(sub_module, 'sub_module'))
                for menu_item in sub_module.get('menuItems') or []:
                    if isinstance(menu_item, dict):
                        items.append(cls._menu_record(menu_item, 'menu_item'))
        return items

    @classmethod
    def _menu_record(cls, item: dict[str, Any], item_type: str) -> dict[str, Any]:
        return {
            'type': item_type,
            'name': item.get('menuName') or item.get('subModuleName') or item.get('moduleName'),
            'display_name': item.get('displayName'),
            'url': item.get('url'),
        }

    @classmethod
    def _is_step_allowed_by_menu(cls, step: AuthorizationStep, menu_items: list[dict[str, Any]]) -> bool:
        if not menu_items:
            return False
        targets = cls._step_targets(step)
        for item in menu_items:
            values = [item.get('name'), item.get('display_name'), item.get('url')]
            normalized_values = {cls._normalize(value) for value in values if value}
            if not normalized_values:
                continue
            if any(target in normalized_values for target in targets):
                return True
            if any(
                cls._is_structural_match(target, value)
                for target in targets
                for value in normalized_values
            ):
                return True
        return False

    @classmethod
    def _step_targets(cls, step: AuthorizationStep) -> set[str]:
        values = {
            step.tool,
            step.capability,
            f'{step.service}.{step.entity}.{step.operation}',
            f'{step.domain}.{step.service}.{step.entity}.{step.operation}',
            f'{step.service}.{step.operation}',
            f'{step.entity}.{step.operation}',
        }
        targets = {cls._normalize(value) for value in values if value}
        return {target for target in targets if target}

    @staticmethod
    def _is_structural_match(target: str, value: str) -> bool:
        if not target or not value:
            return False
        target_parts = target.split('.')
        value_parts = value.split('.')
        if len(target_parts) < 2 or len(value_parts) < 2:
            return False
        return target in value or value in target

    @staticmethod
    def _normalize(value: Any) -> str:
        text = str(value).casefold().strip()
        text = re.sub(r'[^a-z0-9]+', '.', text)
        return re.sub(r'\.+', '.', text).strip('.')

    @staticmethod
    def _authorization_header(jwt_token: str | None) -> str | None:
        if not jwt_token:
            return None
        if jwt_token.lower().startswith('bearer '):
            return jwt_token
        return f'Bearer {jwt_token}'

    @staticmethod
    def _compact_menu_response(body: dict[str, Any]) -> dict[str, Any]:
        data = body.get('data') if isinstance(body.get('data'), dict) else body
        if not isinstance(data, dict):
            return {}
        return {
            'userId': data.get('userId'),
            'branchId': data.get('branchId'),
            'roleId': data.get('roleId'),
            'roleName': data.get('roleName'),
            'moduleCount': len(data.get('modules') or []),
        }
