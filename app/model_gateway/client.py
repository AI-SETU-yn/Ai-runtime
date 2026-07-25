import asyncio
import logging
from time import perf_counter
from typing import Any

import httpx

from app.config.settings import Settings
from app.model_gateway.config import ModelGatewayRequestConfig
from app.model_gateway.exceptions import ModelGatewayError, ModelGatewayTimeoutError
from app.utils.json_logging import pretty_json

logger = logging.getLogger(__name__)


class ModelGatewayClient:
    def __init__(self, settings: Settings) -> None:
        self._config = ModelGatewayRequestConfig(
            url=settings.model_gateway_url,
            path=settings.model_gateway_chat_path,
            timeout_seconds=settings.model_gateway_timeout_seconds,
            connect_timeout_seconds=settings.model_gateway_connect_timeout_seconds,
            read_timeout_seconds=settings.model_gateway_read_timeout_seconds,
            max_retries=settings.model_gateway_max_retries,
        )
        self._planner_path = settings.model_gateway_planner_path
        self._adapter = settings.model_gateway_adapter
        self._timeout = httpx.Timeout(
            timeout=self._config.timeout_seconds,
            connect=self._config.connect_timeout_seconds,
            read=self._config.read_timeout_seconds,
        )

    async def generate(self, prompt: str, *, metadata: dict[str, Any] | None = None) -> str:
        payload = {
            'adapter': self._adapter,
            'prompt': prompt,
        }
        body = await self._post(self._config.path, payload, operation='generate')
        return str(body.get('response') or body.get('answer') or body.get('content') or '')

    async def plan(self, query: str) -> dict[str, Any]:
        payload = {
            'adapter': self._adapter,
            'query': query,
        }
        return await self._post(self._planner_path, payload, operation='planner')

    async def _post(self, path: str, payload: dict[str, Any], *, operation: str) -> dict[str, Any]:
        if not self._adapter:
            raise ModelGatewayError('Model gateway adapter is not configured.')
        last_error: Exception | None = None
        logger.info('model_gateway_%s_request_json\n%s', operation, pretty_json({'url': f"{self._config.url.rstrip('/')}{path}", 'payload': payload, 'timeout_seconds': self._config.timeout_seconds}))
        for attempt in range(self._config.max_retries + 1):
            started = perf_counter()
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.post(
                        f"{self._config.url.rstrip('/')}{path}",
                        json=payload,
                    )
                response.raise_for_status()
                body = response.json()
                latency_ms = round((perf_counter() - started) * 1000, 2)
                logger.info('model_gateway_%s_response latency_ms=%s attempt=%s', operation, latency_ms, attempt + 1)
                logger.info('model_gateway_%s_response_json\n%s', operation, pretty_json(body))
                return body
            except httpx.TimeoutException as exc:
                last_error = exc
                logger.warning('model_gateway_timeout operation=%s attempt=%s', operation, attempt + 1)
            except (httpx.HTTPError, ValueError) as exc:
                last_error = exc
                logger.warning('model_gateway_failure operation=%s attempt=%s error=%s', operation, attempt + 1, str(exc))
            if attempt < self._config.max_retries:
                await asyncio.sleep(0.2 * (attempt + 1))
        if isinstance(last_error, httpx.TimeoutException):
            raise ModelGatewayTimeoutError('Model gateway request timed out.') from last_error
        raise ModelGatewayError('Model gateway request failed.') from last_error
