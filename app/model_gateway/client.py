import asyncio
import logging
from time import perf_counter
from typing import Any

import httpx

from app.config.settings import Settings
from app.model_gateway.config import ModelGatewayRequestConfig
from app.model_gateway.exceptions import ModelGatewayError, ModelGatewayTimeoutError
from app.utils.json_logging import pretty_json
from app.utils.redaction import redact_sensitive

logger = logging.getLogger(__name__)


class ModelGatewayClient:
    def __init__(self, settings: Settings) -> None:
        self._generate_config = ModelGatewayRequestConfig(
            url=settings.model_gateway_url,
            path=settings.model_gateway_chat_path,
            timeout_seconds=getattr(settings, 'model_gateway_generate_timeout_seconds', 90.0),
            connect_timeout_seconds=getattr(settings, 'model_gateway_generate_connect_timeout_seconds', 5.0),
            read_timeout_seconds=getattr(settings, 'model_gateway_generate_read_timeout_seconds', 90.0),
            max_retries=settings.model_gateway_max_retries,
        )
        self._planner_config = ModelGatewayRequestConfig(
            url=settings.model_gateway_url,
            path=settings.model_gateway_planner_path,
            timeout_seconds=getattr(
                settings,
                'model_gateway_planner_timeout_seconds',
                settings.model_gateway_timeout_seconds,
            ),
            connect_timeout_seconds=getattr(
                settings,
                'model_gateway_planner_connect_timeout_seconds',
                settings.model_gateway_connect_timeout_seconds,
            ),
            read_timeout_seconds=getattr(
                settings,
                'model_gateway_planner_read_timeout_seconds',
                settings.model_gateway_read_timeout_seconds,
            ),
            max_retries=settings.model_gateway_max_retries,
        )
        self._adapter = settings.model_gateway_adapter
        self._send_planner_prompt = getattr(settings, 'model_gateway_send_planner_prompt', False)
        self._client: httpx.AsyncClient | None = None

    async def generate(self, prompt: str, *, metadata: dict[str, Any] | None = None) -> str:
        payload = {
            'adapter': self._adapter,
            'prompt': prompt,
        }
        body = await self._post(self._generate_config, payload, operation='generate')
        return str(body.get('response') or body.get('answer') or body.get('content') or '')

    async def plan(self, query: str, *, prompt: str | None = None) -> dict[str, Any]:
        payload = {
            'adapter': self._adapter,
            'query': query,
        }
        if prompt and self._send_planner_prompt:
            payload['prompt'] = prompt
        return await self._post(self._planner_config, payload, operation='planner')

    async def _post(
        self,
        config: ModelGatewayRequestConfig,
        payload: dict[str, Any],
        *,
        operation: str,
    ) -> dict[str, Any]:
        if not self._adapter:
            raise ModelGatewayError('Model gateway adapter is not configured.')
        last_error: Exception | None = None
        url = f"{config.url.rstrip('/')}{config.path}"
        timeout = self._timeout_from_config(config)
        logger.info('model_gateway_%s_request_json\n%s', operation, pretty_json({
            'url': url,
            'payload': redact_sensitive(payload),
            'timeout_seconds': config.timeout_seconds,
            'connect_timeout_seconds': config.connect_timeout_seconds,
            'read_timeout_seconds': config.read_timeout_seconds,
        }))
        for attempt in range(config.max_retries + 1):
            started = perf_counter()
            try:
                client = await self._get_client()
                response = await client.post(
                    url,
                    json=payload,
                    timeout=timeout,
                )
                response.raise_for_status()
                body = response.json()
                latency_ms = round((perf_counter() - started) * 1000, 2)
                logger.info('model_gateway_%s_response latency_ms=%s attempt=%s', operation, latency_ms, attempt + 1)
                logger.info('model_gateway_%s_response_json\n%s', operation, pretty_json(redact_sensitive(body)))
                return body
            except httpx.TimeoutException as exc:
                last_error = exc
                elapsed_ms = round((perf_counter() - started) * 1000, 2)
                logger.warning(
                    'model_gateway_timeout operation=%s attempt=%s url=%s timeout_seconds=%s elapsed_ms=%s',
                    operation,
                    attempt + 1,
                    url,
                    config.timeout_seconds,
                    elapsed_ms,
                )
            except (httpx.HTTPError, ValueError) as exc:
                last_error = exc
                latency_ms = round((perf_counter() - started) * 1000, 2)
                logger.warning(
                    'model_gateway_failure operation=%s attempt=%s url=%s elapsed_ms=%s error=%s',
                    operation,
                    attempt + 1,
                    url,
                    latency_ms,
                    str(exc),
                )
            if attempt < config.max_retries:
                await asyncio.sleep(0.2 * (attempt + 1))
        if isinstance(last_error, httpx.TimeoutException):
            raise ModelGatewayTimeoutError('Model gateway request timed out.') from last_error
        raise ModelGatewayError('Model gateway request failed.') from last_error

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient()
        return self._client

    @staticmethod
    def _timeout_from_config(config: ModelGatewayRequestConfig) -> httpx.Timeout:
        return httpx.Timeout(
            timeout=config.timeout_seconds,
            connect=config.connect_timeout_seconds,
            read=config.read_timeout_seconds,
        )
