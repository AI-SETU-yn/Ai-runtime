from __future__ import annotations

import asyncio
import logging
from time import perf_counter

import httpx

from app.config.settings import Settings
from app.model_gateway.config import ModelGatewayRequestConfig
from app.security.exceptions import (
    SecurityClassificationResponseError,
    SecurityClassifierTimeoutError,
    SecurityClassifierTransportError,
)
from app.security.models import SecurityDecision
from app.security.prompts import SECURITY_CLASSIFIER_SYSTEM_PROMPT
from app.utils.json_logging import pretty_json

logger = logging.getLogger(__name__)


class SecurityClassifierClient:
    def __init__(self, settings: Settings) -> None:
        self._config = ModelGatewayRequestConfig(
            url=settings.model_gateway_url,
            path=settings.model_gateway_security_path,
            timeout_seconds=settings.model_gateway_security_timeout_seconds,
            connect_timeout_seconds=settings.model_gateway_security_connect_timeout_seconds,
            read_timeout_seconds=settings.model_gateway_security_read_timeout_seconds,
            max_retries=settings.model_gateway_max_retries,
        )

    async def classify(self, message: str) -> SecurityDecision:
        payload = {
            'message': message,
            'system_prompt': SECURITY_CLASSIFIER_SYSTEM_PROMPT,
        }
        body = await self._post(payload)
        try:
            return SecurityDecision.model_validate(body)
        except Exception as exc:
            raise SecurityClassificationResponseError('Security classifier returned an invalid response.') from exc

    async def _post(self, payload: dict[str, object]) -> dict[str, object]:
        last_error: Exception | None = None
        url = f"{self._config.url.rstrip('/')}{self._config.path}"
        timeout = httpx.Timeout(
            timeout=self._config.timeout_seconds,
            connect=self._config.connect_timeout_seconds,
            read=self._config.read_timeout_seconds,
        )
        logger.info('security_classifier_request_json\n%s', pretty_json({'url': url, 'payload': payload}))
        for attempt in range(self._config.max_retries + 1):
            started = perf_counter()
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(url, json=payload)
                response.raise_for_status()
                body = response.json()
                latency_ms = round((perf_counter() - started) * 1000, 2)
                logger.info('security_classifier_response latency_ms=%s attempt=%s', latency_ms, attempt + 1)
                logger.info('security_classifier_response_json\n%s', pretty_json(body))
                return body
            except httpx.TimeoutException as exc:
                last_error = exc
                logger.warning('security_classifier_timeout attempt=%s url=%s', attempt + 1, url)
            except (httpx.HTTPError, ValueError) as exc:
                last_error = exc
                logger.warning('security_classifier_failure attempt=%s url=%s error=%s', attempt + 1, url, str(exc))
            if attempt < self._config.max_retries:
                await asyncio.sleep(0.2 * (attempt + 1))
        if isinstance(last_error, httpx.TimeoutException):
            raise SecurityClassifierTimeoutError('Security classifier request timed out.') from last_error
        raise SecurityClassifierTransportError('Security classifier request failed.') from last_error