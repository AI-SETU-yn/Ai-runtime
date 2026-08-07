from __future__ import annotations

import asyncio
import logging
from time import perf_counter

import httpx

from app.config.settings import Settings
from app.conversation.context import ConversationContextStore
from app.guardrails.audit import metrics
from app.model_gateway.config import ModelGatewayRequestConfig
from app.security.exceptions import (
    SecurityClassificationResponseError,
    SecurityClassifierTimeoutError,
    SecurityClassifierTransportError,
)
from app.security.models import SecurityDecision
from app.security.prompts import SECURITY_CLASSIFIER_SYSTEM_PROMPT
from app.utils.tokens import estimate_tokens

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
        prompt_length = len(message)
        token_estimate = estimate_tokens(message)
        started = perf_counter()
        try:
            body = await self._post(payload)
        except Exception:
            latency_ms = round((perf_counter() - started) * 1000, 2)
            metrics.increment('classifier_failed')
            self._log_summary(prompt_length, token_estimate, 'ERROR', latency_ms)
            raise
        latency_ms = round((perf_counter() - started) * 1000, 2)
        try:
            decision = SecurityDecision.model_validate(body)
        except Exception as exc:
            metrics.increment('classifier_failed')
            self._log_summary(prompt_length, token_estimate, 'INVALID_RESPONSE', latency_ms)
            raise SecurityClassificationResponseError('Security classifier returned an invalid response.') from exc

        # classifier_result is a bounded enum + confidence pair, never the
        # model's free-text `reason`, which can otherwise echo back fragments
        # of the (possibly sensitive) classified message into logs.
        classifier_result = f'{"SAFE" if decision.safe else "UNSAFE"}:{decision.category.value}:{decision.confidence:.2f}'
        self._log_summary(prompt_length, token_estimate, classifier_result, latency_ms)
        return decision

    @staticmethod
    def _log_summary(prompt_length: int, token_estimate: int, classifier_result: str, latency_ms: float) -> None:
        # Never log the raw message, the classifier's free-text reason, or
        # the full request/response payload here -- only bounded identifiers
        # and metrics. See app.utils.redaction for the shared helper used by
        # call sites that do need to log structured payloads.
        context = ConversationContextStore.get()
        logger.info(
            'security_classifier_summary request_id=%s conversation_id=%s prompt_length=%s '
            'token_estimate=%s classifier_result=%s latency_ms=%s',
            context.request_id or '-',
            context.conversation_id or '-',
            prompt_length,
            token_estimate,
            classifier_result,
            latency_ms,
        )

    async def _post(self, payload: dict[str, object]) -> dict[str, object]:
        last_error: Exception | None = None
        url = f"{self._config.url.rstrip('/')}{self._config.path}"
        timeout = httpx.Timeout(
            timeout=self._config.timeout_seconds,
            connect=self._config.connect_timeout_seconds,
            read=self._config.read_timeout_seconds,
        )
        for attempt in range(self._config.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(url, json=payload)
                response.raise_for_status()
                return response.json()
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