"""Provider-agnostic, bounded web-search client for current-information requests."""
from __future__ import annotations

import asyncio
import html
import logging
import re
from time import perf_counter
from typing import Any

import httpx

from app.config.settings import Settings
from app.exceptions.errors import WebSearchError, WebSearchTimeoutError

logger = logging.getLogger(__name__)
_MARKUP_PATTERN = re.compile(r'<[^>]+>')
_WHITESPACE_PATTERN = re.compile(r'\s+')
_RESULT_KEYS = ('results', 'items', 'organic_results', 'organicResults', 'data')
_TEXT_KEYS = ('snippet', 'content', 'text', 'description', 'body', 'summary')


class WebSearchClient:
    def __init__(self, settings: Settings) -> None:
        self._enabled = settings.web_search_enabled
        self._url = settings.web_search_url
        self._api_key = settings.web_search_api_key
        self._max_results = settings.web_search_max_results
        self._max_result_characters = settings.web_search_max_result_characters
        self._max_context_characters = settings.web_search_max_context_characters
        self._max_retries = settings.web_search_max_retries
        self._timeout = httpx.Timeout(
            timeout=settings.web_search_timeout_seconds,
            connect=settings.web_search_connect_timeout_seconds,
            read=settings.web_search_read_timeout_seconds,
        )

    @property
    def is_configured(self) -> bool:
        return bool(self._enabled and self._url)

    async def search(self, query: str) -> dict[str, Any]:
        """Return a normalized, prompt-safe result set or raise a web-search error."""
        if not self.is_configured:
            raise WebSearchError('Web search is not configured.', code='WEB_SEARCH_NOT_CONFIGURED')

        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            started = perf_counter()
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.post(
                        self._url,
                        json={'query': query, 'max_results': self._max_results},
                        headers=self._headers(),
                    )
                response.raise_for_status()
                try:
                    body = response.json()
                except ValueError as exc:
                    raise WebSearchError('Web search returned invalid JSON.') from exc
                normalized = self._normalize_response(body)
                logger.info(
                    'web_search_completed attempt=%s latency_ms=%s result_count=%s',
                    attempt + 1,
                    round((perf_counter() - started) * 1000, 2),
                    len(normalized['results']),
                )
                return normalized
            except WebSearchError as exc:
                last_error = exc
            except httpx.TimeoutException as exc:
                last_error = exc
                logger.warning('web_search_timeout attempt=%s', attempt + 1)
            except httpx.HTTPError as exc:
                last_error = exc
                logger.warning('web_search_http_failure attempt=%s error_type=%s', attempt + 1, type(exc).__name__)
            if attempt < self._max_retries:
                await asyncio.sleep(0.2 * (attempt + 1))

        if isinstance(last_error, httpx.TimeoutException):
            raise WebSearchTimeoutError('Web search request timed out.') from last_error
        if isinstance(last_error, WebSearchError):
            raise last_error
        raise WebSearchError('Web search request failed.') from last_error

    def _headers(self) -> dict[str, str]:
        return {'Authorization': f'Bearer {self._api_key}'} if self._api_key else {}

    def _normalize_response(self, body: Any) -> dict[str, list[dict[str, Any]]]:
        if body in (None, '', [], {}):
            raise WebSearchError('Web search returned no usable results.', code='WEB_SEARCH_EMPTY_RESULT')
        if isinstance(body, dict) and any(key in body for key in ('error', 'errors')) and not any(
            key in body for key in _RESULT_KEYS
        ):
            raise WebSearchError('Web search provider returned an error payload.', code='WEB_SEARCH_PROVIDER_ERROR')

        candidates: Any = body
        is_you_com_response = False
        if isinstance(body, dict):
            provider_results = body.get('results')
            if isinstance(provider_results, dict):
                # You.com Search API: {"results": {"web": [...]}, "metadata": {...}}
                candidates = provider_results.get('web')
                is_you_com_response = True
            else:
                candidates = next((body[key] for key in _RESULT_KEYS if key in body), None)
        if not isinstance(candidates, list):
            raise WebSearchError('Web search response has no result list.', code='WEB_SEARCH_MALFORMED_RESULT')

        results: list[dict[str, Any]] = []
        remaining = self._max_context_characters
        for candidate in candidates:
            if len(results) >= self._max_results or remaining <= 0 or not isinstance(candidate, dict):
                continue
            title = self._clean_text(candidate.get('title') or candidate.get('name') or '')
            if is_you_com_response:
                result, used = self._normalize_you_com_result(candidate, title, remaining)
                if result is None:
                    continue
                results.append(result)
                remaining -= used
                continue
            text = self._clean_text(next((candidate[key] for key in _TEXT_KEYS if candidate.get(key)), ''))
            if not title and not text:
                continue
            title = title[: self._max_result_characters]
            text = text[: min(self._max_result_characters, max(0, remaining - len(title)))]
            result = {'title': title, 'content': text}
            url = candidate.get('url') or candidate.get('link')
            if isinstance(url, str) and url.strip():
                result['url'] = url.strip()
            results.append(result)
            remaining -= len(result['title']) + len(result['content'])
        if not results:
            raise WebSearchError('Web search returned no meaningful results.', code='WEB_SEARCH_UNUSABLE_RESULT')
        return {'results': results}

    def _normalize_you_com_result(
        self,
        candidate: dict[str, Any],
        title: str,
        remaining: int,
    ) -> tuple[dict[str, Any] | None, int]:
        """Normalize You.com's nested web result while preserving prompt bounds."""
        title = title[: self._max_result_characters]
        description = self._clean_text(candidate.get('description'))
        snippets = self._clean_snippets(candidate.get('snippets'))
        if not title and not description and not snippets:
            return None, 0

        budget = max(0, remaining - len(title))
        description = description[: min(self._max_result_characters, budget)]
        budget -= len(description)
        bounded_snippets: list[str] = []
        for snippet in snippets:
            if budget <= 0:
                break
            bounded = snippet[: min(self._max_result_characters, budget)]
            if bounded:
                bounded_snippets.append(bounded)
                budget -= len(bounded)

        result: dict[str, Any] = {'title': title}
        url = candidate.get('url')
        if isinstance(url, str) and url.strip():
            result['url'] = url.strip()
        if description:
            result['description'] = description
        if bounded_snippets:
            result['snippets'] = bounded_snippets
        return result, remaining - budget

    @classmethod
    def _clean_snippets(cls, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [cleaned for item in value if (cleaned := cls._clean_text(item))]

    @staticmethod
    def _clean_text(value: Any) -> str:
        if not isinstance(value, str):
            return ''
        return _WHITESPACE_PATTERN.sub(' ', html.unescape(_MARKUP_PATTERN.sub(' ', value))).strip()
