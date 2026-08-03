"""Optional current-information branch used at agent finalization."""
from __future__ import annotations

import logging

from app.exceptions.errors import WebSearchError
from app.graph.state import RuntimeState
from app.utils.current_info_detector import is_current_info_query
from app.web_search.client import WebSearchClient

logger = logging.getLogger(__name__)


class CurrentInfoRouterNode:
    def __init__(self, web_search_client: WebSearchClient | None = None) -> None:
        self._web_search_client = web_search_client

    async def __call__(self, state: RuntimeState) -> dict[str, object]:
        planner_output = state.planner_output
        # Planner/tool routing always wins; this node never replaces ERP data.
        if (
            planner_output is None
            or planner_output.requires_tool
            or state.tool_execution_result is not None
            or not is_current_info_query(state.user_question)
        ):
            return {}
        if self._web_search_client is None or not self._web_search_client.is_configured:
            logger.warning('current_info_web_search_unavailable')
            return self._failure_result(
                code='WEB_SEARCH_NOT_CONFIGURED',
                message='Current information search is not configured.',
            )
        try:
            evidence = await self._web_search_client.search(state.user_question)
        except WebSearchError as exc:
            logger.warning('current_info_web_search_failed error_code=%s', exc.code)
            return self._failure_result(code=exc.code, message=exc.message)
        return {
            'tool_execution_result': {
                'success': True,
                'status': 'ok',
                'data': {'source': 'web_search', **evidence},
            }
        }

    @staticmethod
    def _failure_result(*, code: str, message: str) -> dict[str, object]:
        return {
            'tool_execution_result': {
                'success': False,
                'status': 'error',
                'data': {
                    'source': 'web_search',
                    'error': {
                        'code': code,
                        'message': message,
                    },
                },
                'error': {
                    'code': code,
                    'message': message,
                },
            }
        }
