import logging
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from app.conversation.context import ConversationContextStore
from app.conversation.session import ConversationSessionFactory

logger = logging.getLogger(__name__)


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        context = ConversationSessionFactory.from_headers(request.headers)

        request.state.request_id = context.request_id
        request.state.correlation_id = context.correlation_id
        request.state.conversation_id = context.conversation_id

        ConversationContextStore.set(context)

        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

        response.headers['X-Request-Id'] = context.request_id
        response.headers['X-Correlation-Id'] = context.correlation_id
        response.headers['X-Conversation-Id'] = context.conversation_id
        response.headers['X-Response-Time-Ms'] = str(elapsed_ms)

        logger.info(
            'request_completed',
            extra={
                'path': request.url.path,
                'method': request.method,
                'status_code': response.status_code,
                'request_id': context.request_id,
                'correlation_id': context.correlation_id,
                'conversation_id': context.conversation_id,
                'execution_time_ms': elapsed_ms,
            },
        )
        return response
