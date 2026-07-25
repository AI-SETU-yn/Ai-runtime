import uuid

from app.conversation.context import ConversationContext


class ConversationSessionFactory:
    @staticmethod
    def from_headers(headers) -> ConversationContext:
        request_id = headers.get('X-Request-Id', str(uuid.uuid4()))
        correlation_id = headers.get('X-Correlation-Id', request_id)
        conversation_id = headers.get('X-Conversation-Id', str(uuid.uuid4()))
        return ConversationContext(
            request_id=request_id,
            correlation_id=correlation_id,
            conversation_id=conversation_id,
        )
