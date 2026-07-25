import contextvars
from dataclasses import dataclass

request_id_var = contextvars.ContextVar('request_id', default='')
correlation_id_var = contextvars.ContextVar('correlation_id', default='')
conversation_id_var = contextvars.ContextVar('conversation_id', default='')


@dataclass(frozen=True)
class ConversationContext:
    request_id: str
    correlation_id: str
    conversation_id: str


class ConversationContextStore:
    @staticmethod
    def set(context: ConversationContext) -> None:
        request_id_var.set(context.request_id)
        correlation_id_var.set(context.correlation_id)
        conversation_id_var.set(context.conversation_id)

    @staticmethod
    def get() -> ConversationContext:
        return ConversationContext(
            request_id=request_id_var.get() or '',
            correlation_id=correlation_id_var.get() or '',
            conversation_id=conversation_id_var.get() or '',
        )
