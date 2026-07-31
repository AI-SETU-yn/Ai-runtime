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


@dataclass(frozen=True)
class ConversationContextTokens:
    request_id: contextvars.Token
    correlation_id: contextvars.Token
    conversation_id: contextvars.Token


class ConversationContextStore:
    @staticmethod
    def set(context: ConversationContext) -> ConversationContextTokens:
        return ConversationContextTokens(
            request_id=request_id_var.set(context.request_id),
            correlation_id=correlation_id_var.set(context.correlation_id),
            conversation_id=conversation_id_var.set(context.conversation_id),
        )

    @staticmethod
    def reset(tokens: ConversationContextTokens) -> None:
        request_id_var.reset(tokens.request_id)
        correlation_id_var.reset(tokens.correlation_id)
        conversation_id_var.reset(tokens.conversation_id)

    @staticmethod
    def get() -> ConversationContext:
        return ConversationContext(
            request_id=request_id_var.get() or '',
            correlation_id=correlation_id_var.get() or '',
            conversation_id=conversation_id_var.get() or '',
        )
