import logging
from logging.config import dictConfig

from app.config.settings import Settings


LOG_FORMAT = (
    '%(asctime)s | %(levelname)-8s | %(logger_name)s | '
    'req=%(request_id)s | corr=%(correlation_id)s | conv=%(conversation_id)s | %(message)s'
)


class RequestContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.logger_name = record.name.replace('app.', '')
        try:
            from app.conversation.context import ConversationContextStore

            context = ConversationContextStore.get()
            record.request_id = context.request_id or '-'
            record.correlation_id = context.correlation_id or '-'
            record.conversation_id = context.conversation_id or '-'
        except Exception:
            record.request_id = '-'
            record.correlation_id = '-'
            record.conversation_id = '-'
        return True



def configure_logging(settings: Settings) -> None:
    dictConfig(
        {
            'version': 1,
            'disable_existing_loggers': False,
            'filters': {'request_context': {'()': RequestContextFilter}},
            'formatters': {
                'default': {
                    'format': LOG_FORMAT,
                    'datefmt': '%Y-%m-%d %H:%M:%S',
                }
            },
            'handlers': {
                'console': {
                    'class': 'logging.StreamHandler',
                    'formatter': 'default',
                    'filters': ['request_context'],
                }
            },
            'root': {'level': settings.log_level.upper(), 'handlers': ['console']},
        }
    )
