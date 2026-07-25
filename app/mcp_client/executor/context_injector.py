"""Adds vetted AI Runtime context to MCP request metadata."""
from typing import Any
from ..exceptions import RequestValidationError
from ..models.request import ToolRequest


ALLOWED_CONTEXT_FIELDS = frozenset(
    {
        'request_id',
        'correlation_id',
        'tenant_id',
        'trace_id',
        'user_id',
        'subject',
        'org_id',
        'organization_id',
        'branch_id',
        'app_id',
        'application_ids',
        'roles',
        'permissions',
        'jwt',
        'locale',
        'session_id',
    }
)


def _is_json_value(value: Any) -> bool:
    if value is None or isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, list):
        return all(_is_json_value(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _is_json_value(item) for key, item in value.items())
    return False


class ContextInjector:
    def inject(self, request: ToolRequest, context: dict[str, Any] | None) -> ToolRequest:
        if not context:
            return request
        if not isinstance(context, dict):
            raise RequestValidationError('runtime context must be a mapping')
        unknown = sorted(set(context) - ALLOWED_CONTEXT_FIELDS)
        if unknown:
            raise RequestValidationError(f"Unsupported runtime context fields: {', '.join(unknown)}")
        invalid = sorted(key for key, value in context.items() if not _is_json_value(value))
        if invalid:
            raise RequestValidationError(f"Runtime context fields must be JSON serializable: {', '.join(invalid)}")
        meta = {**request.params.meta, **context}
        return request.model_copy(update={'params': request.params.model_copy(update={'meta': meta})})

