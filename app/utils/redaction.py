from __future__ import annotations

import re
from typing import Any

SENSITIVE_KEYS = frozenset(
    {
        'access_token',
        'api_key',
        'authorization',
        'jwt',
        'password',
        'private_key',
        'refresh_token',
        'secret',
        'token',
    }
)

_BEARER_PATTERN = re.compile(r'(?i)\bbearer\s+[a-z0-9\-._~+/]+=*')
_SECRET_ASSIGNMENT_PATTERN = re.compile(
    r'(?i)\b(api[_ -]?key|access[_ -]?token|refresh[_ -]?token|jwt|password|secret)\s*[:=]\s*\S+'
)


def redact_sensitive(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: '<redacted>' if _is_sensitive_key(key) and item else redact_sensitive(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_sensitive(item) for item in value]
    if isinstance(value, str):
        redacted = _BEARER_PATTERN.sub('Bearer <redacted>', value)
        return _SECRET_ASSIGNMENT_PATTERN.sub(lambda match: f'{match.group(1)}=<redacted>', redacted)
    return value


def _is_sensitive_key(key: str) -> bool:
    normalized = re.sub(r'[^a-zA-Z0-9]+', '_', key).strip('_').lower()
    if normalized in SENSITIVE_KEYS:
        return True
    return normalized.endswith('_token') or normalized.endswith('_secret')
