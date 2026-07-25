"""Custom exceptions for the tool registry module."""

from app.exceptions.errors import AppException


class ToolNotFoundException(AppException):
    """Raised when no tool matches an exact lookup request."""

    status_code = 404
    code = 'TOOL_NOT_FOUND'


class DuplicateToolException(AppException):
    """Raised when multiple tools resolve to the same lookup key."""

    status_code = 409
    code = 'DUPLICATE_TOOL'


class InvalidToolRegistryException(AppException):
    """Raised when a registry file or aggregate registry is invalid."""

    status_code = 500
    code = 'INVALID_TOOL_REGISTRY'
