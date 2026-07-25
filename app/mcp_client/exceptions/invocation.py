from .connection import MCPClientError

class InvocationError(MCPClientError):
    pass

class AuthenticationError(InvocationError):
    pass

class ResponseParseError(InvocationError):
    pass

class RequestValidationError(InvocationError):
    pass
