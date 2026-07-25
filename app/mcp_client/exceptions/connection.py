class MCPClientError(Exception):
    """Base error for all enterprise MCP client failures."""

class MCPConnectionError(MCPClientError):
    pass

class MCPTimeoutError(MCPConnectionError):
    pass

class ServerUnavailableError(MCPConnectionError):
    pass
