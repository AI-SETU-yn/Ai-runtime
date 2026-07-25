from .connection import MCPClientError

class ServerRegistryError(MCPClientError):
    pass

class DuplicateServerError(ServerRegistryError):
    pass

class ServerNotFoundError(ServerRegistryError):
    pass
