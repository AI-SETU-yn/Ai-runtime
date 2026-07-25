"""In-memory, typed server registry."""
from ..exceptions import DuplicateServerError, ServerNotFoundError
from ..models.server import ServerConfig


class ServerRegistry:
    def __init__(self) -> None:
        self._servers: dict[str, ServerConfig] = {}

    def register(self, server: ServerConfig) -> None:
        if server.name in self._servers:
            raise DuplicateServerError(f"Server name already registered: {server.name}")
        if any(item.base_url == server.base_url for item in self._servers.values()):
            raise DuplicateServerError(f"Server URL already registered: {server.base_url}")
        self._servers[server.name] = server

    def remove(self, name: str) -> ServerConfig:
        try:
            return self._servers.pop(name)
        except KeyError as exc:
            raise ServerNotFoundError(f"Unknown MCP server: {name}") from exc

    def get(self, name: str) -> ServerConfig:
        try:
            return self._servers[name]
        except KeyError as exc:
            raise ServerNotFoundError(f"Unknown MCP server: {name}") from exc

    def list(self) -> tuple[ServerConfig, ...]:
        return tuple(self._servers.values())
