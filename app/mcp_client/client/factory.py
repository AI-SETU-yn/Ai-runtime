"""Dependency-injection factory for V1 Streamable HTTP transports."""
from ..connection.pool import ConnectionPool
from ..connection.session import ConnectionSession
from ..models.server import ServerConfig
from ..transport.base import BaseTransport
from ..transport.streamable_http import StreamableHTTPTransport


class MCPClientFactory:
    def __init__(self, pool: ConnectionPool) -> None:
        self._pool = pool

    def create_transport(self, server: ServerConfig) -> BaseTransport:
        return StreamableHTTPTransport(ConnectionSession(server, self._pool))
