"""Owns registered server configurations and the shared connection pool."""
from ..connection.pool import ConnectionPool
from ..discovery.server_registry import ServerRegistry
from ..models.server import ServerConfig


class MCPClientManager:
    def __init__(self, registry: ServerRegistry | None = None, pool: ConnectionPool | None = None) -> None:
        self.registry, self.pool = registry or ServerRegistry(), pool or ConnectionPool()

    def register(self, server: ServerConfig) -> None:
        self.registry.register(server)

    async def shutdown(self) -> None:
        await self.pool.close()
