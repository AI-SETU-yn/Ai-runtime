"""A lightweight typed session bound to one server."""
import httpx
import asyncio
from ..models.server import ServerConfig
from .pool import ConnectionPool


class ConnectionSession:
    def __init__(self, server: ServerConfig, pool: ConnectionPool) -> None:
        self.server, self._pool = server, pool

    async def client(self) -> httpx.AsyncClient:
        return await self._pool.acquire(self.server)

    def session_id(self) -> str | None:
        return self._pool.session_id(self.server.name)

    def set_session_id(self, session_id: str) -> None:
        self._pool.set_session_id(self.server.name, session_id)

    def clear_session_id(self) -> None:
        self._pool.clear_session_id(self.server.name)

    def is_initialized(self) -> bool:
        return self._pool.is_initialized(self.server.name)

    def mark_initialized(self, protocol_version: str) -> None:
        self._pool.mark_initialized(self.server.name, protocol_version)

    def protocol_version(self) -> str | None:
        return self._pool.protocol_version(self.server.name)

    def clear_session_state(self) -> None:
        self._pool.clear_session_state(self.server.name)

    def initialization_lock(self) -> asyncio.Lock:
        return self._pool.initialization_lock(self.server.name)

    async def close(self) -> None:
        await self._pool.release(self.server.name)
