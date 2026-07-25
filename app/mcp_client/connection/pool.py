"""One reusable async HTTP client per MCP server."""
import asyncio
import httpx
from ..exceptions import MCPConnectionError
from ..models.connection import ConnectionSettings
from ..models.server import ServerConfig


class ConnectionPool:
    def __init__(self, settings: ConnectionSettings | None = None, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._settings = settings or ConnectionSettings()
        self._transport = transport
        self._clients: dict[str, httpx.AsyncClient] = {}
        self._session_ids: dict[str, str] = {}
        self._initialized: set[str] = set()
        self._protocol_versions: dict[str, str] = {}
        self._initialization_locks: dict[str, asyncio.Lock] = {}
        self._lock = asyncio.Lock()

    async def acquire(self, server: ServerConfig) -> httpx.AsyncClient:
        async with self._lock:
            client = self._clients.get(server.name)
            if client and not client.is_closed:
                return client
            s = self._settings
            client = httpx.AsyncClient(
                base_url=server.base_url,
                transport=self._transport,
                verify=s.verify_tls,
                timeout=httpx.Timeout(s.read_timeout_seconds, connect=s.connect_timeout_seconds, write=s.write_timeout_seconds, pool=s.pool_timeout_seconds),
                limits=httpx.Limits(max_connections=s.max_connections, max_keepalive_connections=s.max_keepalive_connections, keepalive_expiry=s.keepalive_expiry_seconds),
            )
            self._clients[server.name] = client
            return client

    async def release(self, server_name: str) -> None:
        client = self._clients.pop(server_name, None)
        self.clear_session_state(server_name)
        if client:
            await client.aclose()

    async def close(self) -> None:
        clients = [client for client in self._clients.values() if not client.is_closed]
        self._clients.clear()
        self.clear_all_session_state()
        results = await asyncio.gather(*(client.aclose() for client in clients), return_exceptions=True)
        failures = [result for result in results if isinstance(result, Exception)]
        if failures:
            raise MCPConnectionError(f"Failed to close {len(failures)} MCP HTTP client(s)")

    def session_id(self, server_name: str) -> str | None:
        return self._session_ids.get(server_name)

    def set_session_id(self, server_name: str, session_id: str) -> None:
        self._session_ids[server_name] = session_id

    def clear_session_id(self, server_name: str) -> None:
        self._session_ids.pop(server_name, None)

    def is_initialized(self, server_name: str) -> bool:
        return server_name in self._initialized

    def mark_initialized(self, server_name: str, protocol_version: str) -> None:
        self._initialized.add(server_name)
        self._protocol_versions[server_name] = protocol_version

    def protocol_version(self, server_name: str) -> str | None:
        return self._protocol_versions.get(server_name)

    def clear_session_state(self, server_name: str) -> None:
        self._session_ids.pop(server_name, None)
        self._initialized.discard(server_name)
        self._protocol_versions.pop(server_name, None)

    def clear_all_session_state(self) -> None:
        self._session_ids.clear()
        self._initialized.clear()
        self._protocol_versions.clear()
        self._initialization_locks.clear()

    def initialization_lock(self, server_name: str) -> asyncio.Lock:
        lock = self._initialization_locks.get(server_name)
        if lock is None:
            lock = asyncio.Lock()
            self._initialization_locks[server_name] = lock
        return lock
