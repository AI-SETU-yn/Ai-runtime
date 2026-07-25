"""Application lifecycle adapter for FastAPI lifespan handlers."""
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from .manager import MCPClientManager


@asynccontextmanager
async def managed_mcp_client(manager: MCPClientManager) -> AsyncIterator[MCPClientManager]:
    try:
        yield manager
    finally:
        await manager.shutdown()
