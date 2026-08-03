"""Extensible transport interface."""
from abc import ABC, abstractmethod
from typing import Any

from ..models.request import ToolRequest
from ..models.response import ToolResponse


class BaseTransport(ABC):
    @abstractmethod
    async def invoke(self, request: ToolRequest) -> ToolResponse:
        """Send one request and return a typed response."""

    @abstractmethod
    async def list_tools(self) -> list[dict[str, Any]]:
        """Return the server's MCP tools/list response."""
