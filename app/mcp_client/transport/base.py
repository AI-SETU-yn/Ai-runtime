"""Extensible transport interface."""
from abc import ABC, abstractmethod
from ..models.request import ToolRequest
from ..models.response import ToolResponse


class BaseTransport(ABC):
    @abstractmethod
    async def invoke(self, request: ToolRequest) -> ToolResponse:
        """Send one request and return a typed response."""
