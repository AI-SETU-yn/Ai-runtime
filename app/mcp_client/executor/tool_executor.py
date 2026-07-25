"""Coordinates request creation, context injection, and transport invocation."""
from typing import Any
from ..models.response import ToolResponse
from ..transport.base import BaseTransport
from .context_injector import ContextInjector
from .request_builder import RequestBuilder


class ToolExecutor:
    def __init__(self, builder: RequestBuilder | None = None, injector: ContextInjector | None = None) -> None:
        self._builder, self._injector = builder or RequestBuilder(), injector or ContextInjector()

    async def execute(self, transport: BaseTransport, tool: str, arguments: dict[str, Any] | None = None, *, context: dict[str, Any] | None = None, trace_id: str | None = None, request_id: str | None = None) -> ToolResponse:
        return await transport.invoke(self._injector.inject(self._builder.build(tool, arguments, trace_id=trace_id, request_id=request_id), context))
