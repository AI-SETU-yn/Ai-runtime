"""Creates official MCP JSON-RPC tool call requests."""
from typing import Any
from pydantic import ValidationError
from ..exceptions import RequestValidationError
from ..models.request import ToolRequest


class RequestBuilder:
    def build(self, tool: str, arguments: dict[str, Any] | None = None, *, trace_id: str | None = None, request_id: str | None = None) -> ToolRequest:
        if not isinstance(tool, str) or not tool.strip():
            raise RequestValidationError("tool must be a non-empty string")
        if arguments is not None and not isinstance(arguments, dict):
            raise RequestValidationError("MCP tool arguments must be a JSON object")
        meta: dict[str, Any] = {}
        if trace_id is not None:
            if not isinstance(trace_id, str) or not trace_id:
                raise RequestValidationError("trace_id must be a non-empty string")
            meta["trace_id"] = trace_id
        values: dict[str, Any] = {"params": {"name": tool, "arguments": arguments or {}}}
        if meta:
            values["params"]["_meta"] = meta
        if request_id is not None:
            if not isinstance(request_id, str) or not request_id:
                raise RequestValidationError("request_id must be a non-empty string")
            values["id"] = request_id
        try:
            return ToolRequest(**values)
        except ValidationError as exc:
            raise RequestValidationError(str(exc)) from exc
