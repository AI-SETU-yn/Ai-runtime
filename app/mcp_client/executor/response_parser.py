"""Normalizes official MCP JSON-RPC responses into public response models."""
from typing import Any
from ..exceptions import ResponseParseError
from ..models.request import ToolRequest
from ..models.response import InvocationStatus, ToolError, ToolResponse


class ResponseParser:
    def parse(self, payload: Any, request: ToolRequest, elapsed: float, status_code: int = 200) -> ToolResponse:
        if not isinstance(payload, dict):
            raise ResponseParseError("Expected MCP JSON-RPC response object")
        if payload.get("jsonrpc") != "2.0":
            raise ResponseParseError("MCP JSON-RPC response must include jsonrpc='2.0'")
        if str(payload.get("id")) != request.id:
            raise ResponseParseError("MCP JSON-RPC response id does not match request id")
        if "error" in payload:
            error = self._parse_error(payload["error"])
            return ToolResponse(success=False, status=InvocationStatus.ERROR, tool_results=None, execution_time=elapsed, error=error, metadata=self._metadata(request, status_code))
        if "result" not in payload:
            raise ResponseParseError("MCP JSON-RPC response must include result or error")
        result = payload["result"]
        tool_error = self._tool_error(result)
        return ToolResponse(
            success=tool_error is None,
            status=InvocationStatus.SUCCESS if tool_error is None else InvocationStatus.ERROR,
            tool_results=result,
            execution_time=elapsed,
            error=tool_error,
            metadata=self._metadata(request, status_code),
        )

    def _parse_error(self, payload: Any) -> ToolError:
        if not isinstance(payload, dict):
            raise ResponseParseError("MCP JSON-RPC error must be an object")
        code = payload.get("code")
        message = payload.get("message")
        if not isinstance(message, str):
            raise ResponseParseError("MCP JSON-RPC error message must be a string")
        return ToolError(code=str(code), message=message, data=payload.get("data"))

    def _tool_error(self, result: Any) -> ToolError | None:
        if not isinstance(result, dict) or result.get("isError") is not True:
            return None
        return ToolError(code="TOOL_ERROR", message="MCP tool returned isError=true", data=result)

    def _metadata(self, request: ToolRequest, status_code: int) -> dict[str, Any]:
        metadata: dict[str, Any] = {"request_id": request.id, "status_code": status_code}
        if request.trace_id:
            metadata["trace_id"] = request.trace_id
        return metadata
