"""FastMCP Streamable HTTP transport for official MCP JSON-RPC messages."""
import json
import logging
import time
from typing import Any
from uuid import uuid4
import httpx
from ..connection.session import ConnectionSession
from ..exceptions import AuthenticationError, InvocationError, MCPConnectionError, MCPTimeoutError, ResponseParseError, ServerUnavailableError
from ..executor.response_parser import ResponseParser
from ..models.request import ToolRequest
from ..models.response import ToolResponse
from ...utils.json_logging import pretty_json
from .base import BaseTransport


MCP_PROTOCOL_VERSION = "2025-11-25"
MCP_CLIENT_INFO = {"name": "ai-runtime-mcp-client", "version": "1.0.0"}
logger = logging.getLogger(__name__)


class StreamableHTTPTransport(BaseTransport):
    def __init__(self, session: ConnectionSession, parser: ResponseParser | None = None) -> None:
        self._session, self._parser = session, parser or ResponseParser()

    async def invoke(self, request: ToolRequest) -> ToolResponse:
        await self._ensure_initialized()
        started = time.perf_counter()
        wire_payload = request.wire_payload()
        headers = self._headers(request)
        logger.info('mcp_transport_request_json\n%s', pretty_json({
            'server': self._session.server.name,
            'endpoint': f"{self._session.server.base_url}{self._session.server.endpoint_path}",
            'headers': headers,
            'payload': wire_payload,
        }))
        try:
            response = await (await self._session.client()).post(
                self._session.server.endpoint_path,
                json=wire_payload,
                headers=headers,
            )
        except httpx.TimeoutException as exc:
            raise MCPTimeoutError(str(exc)) from exc
        except httpx.HTTPError as exc:
            raise MCPConnectionError(str(exc)) from exc
        self._capture_session_id(response)
        if response.status_code in (401, 403):
            raise AuthenticationError(f"MCP server returned {response.status_code}")
        if response.status_code == 404 and self._session.session_id():
            self._session.clear_session_state()
            raise MCPConnectionError("MCP session expired or MCP endpoint was not found")
        if response.status_code >= 500:
            raise ServerUnavailableError(f"MCP server returned {response.status_code}")
        if response.status_code in (202, 204):
            raise ResponseParseError("MCP tools/call request did not return a JSON-RPC response")
        if response.status_code >= 400 and not response.content:
            raise InvocationError(f"MCP server returned {response.status_code}")
        payload = self._response_payload(response, request)
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        logger.info('mcp_transport_response_json\n%s', pretty_json({
            'server': self._session.server.name,
            'status_code': response.status_code,
            'latency_ms': latency_ms,
            'payload': payload,
        }))
        return self._parser.parse(payload, request, latency_ms / 1000, response.status_code)

    def _headers(self, request: ToolRequest) -> dict[str, str]:
        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            "Mcp-Method": request.method,
            "Mcp-Name": request.params.name,
        }
        session_id = self._session.session_id()
        if session_id:
            headers["Mcp-Session-Id"] = session_id
        protocol_version = self._session.protocol_version()
        if protocol_version:
            headers["MCP-Protocol-Version"] = protocol_version
        return headers

    async def _ensure_initialized(self) -> None:
        if self._session.is_initialized():
            return
        async with self._session.initialization_lock():
            if self._session.is_initialized():
                return
            await self._initialize_session()

    async def _initialize_session(self) -> None:
        initialize_id = str(uuid4())
        payload = {
            "jsonrpc": "2.0",
            "id": initialize_id,
            "method": "initialize",
            "params": {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": MCP_CLIENT_INFO,
            },
        }
        response = await self._post_protocol_message(payload, self._protocol_headers("initialize"))
        self._capture_session_id(response)
        result = self._parse_initialize_response(response, initialize_id)
        protocol_version = result["protocolVersion"]
        await self._send_initialized_notification(protocol_version)
        self._session.mark_initialized(protocol_version)

    async def _send_initialized_notification(self, protocol_version: str) -> None:
        payload = {"jsonrpc": "2.0", "method": "notifications/initialized"}
        response = await self._post_protocol_message(
            payload,
            self._protocol_headers("notifications/initialized", protocol_version=protocol_version),
        )
        self._capture_session_id(response)
        if response.status_code >= 400:
            raise MCPConnectionError(f"MCP initialized notification failed with {response.status_code}")

    async def _post_protocol_message(self, payload: dict[str, Any], headers: dict[str, str]) -> httpx.Response:
        try:
            return await (await self._session.client()).post(
                self._session.server.endpoint_path,
                json=payload,
                headers=headers,
            )
        except httpx.TimeoutException as exc:
            raise MCPTimeoutError(str(exc)) from exc
        except httpx.HTTPError as exc:
            raise MCPConnectionError(str(exc)) from exc

    def _protocol_headers(self, method: str, *, protocol_version: str | None = None) -> dict[str, str]:
        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            "Mcp-Method": method,
        }
        session_id = self._session.session_id()
        if session_id:
            headers["Mcp-Session-Id"] = session_id
        if protocol_version:
            headers["MCP-Protocol-Version"] = protocol_version
        return headers

    def _parse_initialize_response(self, response: httpx.Response, request_id: str) -> dict[str, Any]:
        if response.status_code in (401, 403):
            raise AuthenticationError(f"MCP server returned {response.status_code}")
        if response.status_code >= 500:
            raise ServerUnavailableError(f"MCP server returned {response.status_code}")
        if response.status_code >= 400:
            raise MCPConnectionError(f"MCP initialize failed with {response.status_code}")
        payload = self._response_payload_for_id(response, request_id)
        if not isinstance(payload, dict):
            raise ResponseParseError("Expected MCP initialize response object")
        if payload.get("jsonrpc") != "2.0":
            raise ResponseParseError("MCP initialize response must include jsonrpc='2.0'")
        if str(payload.get("id")) != request_id:
            raise ResponseParseError("MCP initialize response id does not match request id")
        if "error" in payload:
            raise MCPConnectionError("MCP initialize returned a JSON-RPC error")
        result = payload.get("result")
        if not isinstance(result, dict):
            raise ResponseParseError("MCP initialize response must include a result object")
        protocol_version = result.get("protocolVersion")
        if not isinstance(protocol_version, str) or not protocol_version:
            raise ResponseParseError("MCP initialize result must include protocolVersion")
        if not isinstance(result.get("capabilities"), dict):
            raise ResponseParseError("MCP initialize result must include capabilities")
        if not isinstance(result.get("serverInfo"), dict):
            raise ResponseParseError("MCP initialize result must include serverInfo")
        return result

    def _capture_session_id(self, response: httpx.Response) -> None:
        session_id = response.headers.get("Mcp-Session-Id")
        if session_id:
            self._session.set_session_id(session_id)

    def _response_payload(self, response: httpx.Response, request: ToolRequest) -> Any:
        return self._response_payload_for_id(response, request.id)

    def _response_payload_for_id(self, response: httpx.Response, request_id: str) -> Any:
        content_type = response.headers.get("content-type", "").lower()
        if "text/event-stream" in content_type:
            return self._parse_event_stream(response.text, request_id)
        try:
            return response.json()
        except ValueError as exc:
            raise ResponseParseError("MCP server did not return JSON") from exc

    def _parse_event_stream(self, body: str, request_id: str) -> Any:
        for event in body.replace("\r\n", "\n").split("\n\n"):
            data_lines = [line[5:].strip() for line in event.splitlines() if line.startswith("data:")]
            if not data_lines:
                continue
            try:
                payload = json.loads("\n".join(data_lines))
            except json.JSONDecodeError as exc:
                raise ResponseParseError("MCP event stream contained invalid JSON") from exc
            if isinstance(payload, dict) and str(payload.get("id")) == request_id:
                return payload
        raise ResponseParseError("MCP event stream did not contain a matching JSON-RPC response")
