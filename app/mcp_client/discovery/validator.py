"""Validation for loaded server configuration."""
from ..exceptions import DuplicateServerError
from ..models.server import ServerConfig


class ServerConfigValidator:
    @staticmethod
    def validate(servers: list[ServerConfig]) -> None:
        names = [server.name for server in servers]
        urls = [server.base_url for server in servers]
        if len(names) != len(set(names)):
            raise DuplicateServerError("Duplicate MCP server names in configuration")
        if len(urls) != len(set(urls)):
            raise DuplicateServerError("Duplicate MCP server URLs in configuration")
