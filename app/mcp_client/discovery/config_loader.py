"""YAML configuration loader kept separate from registry concerns."""
import os
from pathlib import Path
from typing import Any
import yaml
from ..models.server import ServerConfig
from .validator import ServerConfigValidator


class ServerConfigLoader:
    def load(self, path: Path) -> list[ServerConfig]:
        payload: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        entries = payload.get("servers", {})
        if not isinstance(entries, dict):
            raise ValueError("'servers' must be a mapping")
        invalid = [name for name, value in entries.items() if not isinstance(name, str) or not isinstance(value, dict)]
        if invalid:
            raise ValueError("each MCP server entry must be a named mapping")
        servers = [ServerConfig(name=name, **self._resolve_entry(value)) for name, value in entries.items()]
        ServerConfigValidator.validate(servers)
        return servers

    def _resolve_entry(self, entry: dict[str, Any]) -> dict[str, Any]:
        return {key: self._resolve_value(value) for key, value in entry.items()}

    def _resolve_value(self, value: Any) -> Any:
        if not isinstance(value, str) or not value.startswith("env:"):
            return value
        variable, default = self._parse_env_expression(value)
        return os.getenv(variable) or default

    def _parse_env_expression(self, value: str) -> tuple[str, str]:
        expression = value.removeprefix("env:")
        variable, separator, default = expression.partition("|")
        if not variable or not separator or not default:
            raise ValueError("environment-backed config values must use env:VARIABLE|default")
        return variable, default
