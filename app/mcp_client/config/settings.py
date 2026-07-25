"""Top-level settings passed to the MCP subsystem."""
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field
from ..models.connection import ConnectionSettings


class MCPSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    servers_file: Path = Field(default=Path(__file__).with_name("servers.yaml"))
    connection: ConnectionSettings = Field(default_factory=ConnectionSettings)
