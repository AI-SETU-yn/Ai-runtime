from pydantic import BaseModel, ConfigDict, Field


class ConnectionSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    connect_timeout_seconds: float = Field(default=5.0, gt=0)
    read_timeout_seconds: float = Field(default=30.0, gt=0)
    write_timeout_seconds: float = Field(default=10.0, gt=0)
    pool_timeout_seconds: float = Field(default=5.0, gt=0)
    verify_tls: bool = True
    max_connections: int = Field(default=100, gt=0)
    max_keepalive_connections: int = Field(default=20, ge=0)
    keepalive_expiry_seconds: float = Field(default=30.0, gt=0)
