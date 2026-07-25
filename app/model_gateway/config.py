from pydantic import BaseModel


class ModelGatewayRequestConfig(BaseModel):
    url: str
    path: str
    timeout_seconds: float
    connect_timeout_seconds: float
    read_timeout_seconds: float
    max_retries: int
