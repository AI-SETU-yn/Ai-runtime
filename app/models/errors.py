from pydantic import BaseModel


class ErrorResponse(BaseModel):
    code: str
    message: str
    request_id: str | None = None
    correlation_id: str | None = None
