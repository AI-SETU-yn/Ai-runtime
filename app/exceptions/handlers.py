from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.exceptions.errors import AppException
from app.models.errors import ErrorResponse


async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
    body = ErrorResponse(
        code=exc.code,
        message=exc.message,
        request_id=getattr(request.state, 'request_id', None),
        correlation_id=getattr(request.state, 'correlation_id', None),
    )
    return JSONResponse(status_code=exc.status_code, content=body.model_dump())


async def request_validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    body = ErrorResponse(
        code='VALIDATION_ERROR',
        message=str(exc),
        request_id=getattr(request.state, 'request_id', None),
        correlation_id=getattr(request.state, 'correlation_id', None),
    )
    return JSONResponse(status_code=422, content=body.model_dump())


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    body = ErrorResponse(
        code='INTERNAL_ERROR',
        message='An unexpected error occurred.',
        request_id=getattr(request.state, 'request_id', None),
        correlation_id=getattr(request.state, 'correlation_id', None),
    )
    return JSONResponse(status_code=500, content=body.model_dump())
