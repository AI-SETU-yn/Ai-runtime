import logging

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.exceptions.errors import AppException
from app.models.errors import ErrorResponse

logger = logging.getLogger(__name__)


async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
    logger.warning(
        'app_exception_handled code=%s status_code=%s path=%s',
        exc.code,
        exc.status_code,
        request.url.path,
    )
    body = ErrorResponse(
        code=exc.code,
        message=exc.message,
        request_id=getattr(request.state, 'request_id', None),
        correlation_id=getattr(request.state, 'correlation_id', None),
    )
    return JSONResponse(status_code=exc.status_code, content=body.model_dump())


async def request_validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    logger.warning('request_validation_failed path=%s error_count=%s', request.url.path, len(exc.errors()))
    body = ErrorResponse(
        code='VALIDATION_ERROR',
        message=str(exc),
        request_id=getattr(request.state, 'request_id', None),
        correlation_id=getattr(request.state, 'correlation_id', None),
    )
    return JSONResponse(status_code=422, content=body.model_dump())


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception('unhandled_exception path=%s error_type=%s', request.url.path, type(exc).__name__)
    body = ErrorResponse(
        code='INTERNAL_ERROR',
        message='An unexpected error occurred.',
        request_id=getattr(request.state, 'request_id', None),
        correlation_id=getattr(request.state, 'correlation_id', None),
    )
    return JSONResponse(status_code=500, content=body.model_dump())
