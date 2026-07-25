class AppException(Exception):
    status_code = 500
    code = 'INTERNAL_ERROR'

    def __init__(self, message: str, *, code: str | None = None, status_code: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        if status_code is not None:
            self.status_code = status_code


class UnauthorizedError(AppException):
    status_code = 401
    code = 'UNAUTHORIZED'


class ForbiddenError(AppException):
    status_code = 403
    code = 'FORBIDDEN'


class PlannerError(AppException):
    status_code = 502
    code = 'PLANNER_ERROR'


class ModelGatewayError(AppException):
    status_code = 502
    code = 'MODEL_GATEWAY_ERROR'


class ModelGatewayTimeoutError(ModelGatewayError):
    status_code = 504
    code = 'MODEL_GATEWAY_TIMEOUT'
