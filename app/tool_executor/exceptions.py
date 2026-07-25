from app.exceptions.errors import AppException


class PlannerExecutionError(AppException):
    status_code = 400
    code = 'PLANNER_ERROR'


class ToolExecutionError(AppException):
    status_code = 502
    code = 'TOOL_EXECUTION_ERROR'


class ToolResolutionError(AppException):
    status_code = 404
    code = 'TOOL_RESOLUTION_ERROR'
