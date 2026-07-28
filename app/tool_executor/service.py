from __future__ import annotations

import logging
import json
import re
from time import perf_counter
from typing import Any

from app.mcp_client import MCPClient
from app.mcp_client.exceptions import (
    AuthenticationError,
    MCPConnectionError,
    MCPTimeoutError,
    ResponseParseError,
    ServerNotFoundError,
    ServerUnavailableError,
)
from app.models.planner import ExecutionPlanStep, PlannerOutput
from app.models.runtime import RuntimeContext
from app.tool_executor.exceptions import ToolExecutionError, ToolResolutionError
from app.tool_registry.exceptions import DuplicateToolException, ToolNotFoundException
from app.tool_registry.models import ResolvedTool
from app.tool_registry.service import ToolRegistryService
from app.utils.json_logging import pretty_json

logger = logging.getLogger(__name__)


class ToolExecutorService:
    def __init__(self, tool_registry_service: ToolRegistryService, mcp_client: MCPClient) -> None:
        self._tool_registry_service = tool_registry_service
        self._mcp_client = mcp_client

    async def execute(
        self,
        *,
        planner_output: PlannerOutput,
        runtime_context: RuntimeContext,
        request_id: str,
        correlation_id: str,
        trace_id: str,
    ) -> dict[str, Any] | None:
        if not planner_output.requires_tool:
            return None
        if planner_output.execution_plan:
            return await self._execute_plan(
                planner_output=planner_output,
                runtime_context=runtime_context,
                request_id=request_id,
                correlation_id=correlation_id,
                trace_id=trace_id,
            )
        if not all((planner_output.domain, planner_output.service, planner_output.entity, planner_output.operation)):
            raise ToolResolutionError(
                'Planner output is missing one or more resolution fields: domain, service, entity, operation.'
            )

        lookup_started = perf_counter()
        resolved_tool = self._resolve_tool(planner_output)
        lookup_latency_ms = round((perf_counter() - lookup_started) * 1000, 2)
        logger.info('tool_registry_lookup_json\n%s', pretty_json({
            'requested_lookup': {
                'domain': planner_output.domain,
                'service': planner_output.service,
                'entity': planner_output.entity,
                'operation': planner_output.operation,
            },
            'resolved_tool': resolved_tool.tool.name,
            'resolved_mcp_server': resolved_tool.server,
            'lookup_latency_ms': lookup_latency_ms,
        }))

        try:
            self._validate_required_parameters(resolved_tool, planner_output.parameters, 'single_step')
        except ToolExecutionError as exc:
            logger.info('tool_execution_skipped_missing_required_parameters_json\n%s', pretty_json({
                'tool': resolved_tool.tool.name,
                'server': resolved_tool.server,
                'required_parameters': resolved_tool.tool.required_parameters,
                'provided_parameters': sorted(planner_output.parameters.keys()),
                'error_code': exc.code,
            }))
            return self._dependency_error(
                message=exc.message,
                step_id='single_step',
                steps=[],
                details={
                    'code': exc.code,
                    'required_parameters': resolved_tool.tool.required_parameters,
                    'provided_parameters': sorted(planner_output.parameters.keys()),
                },
            )

        mcp_arguments = self._build_arguments(planner_output.parameters, runtime_context, request_id, correlation_id, trace_id)
        mcp_context = self._build_context(runtime_context, request_id, correlation_id, trace_id)
        logger.info('tool_execution_request_json\n%s', pretty_json(self._redact_sensitive({
            'tool': resolved_tool.tool.name,
            'server': resolved_tool.server,
            'arguments': mcp_arguments,
            'context': mcp_context,
        })))

        tool_started = perf_counter()
        try:
            response = await self._mcp_client.invoke_tool(
                resolved_tool.server,
                resolved_tool.tool.name,
                mcp_arguments,
                context=mcp_context,
                trace_id=trace_id,
                request_id=request_id,
            )
        except AuthenticationError as exc:
            raise ToolExecutionError('Authentication failed while calling MCP server.', code='AUTHENTICATION_ERROR', status_code=401) from exc
        except ServerNotFoundError as exc:
            raise ToolExecutionError(str(exc), code='SERVER_ERROR', status_code=502) from exc
        except MCPTimeoutError as exc:
            raise ToolExecutionError('MCP server request timed out.', code='TIMEOUT', status_code=504) from exc
        except (MCPConnectionError, ServerUnavailableError) as exc:
            raise ToolExecutionError('Transport error while calling MCP server.', code='TRANSPORT_ERROR', status_code=502) from exc
        except ResponseParseError as exc:
            raise ToolExecutionError('Invalid MCP server response.', code='SERVER_ERROR', status_code=502) from exc

        execution_latency_ms = round((perf_counter() - tool_started) * 1000, 2)
        normalized = {
            'tool_name': resolved_tool.tool.name,
            'server': resolved_tool.server,
            'status': response.status.value,
            'success': response.success,
            'response_type': resolved_tool.tool.response_type.value,
            'data': response.tool_results,
            'error': response.error.model_dump() if response.error else None,
            'registry_lookup_latency_ms': lookup_latency_ms,
            'tool_execution_latency_ms': execution_latency_ms,
        }
        logger.info('tool_execution_response_json\n%s', pretty_json(normalized))
        logger.info(
            'tool_execution_completed planner_latency_ms=%s registry_lookup_latency_ms=%s tool_execution_latency_ms=%s mcp_server=%s tool_name=%s status=%s duration=%s',
            0.0,
            lookup_latency_ms,
            execution_latency_ms,
            resolved_tool.server,
            resolved_tool.tool.name,
            response.status.value,
            execution_latency_ms,
        )
        return normalized

    async def _execute_plan(
        self,
        *,
        planner_output: PlannerOutput,
        runtime_context: RuntimeContext,
        request_id: str,
        correlation_id: str,
        trace_id: str,
    ) -> dict[str, Any]:
        step_results: dict[str, dict[str, Any]] = {}
        ordered_results: list[dict[str, Any]] = []
        plan_order_result = self._order_plan_steps(planner_output.execution_plan)
        if isinstance(plan_order_result, dict):
            return plan_order_result
        ordered_steps = plan_order_result
        logger.info('tool_execution_plan_started step_count=%s', len(ordered_steps))

        for index, step in enumerate(ordered_steps, start=1):
            step_id = step.step_id or f'step_{index}'
            step_output = PlannerOutput(
                intent=step.intent or '',
                requires_tool=True,
                domain=step.domain,
                service=step.service,
                entity=step.entity,
                operation=step.operation,
                tool=step.tool,
                parameters={},
                adapter=planner_output.adapter,
                model=planner_output.model,
            )
            try:
                resolved_tool = self._resolve_tool(step_output)
                parameters = self._resolve_step_parameters(step, step_results)
                self._validate_required_parameters(resolved_tool, parameters, step_id)
            except ToolExecutionError as exc:
                return self._dependency_error(
                    message=exc.message,
                    step_id=step_id,
                    steps=ordered_results,
                    details={'code': exc.code},
                )
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                return self._dependency_error(
                    message=f'Execution plan step {step_id} could not resolve parameter binding: {exc}',
                    step_id=step_id,
                    steps=ordered_results,
                    details={'code': 'INVALID_PARAMETER_BINDING'},
                )
            except ToolResolutionError as exc:
                return self._dependency_error(
                    message=exc.message,
                    step_id=step_id,
                    steps=ordered_results,
                    details={'code': exc.code},
                )
            step_output.parameters = parameters
            logger.info('tool_execution_plan_step_request_json\n%s', pretty_json(self._redact_sensitive({
                'step_id': step_id,
                'depends_on': step.depends_on,
                'parameter_bindings': step.parameter_bindings,
                'resolved_parameters': parameters,
                'target': {
                    'domain': step.domain,
                    'service': step.service,
                    'entity': step.entity,
                    'operation': step.operation,
                    'intent': step.intent,
                },
            })))
            result = await self.execute(
                planner_output=step_output,
                runtime_context=runtime_context,
                request_id=request_id,
                correlation_id=correlation_id,
                trace_id=trace_id,
            )
            assert result is not None
            step_result = {
                'step_id': step_id,
                'depends_on': step.depends_on,
                'parameter_bindings': step.parameter_bindings,
                'result': result,
            }
            step_results[step_id] = result
            ordered_results.append(step_result)
            if result.get('success') is False or result.get('status') == 'error':
                return {
                    'status': 'error',
                    'success': False,
                    'steps': ordered_results,
                    'failed_step_id': step_id,
                    'data': result.get('data'),
                    'error': result.get('error') or {'message': f'Execution plan failed at step {step_id}.'},
                }

        final_step = ordered_results[-1]
        final_result = final_step['result']
        plan_result = {
            'status': final_result.get('status', 'success'),
            'success': final_result.get('success', True),
            'tool_name': final_result.get('tool_name'),
            'server': final_result.get('server'),
            'response_type': final_result.get('response_type'),
            'data': final_result.get('data'),
            'error': final_result.get('error'),
            'steps': ordered_results,
            'final_step_id': final_step['step_id'],
        }
        logger.info('tool_execution_plan_completed_json\n%s', pretty_json(self._redact_sensitive(plan_result)))
        return plan_result

    @classmethod
    def _order_plan_steps(cls, steps: list[ExecutionPlanStep]) -> list[ExecutionPlanStep] | dict[str, Any]:
        step_by_id: dict[str, ExecutionPlanStep] = {}
        for index, step in enumerate(steps, start=1):
            step_id = step.step_id or f'step_{index}'
            if step_id in step_by_id:
                return cls._dependency_error(
                    message=f'Duplicate execution plan step id: {step_id}',
                    step_id=step_id,
                    steps=[],
                    details={'code': 'DUPLICATE_STEP_ID'},
                )
            step_by_id[step_id] = step

        dependencies = {
            step_id: cls._declared_dependencies(step)
            for step_id, step in step_by_id.items()
        }
        for step_id, depends_on in dependencies.items():
            missing = sorted(dep for dep in depends_on if dep not in step_by_id)
            if missing:
                return cls._dependency_error(
                    message=f'Execution plan step {step_id} depends on unknown step(s): {missing}',
                    step_id=step_id,
                    steps=[],
                    details={'code': 'MISSING_DEPENDENCY', 'missing_dependencies': missing},
                )

        ordered_ids: list[str] = []
        temporary: set[str] = set()
        permanent: set[str] = set()

        def visit(step_id: str) -> bool:
            if step_id in permanent:
                return True
            if step_id in temporary:
                return False
            temporary.add(step_id)
            for dependency_id in dependencies[step_id]:
                if not visit(dependency_id):
                    return False
            temporary.remove(step_id)
            permanent.add(step_id)
            ordered_ids.append(step_id)
            return True

        for step_id in step_by_id:
            if not visit(step_id):
                return cls._dependency_error(
                    message='Execution plan contains a circular dependency.',
                    step_id=step_id,
                    steps=[],
                    details={'code': 'CIRCULAR_DEPENDENCY'},
                )

        return [step_by_id[step_id] for step_id in ordered_ids]

    @classmethod
    def _declared_dependencies(cls, step: ExecutionPlanStep) -> set[str]:
        dependencies = set(step.depends_on)
        for binding in step.parameter_bindings.values():
            step_id = cls._binding_step_id(binding)
            if step_id:
                dependencies.add(step_id)
        return dependencies

    @staticmethod
    def _binding_step_id(binding: Any) -> str | None:
        if isinstance(binding, dict):
            value = binding.get('from_step') or binding.get('source_step') or binding.get('step')
            return value if isinstance(value, str) else None
        if isinstance(binding, str):
            stripped = binding.strip()
            brace_match = re.fullmatch(r'\$\{(?P<step>[^.}]+)\.(?P<path>.+)\}', stripped)
            if brace_match:
                return brace_match.group('step')
            short_match = re.fullmatch(r'\$(?P<step>[A-Za-z0-9_-]+)\.(?P<path>.+)', stripped)
            if short_match:
                return short_match.group('step')
        return None

    @staticmethod
    def _dependency_error(
        *,
        message: str,
        step_id: str | None,
        steps: list[dict[str, Any]],
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            'status': 'error',
            'success': False,
            'error': {
                'code': 'DEPENDENCY_RESOLUTION_ERROR',
                'message': message,
                'data': details or {},
            },
            'data': None,
            'steps': steps,
            'failed_step_id': step_id,
        }

    def _resolve_tool(self, planner_output: PlannerOutput) -> ResolvedTool:
        domain = planner_output.domain or ''
        service = planner_output.service or ''
        entity = planner_output.entity or ''
        operation = planner_output.operation or ''
        try:
            return self._tool_registry_service.find_tool(domain, service, entity, operation)
        except ToolNotFoundException as exc:
            raise ToolResolutionError(
                f'No tool found for domain={domain} service={service} entity={entity} operation={operation}'
            ) from exc
        except DuplicateToolException as exc:
            raise ToolResolutionError(
                f'Multiple tools found for domain={domain} service={service} entity={entity} operation={operation}'
            ) from exc

    @classmethod
    def _resolve_step_parameters(
        cls,
        step: ExecutionPlanStep,
        step_results: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        parameters = dict(step.parameters)
        for parameter_name, binding in step.parameter_bindings.items():
            parameters[parameter_name] = cls._resolve_binding(binding, step_results)
        return parameters

    @classmethod
    def _resolve_binding(cls, binding: Any, step_results: dict[str, dict[str, Any]]) -> Any:
        if isinstance(binding, dict):
            step_id = binding.get('from_step') or binding.get('source_step') or binding.get('step')
            path = binding.get('path')
            if not isinstance(step_id, str) or not isinstance(path, str):
                raise ToolExecutionError(
                    'Invalid execution plan parameter binding. Expected from_step and path.',
                    code='INVALID_EXECUTION_PLAN',
                    status_code=422,
                )
            if step_id not in step_results:
                raise ToolExecutionError(
                    f'Execution plan binding references unknown step: {step_id}',
                    code='INVALID_EXECUTION_PLAN',
                    status_code=422,
                )
            return cls._resolve_path_with_business_fallback(step_results[step_id], path)
        if isinstance(binding, str):
            match = re.fullmatch(r'\$\{(?P<step>[^.}]+)\.(?P<path>.+)\}', binding.strip())
            if match is None:
                match = re.fullmatch(r'\$(?P<step>[A-Za-z0-9_-]+)\.(?P<path>.+)', binding.strip())
            if match:
                step_id = match.group('step')
                if step_id not in step_results:
                    raise ToolExecutionError(
                        f'Execution plan binding references unknown step: {step_id}',
                        code='INVALID_EXECUTION_PLAN',
                        status_code=422,
                    )
                return cls._resolve_path_with_business_fallback(step_results[step_id], f"$.{match.group('path')}")
        return binding

    @staticmethod
    def _validate_required_parameters(resolved_tool: ResolvedTool, parameters: dict[str, Any], step_id: str) -> None:
        missing = [
            parameter
            for parameter in resolved_tool.tool.required_parameters
            if parameter not in parameters or parameters[parameter] in (None, '')
        ]
        if missing:
            raise ToolExecutionError(
                f'Execution plan step {step_id} is missing required parameter(s): {missing}',
                code='MISSING_REQUIRED_PARAMETERS',
                status_code=422,
            )

    @classmethod
    def _resolve_path_with_business_fallback(cls, step_result: dict[str, Any], path: str) -> Any:
        try:
            return cls._resolve_path(step_result, path)
        except KeyError:
            business_data = cls._extract_business_data(step_result)
            try:
                return cls._resolve_path(business_data, path)
            except KeyError:
                if path.strip().startswith('$.data'):
                    return cls._resolve_path(business_data, '$' + path.strip()[len('$.data'):])
                raise

    @classmethod
    def _extract_business_data(cls, value: Any) -> Any:
        parsed = cls._parse_json_string(value)
        if parsed is not value:
            return cls._extract_business_data(parsed)
        if isinstance(value, dict):
            data = value.get('data')
            if isinstance(data, dict) and isinstance(data.get('content'), list):
                extracted = [cls._extract_business_data(item) for item in data['content']]
                extracted = [item for item in extracted if item not in (None, '')]
                return extracted[0] if len(extracted) == 1 else extracted
            if isinstance(value.get('content'), list):
                extracted = [cls._extract_business_data(item) for item in value['content']]
                extracted = [item for item in extracted if item not in (None, '')]
                return extracted[0] if len(extracted) == 1 else extracted
            if 'text' in value and isinstance(value['text'], str):
                return cls._extract_business_data(value['text'])
            if data is not None:
                return cls._extract_business_data(data)
        return value

    @classmethod
    def _resolve_path(cls, source: Any, path: str) -> Any:
        current = source
        normalized_path = path.strip()
        if normalized_path.startswith('$'):
            normalized_path = normalized_path[1:]
        if normalized_path.startswith('.'):
            normalized_path = normalized_path[1:]
        if not normalized_path:
            return current
        for segment in cls._split_path(normalized_path):
            current = cls._apply_path_segment(current, segment)
        return current

    @staticmethod
    def _split_path(path: str) -> list[str]:
        segments: list[str] = []
        current = []
        bracket_depth = 0
        for char in path:
            if char == '[':
                bracket_depth += 1
            elif char == ']':
                bracket_depth -= 1
            if char == '.' and bracket_depth == 0:
                if current:
                    segments.append(''.join(current))
                    current = []
                continue
            current.append(char)
        if current:
            segments.append(''.join(current))
        return segments

    @classmethod
    def _apply_path_segment(cls, value: Any, segment: str) -> Any:
        field_match = re.match(r'^(?P<field>[^\[]+)', segment)
        remainder = segment
        current = value
        if field_match:
            field = field_match.group('field')
            current = cls._select_field(current, field)
            remainder = segment[len(field):]
        for bracket in re.findall(r'\[([^\]]+)\]', remainder):
            current = cls._apply_bracket(current, bracket)
        return current

    @classmethod
    def _select_field(cls, value: Any, field: str) -> Any:
        if isinstance(value, dict):
            if field in value:
                return value[field]
            requested = cls._field_key(field)
            for key, item in value.items():
                if cls._field_key(str(key)) == requested:
                    return item
        if isinstance(value, list):
            return [cls._select_field(item, field) for item in value]
        raise KeyError(field)

    @classmethod
    def _apply_bracket(cls, value: Any, bracket: str) -> Any:
        if bracket.isdigit():
            return value[int(bracket)]
        filter_match = re.fullmatch(
            r"\?\(@\.(?P<field>[A-Za-z0-9_]+)\s*==\s*(?P<expected>true|false|null|'[^']*'|\"[^\"]*\"|[^)]+)\)",
            bracket.strip(),
        )
        if filter_match and isinstance(value, list):
            field = filter_match.group('field')
            expected = cls._parse_filter_value(filter_match.group('expected'))
            matches = [item for item in value if cls._select_field(item, field) == expected]
            if not matches:
                raise KeyError(bracket)
            return matches[0]
        raise KeyError(bracket)

    @staticmethod
    def _parse_filter_value(value: str) -> Any:
        stripped = value.strip()
        if stripped == 'true':
            return True
        if stripped == 'false':
            return False
        if stripped == 'null':
            return None
        if (stripped.startswith('"') and stripped.endswith('"')) or (stripped.startswith("'") and stripped.endswith("'")):
            return stripped[1:-1]
        return stripped

    @staticmethod
    def _parse_json_string(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        stripped = value.strip()
        if not stripped or stripped[0] not in '[{':
            return value
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return value

    @staticmethod
    def _field_key(value: str) -> str:
        return re.sub(r'[^a-zA-Z0-9]+', '', value).lower()

    @staticmethod
    def _build_arguments(
        parameters: dict[str, object],
        runtime_context: RuntimeContext,
        request_id: str,
        correlation_id: str,
        trace_id: str,
    ) -> dict[str, Any]:
        arguments = dict(parameters)
        arguments['context'] = {
            'organization_id': runtime_context.organization_id or '',
            'branch_id': runtime_context.branch_id or '',
            'user_id': runtime_context.user_id,
            'authorization': ToolExecutorService._authorization_header(runtime_context.jwt),
            'tenant_id': runtime_context.tenant_id,
            'locale': runtime_context.locale,
            'app_id': runtime_context.app_id,
            'session_id': runtime_context.session_id,
            'request_id': request_id,
            'correlation_id': correlation_id,
            'trace_id': trace_id,
        }
        return arguments

    @staticmethod
    def _build_context(
        runtime_context: RuntimeContext,
        request_id: str,
        correlation_id: str,
        trace_id: str,
    ) -> dict[str, Any]:
        return {
            'request_id': request_id,
            'correlation_id': correlation_id,
            'trace_id': trace_id,
            'user_id': runtime_context.user_id,
            'subject': runtime_context.subject,
            'org_id': runtime_context.organization_id,
            'organization_id': runtime_context.organization_id,
            'branch_id': runtime_context.branch_id,
            'app_id': runtime_context.app_id,
            'application_ids': runtime_context.application_ids,
            'roles': runtime_context.roles,
            'permissions': runtime_context.permissions,
            'jwt': runtime_context.jwt,
            'locale': runtime_context.locale,
            'tenant_id': runtime_context.tenant_id,
            'session_id': runtime_context.session_id,
        }

    @staticmethod
    def _authorization_header(jwt_token: str | None) -> str | None:
        if not jwt_token:
            return None
        if jwt_token.lower().startswith('bearer '):
            return jwt_token
        return f'Bearer {jwt_token}'

    @classmethod
    def _redact_sensitive(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: '<redacted>' if key in {'jwt', 'authorization'} and item else cls._redact_sensitive(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [cls._redact_sensitive(item) for item in value]
        return value
