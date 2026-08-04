from __future__ import annotations

import logging
from typing import Any

from app.models.planner import ExecutionPlanStep, PlannerOutput
from app.models.runtime import RuntimeContext
from app.rbac.client import RBACClient, RBACClientError
from app.rbac.models import AuthorizationDecision, AuthorizationRequest, AuthorizationStep
from app.tool_registry.exceptions import DuplicateToolException, ToolNotFoundException
from app.tool_registry.models import ResolvedTool
from app.tool_registry.service import ToolRegistryService
from app.utils.json_logging import pretty_json
from app.utils.redaction import redact_sensitive

logger = logging.getLogger(__name__)


class RBACAuthorizationService:
    def __init__(
        self,
        *,
        enabled: bool,
        rbac_client: RBACClient,
        tool_registry_service: ToolRegistryService,
    ) -> None:
        self._enabled = enabled
        self._rbac_client = rbac_client
        self._tool_registry_service = tool_registry_service

    async def authorize_plan(
        self,
        *,
        planner_output: PlannerOutput,
        runtime_context: RuntimeContext,
        request_id: str,
        correlation_id: str,
        trace_id: str,
    ) -> AuthorizationDecision:
        if not self._enabled or not planner_output.requires_tool:
            return AuthorizationDecision(allowed=True, reason='RBAC authorization disabled.')

        steps = self._authorization_steps(planner_output)
        request = AuthorizationRequest(
            user_id=runtime_context.user_id,
            organization_id=runtime_context.organization_id,
            branch_id=runtime_context.branch_id,
            app_id=runtime_context.app_id,
            tenant_id=runtime_context.tenant_id,
            session_id=runtime_context.session_id,
            request_id=request_id,
            correlation_id=correlation_id,
            trace_id=trace_id,
            steps=steps,
        )
        logger.info('rbac_authorization_plan_json\n%s', pretty_json(redact_sensitive(request.model_dump(exclude_none=True))))
        try:
            decision = await self._rbac_client.authorize(request, runtime_context.jwt)
        except RBACClientError as exc:
            first_step = steps[0] if steps else None
            decision = AuthorizationDecision(
                allowed=False,
                step_id=first_step.step_id if first_step else None,
                tool=first_step.tool if first_step else planner_output.tool,
                capability=first_step.capability if first_step else None,
                domain=first_step.domain if first_step else planner_output.domain,
                service=first_step.service if first_step else planner_output.service,
                entity=first_step.entity if first_step else planner_output.entity,
                operation=first_step.operation if first_step else planner_output.operation,
                reason=str(exc),
                source='rbac_unavailable',
            )
        logger.info('rbac_authorization_decision_json\n%s', pretty_json(redact_sensitive(decision.model_dump(exclude_none=True))))
        return decision

    @classmethod
    def authorization_failure_result(cls, decision: AuthorizationDecision) -> dict[str, Any]:
        return {
            'status': 'authorization_failure',
            'success': False,
            'response_type': 'authorization_failure',
            'tool_name': decision.tool,
            'data': None,
            'error': {
                'code': 'AUTHORIZATION_FAILED',
                'message': decision.reason or 'RBAC denied access to the requested tool.',
                'reason': decision.reason,
                'denied_step_id': decision.step_id,
                'denied_tool': decision.tool,
                'denied_capability': decision.capability,
                'domain': decision.domain,
                'service': decision.service,
                'entity': decision.entity,
                'operation': decision.operation,
                'source': decision.source,
            },
        }

    def _authorization_steps(self, planner_output: PlannerOutput) -> list[AuthorizationStep]:
        if planner_output.execution_plan:
            return [
                self._step_from_execution_plan_item(step, index)
                for index, step in enumerate(planner_output.execution_plan, start=1)
            ]
        resolved_tool = self._resolve_tool(
            tool_name=planner_output.tool,
            domain=planner_output.domain,
            service=planner_output.service,
            entity=planner_output.entity,
            operation=planner_output.operation,
        )
        return [
            AuthorizationStep(
                step_id='single_step',
                tool=resolved_tool.tool.name,
                capability=resolved_tool.tool.capability,
                domain=resolved_tool.domain,
                service=resolved_tool.service,
                entity=resolved_tool.tool.entity,
                operation=resolved_tool.tool.operation.value,
                parameters=dict(planner_output.parameters),
            )
        ]

    def _step_from_execution_plan_item(self, step: ExecutionPlanStep, index: int) -> AuthorizationStep:
        resolved_tool = self._resolve_tool(
            tool_name=step.tool,
            domain=step.domain,
            service=step.service,
            entity=step.entity,
            operation=step.operation,
        )
        return AuthorizationStep(
            step_id=step.step_id or f'step_{index}',
            tool=resolved_tool.tool.name,
            capability=resolved_tool.tool.capability,
            domain=resolved_tool.domain,
            service=resolved_tool.service,
            entity=resolved_tool.tool.entity,
            operation=resolved_tool.tool.operation.value,
            parameters=dict(step.parameters),
        )

    def _resolve_tool(
        self,
        *,
        tool_name: str | None,
        domain: str | None,
        service: str | None,
        entity: str | None,
        operation: str | None,
    ) -> ResolvedTool:
        if tool_name:
            try:
                return self._tool_registry_service.find_tool_by_name(tool_name)
            except ToolNotFoundException:
                logger.warning('rbac_tool_name_lookup_failed tool=%s', tool_name)
        try:
            return self._tool_registry_service.find_tool(domain or '', service or '', entity or '', operation or '')
        except (ToolNotFoundException, DuplicateToolException) as exc:
            raise RBACClientError(str(exc)) from exc
