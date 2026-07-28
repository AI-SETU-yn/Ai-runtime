from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from app.models.planner import ExecutionPlanStep, PlannerOutput
from app.tool_registry.exceptions import DuplicateToolException, ToolNotFoundException
from app.tool_registry.models import ParameterDependencyDefinition, ResolvedTool, ToolDefinition
from app.tool_registry.service import ToolRegistryService

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PlannerRegistryTarget:
    domain: str
    service: str
    entity: str
    operation: str
    tool: ToolDefinition


@dataclass(frozen=True)
class PlannerValidationResult:
    output: PlannerOutput
    resolved_tool: ResolvedTool | None
    normalized: bool
    reasons: tuple[str, ...]
    failure_reason: str | None = None


class PlannerRegistryValidator:
    def __init__(self, tool_registry_service: ToolRegistryService) -> None:
        self._tool_registry_service = tool_registry_service

    def registry_prompt_context(self) -> str:
        rows = [
            (
                f"- domain={target.domain} service={target.service} entity={target.entity} "
                f"operation={target.operation} intent={self._intent_for(target)} tool={target.tool.name} "
                f"required_parameters={target.tool.required_parameters} optional_parameters={target.tool.optional_parameters}"
                f"{self._dependency_prompt_suffix(target)}"
            )
            for target in self._targets()
        ]
        return '\n'.join(rows)

    def normalize_and_validate(self, output: PlannerOutput) -> PlannerValidationResult:
        if not output.requires_tool:
            return PlannerValidationResult(output=output, resolved_tool=None, normalized=False, reasons=())

        if output.execution_plan:
            return self._normalize_and_validate_plan(output)

        return self._normalize_and_validate_single(output)

    def _normalize_and_validate_plan(self, output: PlannerOutput) -> PlannerValidationResult:
        original = output.model_dump()
        normalized_steps: list[ExecutionPlanStep] = []
        reasons: list[str] = []
        resolved_tool: ResolvedTool | None = None

        for index, step in enumerate(output.execution_plan, start=1):
            step_output = PlannerOutput(
                intent=step.intent or '',
                requires_tool=True,
                domain=step.domain or output.domain,
                service=step.service or output.service,
                entity=step.entity,
                operation=step.operation,
                tool=step.tool,
                parameters=step.parameters,
                adapter=output.adapter,
                model=output.model,
            )
            validation = self._normalize_and_validate_single(step_output, allow_dependency_expansion=False)
            if validation.failure_reason:
                return PlannerValidationResult(
                    output=output,
                    resolved_tool=None,
                    normalized=False,
                    reasons=tuple(reasons),
                    failure_reason=f'execution_plan step {index} failed validation: {validation.failure_reason}',
                )
            normalized = validation.output
            step_id = step.step_id or f'step_{index}'
            normalized_steps.append(
                ExecutionPlanStep(
                    step_id=step_id,
                    intent=normalized.intent,
                    domain=normalized.domain,
                    service=normalized.service,
                    entity=normalized.entity,
                    operation=normalized.operation,
                    tool=step.tool,
                    parameters=normalized.parameters,
                    parameter_bindings=step.parameter_bindings,
                    depends_on=step.depends_on,
                )
            )
            reasons.extend(f'step_{index}_{reason}' for reason in validation.reasons)
            resolved_tool = validation.resolved_tool

        final_step = normalized_steps[-1]
        normalized_output = output.model_copy(
            update={
                'intent': final_step.intent or output.intent,
                'domain': final_step.domain,
                'service': final_step.service,
                'entity': final_step.entity,
                'operation': final_step.operation,
                'parameters': final_step.parameters,
                'execution_plan': normalized_steps,
            }
        )
        return PlannerValidationResult(
            output=normalized_output,
            resolved_tool=resolved_tool,
            normalized=normalized_output.model_dump() != original,
            reasons=tuple(dict.fromkeys(reasons)),
        )

    def _normalize_and_validate_single(
        self,
        output: PlannerOutput,
        *,
        allow_dependency_expansion: bool = True,
    ) -> PlannerValidationResult:
        original = output.model_dump()
        reasons: list[str] = []
        domain = self._clean(output.domain)
        service = self._clean(output.service)
        entity = self._clean(output.entity)
        operation = self._clean(output.operation)

        if not any((domain, service, entity, operation, output.intent, output.tool)):
            return PlannerValidationResult(
                output=output,
                resolved_tool=None,
                normalized=False,
                reasons=(),
                failure_reason='Planner output did not include any registry target fields to validate.',
            )

        intent_target = self._target_from_intent(output.intent)
        if intent_target is not None:
            if not service:
                service = intent_target.service
                reasons.append('service_derived_from_intent')
            if not entity:
                entity = intent_target.entity
                reasons.append('entity_derived_from_intent')
            if not operation:
                operation = intent_target.operation
                reasons.append('operation_derived_from_intent')

        tool_target = self._target_from_tool(output.tool)
        if tool_target is not None:
            if not domain:
                domain = tool_target.domain
                reasons.append('domain_derived_from_tool')
            if not service:
                service = tool_target.service
                reasons.append('service_derived_from_tool')
            if not entity:
                entity = tool_target.entity
                reasons.append('entity_derived_from_tool')
            if not operation:
                operation = tool_target.operation
                reasons.append('operation_derived_from_tool')

        if not service and output.adapter:
            service = self._clean(output.adapter)
            reasons.append('service_derived_from_adapter')

        target_result = self._select_target(domain, service, entity, operation)

        if target_result is None and not output.service and output.adapter:
            target_result = self._select_target(domain, None, entity, operation)
            if target_result is not None:
                reasons.append('adapter_service_not_registered_service_fell_back_to_registry_match')

        if target_result is None and not output.service:
            target_result = self._select_target(domain, None, entity, operation)
            if target_result is not None:
                reasons.append('service_derived_from_unique_registry_match')

        if target_result is None:
            failure_reason = self._failure_reason(domain, service, entity, operation)
            return PlannerValidationResult(
                output=output,
                resolved_tool=None,
                normalized=False,
                reasons=tuple(reasons),
                failure_reason=failure_reason,
            )

        target, match_reasons = target_result
        reasons.extend(match_reasons)

        normalized_intent = self._intent_for(target)
        if output.intent != normalized_intent:
            reasons.append('intent_derived_from_registry_target' if not output.intent else 'intent_normalized_to_registry_target')

        normalized_output = output.model_copy(
            update={
                'intent': normalized_intent,
                'domain': target.domain,
                'service': target.service,
                'entity': target.entity,
                'operation': target.operation,
            }
        )

        try:
            resolved_tool = self._tool_registry_service.find_tool(
                target.domain,
                target.service,
                target.entity,
                target.operation,
            )
        except ToolNotFoundException as exc:
            return PlannerValidationResult(
                output=normalized_output,
                resolved_tool=None,
                normalized=normalized_output.model_dump() != original,
                reasons=tuple(reasons),
                failure_reason=str(exc),
            )
        except DuplicateToolException as exc:
            return PlannerValidationResult(
                output=normalized_output,
                resolved_tool=None,
                normalized=normalized_output.model_dump() != original,
                reasons=tuple(reasons),
                failure_reason=str(exc),
            )

        if allow_dependency_expansion:
            expanded = self._expand_missing_parameter_dependencies(normalized_output, target, resolved_tool)
            if expanded is not None:
                expanded_output, expansion_reasons = expanded
                plan_validation = self._normalize_and_validate_plan(expanded_output)
                if plan_validation.failure_reason:
                    return plan_validation
                return PlannerValidationResult(
                    output=plan_validation.output,
                    resolved_tool=plan_validation.resolved_tool,
                    normalized=True,
                    reasons=tuple(dict.fromkeys([*reasons, *expansion_reasons, *plan_validation.reasons])),
                )

        return PlannerValidationResult(
            output=normalized_output,
            resolved_tool=resolved_tool,
            normalized=normalized_output.model_dump() != original,
            reasons=tuple(dict.fromkeys(reasons)),
        )

    def _expand_missing_parameter_dependencies(
        self,
        output: PlannerOutput,
        target: PlannerRegistryTarget,
        resolved_tool: ResolvedTool,
    ) -> tuple[PlannerOutput, tuple[str, ...]] | None:
        missing_parameters = self._missing_required_parameters(resolved_tool.tool, output.parameters)
        if not missing_parameters:
            return None

        dependencies_by_parameter = {
            dependency.parameter: dependency
            for dependency in resolved_tool.tool.parameter_dependencies
        }
        if any(parameter not in dependencies_by_parameter for parameter in missing_parameters):
            return None

        source_steps: list[ExecutionPlanStep] = []
        source_step_ids: dict[tuple[str, str, str, str], str] = {}
        parameter_bindings: dict[str, dict[str, str]] = {}
        depends_on: list[str] = []
        reasons: list[str] = []

        for parameter in missing_parameters:
            dependency = dependencies_by_parameter[parameter]
            source_target = self._target_from_dependency(dependency, fallback_target=target)
            if source_target is None:
                return None

            source_key = (
                source_target.domain,
                source_target.service,
                source_target.entity,
                source_target.operation,
            )
            step_id = source_step_ids.get(source_key)
            if step_id is None:
                step_id = f'step_{len(source_steps) + 1}'
                source_step_ids[source_key] = step_id
                source_steps.append(
                    ExecutionPlanStep(
                        step_id=step_id,
                        intent=self._intent_for(source_target),
                        domain=source_target.domain,
                        service=source_target.service,
                        entity=source_target.entity,
                        operation=source_target.operation,
                        parameters={},
                    )
                )
            parameter_bindings[parameter] = {
                'from_step': step_id,
                'path': dependency.path,
            }
            depends_on.append(step_id)
            reasons.append(f'parameter_dependency_expanded:{parameter}')

        target_step_id = f'step_{len(source_steps) + 1}'
        target_step = ExecutionPlanStep(
            step_id=target_step_id,
            intent=self._intent_for(target),
            domain=target.domain,
            service=target.service,
            entity=target.entity,
            operation=target.operation,
            parameters=output.parameters,
            parameter_bindings=parameter_bindings,
            depends_on=list(dict.fromkeys(depends_on)),
        )
        expanded_output = output.model_copy(
            update={
                'execution_plan': [*source_steps, target_step],
            }
        )
        return expanded_output, tuple(reasons)

    @staticmethod
    def _missing_required_parameters(tool: ToolDefinition, parameters: dict[str, object]) -> list[str]:
        return [
            parameter
            for parameter in tool.required_parameters
            if parameter not in parameters or parameters[parameter] in (None, '')
        ]

    def _target_from_dependency(
        self,
        dependency: ParameterDependencyDefinition,
        *,
        fallback_target: PlannerRegistryTarget,
    ) -> PlannerRegistryTarget | None:
        if dependency.source.tool:
            return self._target_from_tool(dependency.source.tool)
        target_result = self._select_target(
            dependency.source.domain or fallback_target.domain,
            dependency.source.service or fallback_target.service,
            dependency.source.entity,
            dependency.source.operation,
        )
        return target_result[0] if target_result else None

    def _select_target(
        self,
        domain: str | None,
        service: str | None,
        entity: str | None,
        operation: str | None,
    ) -> tuple[PlannerRegistryTarget, tuple[str, ...]] | None:
        scored: list[tuple[int, PlannerRegistryTarget, tuple[str, ...]]] = []
        for target in self._targets():
            match = self._score_target(target, domain, service, entity, operation)
            if match is not None:
                score, reasons = match
                scored.append((score, target, tuple(reasons)))

        if not scored:
            return None

        scored.sort(key=lambda item: item[0], reverse=True)
        best_score = scored[0][0]
        best = [item for item in scored if item[0] == best_score]
        best_keys = {
            (item[1].domain, item[1].service, item[1].entity, item[1].operation)
            for item in best
        }
        if len(best_keys) > 1:
            return None
        return best[0][1], best[0][2]

    def _score_target(
        self,
        target: PlannerRegistryTarget,
        domain: str | None,
        service: str | None,
        entity: str | None,
        operation: str | None,
    ) -> tuple[int, list[str]] | None:
        score = 0
        reasons: list[str] = []
        for field_name, requested, canonical, weight in (
            ('domain', domain, target.domain, 1),
            ('service', service, target.service, 1),
            ('entity', entity, target.entity, 3),
            ('operation', operation, target.operation, 2),
        ):
            field_match = self._score_field(requested, canonical)
            if field_match is None:
                return None
            field_score, reason = field_match
            score += field_score * weight
            if reason:
                reasons.append(f'{field_name}_{reason}')
        return score, reasons

    def _score_field(self, requested: str | None, canonical: str) -> tuple[int, str | None] | None:
        if not requested:
            return 0, None
        requested_key = self._normalize_key(requested)
        canonical_key = self._normalize_key(canonical)
        if requested_key == canonical_key:
            return 100, None

        requested_tokens = self._tokens(requested)
        canonical_tokens = self._tokens(canonical)
        if requested_tokens and requested_tokens.issubset(canonical_tokens):
            return 70, 'normalized_by_token_alias'
        if canonical_tokens and canonical_tokens.issubset(requested_tokens):
            return 60, 'normalized_from_verbose_alias'
        return None

    def _failure_reason(
        self,
        domain: str | None,
        service: str | None,
        entity: str | None,
        operation: str | None,
    ) -> str:
        return (
            'Planner output does not match any unique registered tool target '
            f'domain={domain!r} service={service!r} entity={entity!r} operation={operation!r}'
        )

    def _targets(self) -> tuple[PlannerRegistryTarget, ...]:
        targets: list[PlannerRegistryTarget] = []
        for registry in self._tool_registry_service.repository.registries:
            for tool in registry.tools:
                targets.append(
                    PlannerRegistryTarget(
                        domain=registry.domain,
                        service=registry.service,
                        entity=tool.entity,
                        operation=tool.operation.value,
                        tool=tool,
                    )
                )
        return tuple(targets)

    def _target_from_intent(self, intent: str | None) -> PlannerRegistryTarget | None:
        if not intent:
            return None
        parts = [part for part in intent.split('.') if part]
        if len(parts) != 3:
            return None
        service, entity, operation = parts
        target_result = self._select_target(None, service, entity, operation)
        return target_result[0] if target_result else None

    def _target_from_tool(self, tool_name: str | None) -> PlannerRegistryTarget | None:
        if not tool_name:
            return None
        requested = self._normalize_key(tool_name)
        matches = [
            target
            for target in self._targets()
            if requested in {self._normalize_key(target.tool.name), self._normalize_key(target.tool.id)}
        ]
        return matches[0] if len(matches) == 1 else None

    def _dependency_prompt_suffix(self, target: PlannerRegistryTarget) -> str:
        if not target.tool.parameter_dependencies:
            return ''
        dependency_parts = []
        for dependency in target.tool.parameter_dependencies:
            source_target = self._target_from_dependency(dependency, fallback_target=target)
            if source_target is None:
                source = dependency.source.tool or 'unresolved'
            else:
                source = self._intent_for(source_target)
            dependency_parts.append(f"{dependency.parameter}<-{source}:{dependency.path}")
        return f" parameter_dependencies={dependency_parts}"

    @staticmethod
    def _intent_for(target: PlannerRegistryTarget) -> str:
        return f'{target.service}.{target.entity}.{target.operation}'

    @staticmethod
    def _clean(value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @classmethod
    def _normalize_key(cls, value: str) -> str:
        return '_'.join(cls._tokens(value))

    @staticmethod
    def _tokens(value: str) -> set[str]:
        normalized = re.sub(r'[^a-zA-Z0-9]+', ' ', value).lower()
        tokens = set()
        for token in normalized.split():
            if not token:
                continue
            tokens.add(token)
            if token.endswith('s') and len(token) > 3:
                tokens.add(token[:-1])
        return tokens
