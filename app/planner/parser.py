import json
from typing import Any

from app.models.planner import PlannerOutput


class PlannerOutputParser:
    def parse(
        self,
        *,
        intent: str | None,
        domain: str | None,
        service: str | None,
        entity: str | None,
        operation: str | None,
        tool: str | None,
        parameters: dict[str, object] | None,
        execution_plan: list[dict[str, object]] | None = None,
        tasks: list[dict[str, object]] | None = None,
        requires_tool: bool | None,
        raw_response: str | None,
        adapter: str | None,
        model: str | None,
    ) -> PlannerOutput:
        raw_payload = self._parse_raw_response(raw_response)
        normalized_parameters = parameters or self._dict_value(raw_payload.get('parameters')) or {}
        normalized_tasks = self._tasks_value(tasks or raw_payload.get('tasks'))
        normalized_requires_tool = requires_tool
        if normalized_requires_tool is None:
            normalized_requires_tool = self._bool_value(raw_payload.get('requiresTool'))
        if normalized_requires_tool is None:
            normalized_requires_tool = self._bool_value(raw_payload.get('requires_tool'))
        if normalized_requires_tool is None:
            normalized_requires_tool = bool(tool or raw_payload.get('tool'))
        normalized_execution_plan = self._execution_plan_value(
            execution_plan
            or raw_payload.get('execution_plan')
            or raw_payload.get('executionPlan')
            or raw_payload.get('steps')
        )
        if normalized_execution_plan and normalized_requires_tool is not True:
            normalized_requires_tool = True
        if normalized_requires_tool is None:
            normalized_requires_tool = False

        reconciled = self._reconcile_tasks_and_legacy_fields(
            tasks=normalized_tasks,
            execution_plan=normalized_execution_plan,
            requires_tool=normalized_requires_tool,
            domain=domain or self._str_value(raw_payload.get('domain')),
            service=service or self._str_value(raw_payload.get('service')),
            entity=entity or self._str_value(raw_payload.get('entity')),
            operation=operation or self._str_value(raw_payload.get('operation')),
            intent=intent or self._str_value(raw_payload.get('intent')) or '',
            parameters=normalized_parameters,
        )
        return PlannerOutput(
            tasks=reconciled['tasks'],
            intent=reconciled['intent'],
            requires_tool=reconciled['requires_tool'],
            domain=reconciled['domain'],
            service=reconciled['service'],
            entity=reconciled['entity'],
            operation=reconciled['operation'],
            parameters=reconciled['parameters'],
            tool=tool or self._str_value(raw_payload.get('tool')),
            rationale=None,
            raw_response=raw_response,
            adapter=adapter or self._str_value(raw_payload.get('adapter')),
            model=model or self._str_value(raw_payload.get('model')),
            execution_plan=reconciled['execution_plan'],
        )

    @classmethod
    def _reconcile_tasks_and_legacy_fields(
        cls,
        *,
        tasks: list[dict[str, object]],
        execution_plan: list[dict[str, object]],
        requires_tool: bool,
        domain: str | None,
        service: str | None,
        entity: str | None,
        operation: str | None,
        intent: str,
        parameters: dict[str, object],
    ) -> dict[str, object]:
        if tasks:
            return cls._legacy_fields_from_tasks(
                tasks=tasks,
                execution_plan=execution_plan,
                domain=domain,
                service=service,
                entity=entity,
                operation=operation,
                intent=intent,
                parameters=parameters,
            )
        return cls._tasks_from_legacy_fields(
            execution_plan=execution_plan,
            requires_tool=requires_tool,
            domain=domain,
            service=service,
            entity=entity,
            operation=operation,
            intent=intent,
            parameters=parameters,
        )

    @staticmethod
    def _legacy_fields_from_tasks(
        *,
        tasks: list[dict[str, object]],
        execution_plan: list[dict[str, object]],
        domain: str | None,
        service: str | None,
        entity: str | None,
        operation: str | None,
        intent: str,
        parameters: dict[str, object],
    ) -> dict[str, object]:
        first_task = tasks[0]
        enterprise_tasks = [task for task in tasks if str(task.get('type', '')).strip().casefold() == 'enterprise']

        if enterprise_tasks:
            first_enterprise = enterprise_tasks[0]
            domain = domain or first_enterprise.get('domain')
            service = service or first_enterprise.get('service')
            entity = entity or first_enterprise.get('entity')
            operation = operation or first_enterprise.get('operation')
            parameters = parameters or dict(first_enterprise.get('parameters') or {})
            if not intent and all((first_enterprise.get('service'), first_enterprise.get('entity'), first_enterprise.get('operation'))):
                intent = f"{first_enterprise['service']}.{first_enterprise['entity']}.{first_enterprise['operation']}"
            if not execution_plan:
                execution_plan = [
                    {
                        'step_id': f'step_{index}',
                        'intent': (
                            f"{task['service']}.{task['entity']}.{task['operation']}"
                            if all((task.get('service'), task.get('entity'), task.get('operation')))
                            else None
                        ),
                        'domain': task.get('domain'),
                        'service': task.get('service'),
                        'entity': task.get('entity'),
                        'operation': task.get('operation'),
                        'parameters': dict(task.get('parameters') or {}),
                        'visible_in_response': True,
                    }
                    for index, task in enumerate(enterprise_tasks, start=1)
                ]
            requires_tool = True
        else:
            requires_tool = False
            parameters = parameters or dict(first_task.get('parameters') or {})
            if not intent and str(first_task.get('type', '')).strip().casefold() == 'general':
                intent = 'general.chat'

        return {
            'tasks': tasks,
            'domain': domain,
            'service': service,
            'entity': entity,
            'operation': operation,
            'parameters': parameters,
            'intent': intent,
            'execution_plan': execution_plan,
            'requires_tool': requires_tool,
        }

    @staticmethod
    def _tasks_from_legacy_fields(
        *,
        execution_plan: list[dict[str, object]],
        requires_tool: bool,
        domain: str | None,
        service: str | None,
        entity: str | None,
        operation: str | None,
        intent: str,
        parameters: dict[str, object],
    ) -> dict[str, object]:
        if execution_plan:
            tasks = [
                {
                    'type': 'enterprise',
                    'domain': step.get('domain') or domain,
                    'service': step.get('service') or service,
                    'entity': step.get('entity'),
                    'operation': step.get('operation'),
                    'parameters': dict(step.get('parameters') or {}),
                    'requires_clarification': bool(step.get('question')),
                }
                for step in execution_plan
            ]
        elif requires_tool or any((domain, service, entity, operation)):
            tasks = [
                {
                    'type': 'enterprise',
                    'domain': domain,
                    'service': service,
                    'entity': entity,
                    'operation': operation,
                    'parameters': dict(parameters or {}),
                    'requires_clarification': False,
                }
            ]
        elif intent:
            tasks = [
                {
                    'type': 'general',
                    'parameters': dict(parameters or {}),
                    'requires_clarification': False,
                }
            ]
        else:
            tasks = []

        return {
            'tasks': tasks,
            'domain': domain,
            'service': service,
            'entity': entity,
            'operation': operation,
            'parameters': parameters,
            'intent': intent,
            'execution_plan': execution_plan,
            'requires_tool': requires_tool,
        }

    @classmethod
    def _parse_raw_response(cls, raw_response: str | None) -> dict[str, Any]:
        if not raw_response:
            return {}
        decoder = json.JSONDecoder()
        for index, char in enumerate(raw_response):
            if char != '{':
                continue
            try:
                payload, _ = decoder.raw_decode(raw_response[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload
        return {}

    @staticmethod
    def _str_value(value: Any) -> str | None:
        return value if isinstance(value, str) and value else None

    @staticmethod
    def _dict_value(value: Any) -> dict[str, object] | None:
        return value if isinstance(value, dict) else None

    @classmethod
    def _execution_plan_value(cls, value: Any) -> list[dict[str, object]]:
        if not isinstance(value, list):
            return []
        if not all(isinstance(item, dict) for item in value):
            return []
        return [cls._normalize_step(item) for item in value]

    @classmethod
    def _tasks_value(cls, value: Any) -> list[dict[str, object]]:
        if not isinstance(value, list):
            return []
        normalized: list[dict[str, object]] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            normalized.append(
                {
                    'type': cls._str_value(item.get('type')) or 'enterprise',
                    'domain': cls._str_value(item.get('domain')),
                    'service': cls._str_value(item.get('service')),
                    'entity': cls._str_value(item.get('entity')),
                    'operation': cls._str_value(item.get('operation')),
                    'parameters': cls._dict_value(item.get('parameters')) or {},
                    'requires_clarification': cls._bool_value(
                        item.get('requires_clarification')
                        if 'requires_clarification' in item
                        else item.get('requiresClarification')
                    )
                    or False,
                }
            )
        return normalized

    @classmethod
    def _normalize_step(cls, value: dict[str, object]) -> dict[str, object]:
        step = dict(value)
        tool_name = step.get('tool_name') or step.get('toolName')
        if tool_name and not step.get('tool'):
            step['tool'] = tool_name
        step['depends_on'] = cls._depends_on_value(step.get('depends_on') or step.get('dependsOn'))
        bindings = (
            step.get('parameter_bindings')
            or step.get('parameterBindings')
            or step.get('bindings')
            or {}
        )
        step['parameter_bindings'] = cls._parameter_bindings_value(bindings)
        return step

    @staticmethod
    def _depends_on_value(value: Any) -> list[str]:
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        if isinstance(value, list):
            return [item for item in value if isinstance(item, str) and item.strip()]
        return []

    @classmethod
    def _parameter_bindings_value(cls, value: Any) -> dict[str, object]:
        if isinstance(value, list):
            normalized: dict[str, object] = {}
            for item in value:
                if not isinstance(item, dict):
                    continue
                parameter = cls._binding_parameter_name(item)
                if parameter:
                    normalized[parameter] = cls._binding_source(item)
            return normalized
        if isinstance(value, dict):
            parameter = cls._binding_parameter_name(value)
            if parameter:
                return {parameter: cls._binding_source(value)}
            return value
        return {}

    @staticmethod
    def _binding_parameter_name(value: dict[str, object]) -> str | None:
        parameter = (
            value.get('parameter')
            or value.get('target_parameter')
            or value.get('targetParameter')
            or value.get('name')
        )
        return parameter if isinstance(parameter, str) and parameter.strip() else None

    @staticmethod
    def _binding_source(value: dict[str, object]) -> dict[str, object]:
        source: dict[str, object] = {}
        for output_key, candidates in {
            'from_step': ('from_step', 'fromStep', 'source_step', 'sourceStep', 'step'),
            'path': ('path', 'json_path', 'jsonPath', 'source_path', 'sourcePath'),
        }.items():
            for candidate in candidates:
                if candidate in value:
                    source[output_key] = value[candidate]
                    break
        return source

    @staticmethod
    def _bool_value(value: Any) -> bool | None:
        return value if isinstance(value, bool) else None
