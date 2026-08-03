from app.prompts.templates import PLANNER_PROMPT


class PromptBuilder:
    def build_planner_prompt(
        self,
        message: str,
        *,
        registry_context: str | None = None,
        base_prompt: str = PLANNER_PROMPT,
    ) -> str:
        registry_section = ''
        if registry_context:
            registry_section = (
                '\n\nRegistered tool targets. If requires_tool is true, choose only one of these exact '
                f'domain/service/entity/operation combinations and derive intent as service.entity.operation:\n'
                f'{registry_context}\n\n'
                'If one user request needs multiple tools, return execution_plan as an ordered list of steps. '
                'If the user asks for multiple independent datasets, include one step per requested dataset even when no step depends on another. '
                'Each step must include step_id, domain, service, entity, operation, intent, parameters, '
                'depends_on, and optional parameter_bindings. Use parameter_bindings when a parameter must be '
                'read from a previous step, for example: '
                '{"targetParameter":{"from_step":"step_1","path":"$.data[0].declaredOutputField"}}. '
                'The path must point to factual data returned by a previous step.'
            )
        return f"{base_prompt}{registry_section}\n\nUser message:\n{message}"
