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
        requires_tool: bool | None,
        raw_response: str | None,
        adapter: str | None,
        model: str | None,
    ) -> PlannerOutput:
        normalized_parameters = parameters or {}
        normalized_requires_tool = requires_tool if requires_tool is not None else bool(tool)
        return PlannerOutput(
            intent=intent or '',
            requires_tool=normalized_requires_tool,
            domain=domain,
            service=service,
            entity=entity,
            operation=operation,
            parameters=normalized_parameters,
            tool=tool,
            rationale=None,
            raw_response=raw_response,
            adapter=adapter,
            model=model,
        )
