from app.model_gateway.adapter_resolver import ModelGatewayAdapterResolver
from app.models.planner import PlannerOutput


class Registry:
    adapter = 'service_alpha'
    tools = []


class Repository:
    registries = [Registry()]


class RegistryService:
    repository = Repository()

    def find_tool(self, domain, service, entity, operation):
        return type('Resolved', (), {'adapter': 'tool_adapter'})()


def test_planning_adapter_is_blank_without_unique_registry_adapter() -> None:
    resolver = ModelGatewayAdapterResolver(None)

    assert resolver.resolve_for_planning() == ''


def test_planning_adapter_uses_single_registry_adapter() -> None:
    resolver = ModelGatewayAdapterResolver(RegistryService())

    assert resolver.resolve_for_planning() == 'service_alpha'


def test_general_and_current_info_responses_do_not_use_enterprise_adapter() -> None:
    resolver = ModelGatewayAdapterResolver(RegistryService())
    planner_output = PlannerOutput(intent='general.chat', requires_tool=False, adapter='service_alpha')

    assert resolver.resolve_for_response(planner_output, 'general') == ''
    assert resolver.resolve_for_response(planner_output, 'current_info') == ''


def test_enterprise_response_adapter_comes_from_resolved_tool() -> None:
    resolver = ModelGatewayAdapterResolver(RegistryService())
    planner_output = PlannerOutput(
        intent='service_alpha.record.list',
        requires_tool=True,
        domain='example',
        service='service_alpha',
        entity='record',
        operation='list',
        adapter='legacy_adapter',
    )

    assert resolver.resolve_for_response(planner_output, 'enterprise') == 'tool_adapter'
