"""Regression tests for the Vidhya Attendance, Fee, and Profile registries."""

from pathlib import Path

import pytest

from app.tool_registry.loader import ToolRegistryLoader


REGISTRY_ROOT = Path(__file__).resolve().parents[1] / 'tool-registry'
SERVICES = ('attendance', 'fee', 'profile')


@pytest.fixture(scope='module')
def registries():
    loader = ToolRegistryLoader()
    return {
        service: loader.load_registry_file(REGISTRY_ROOT / 'vidhya' / f'{service}.yaml')
        for service in SERVICES
    }


def test_vidhya_service_registries_have_complete_unique_targets(registries):
    for service, registry in registries.items():
        assert registry.domain == 'vidhya'
        assert registry.service == service
        assert registry.server == 'vidhya-mcp'
        assert registry.adapter == service

        targets = [(tool.entity, tool.operation.value) for tool in registry.tools]
        assert len(targets) == len(set(targets))

        for tool in registry.tools:
            assert tool.name.startswith(f'{service}.')
            assert tool.id.startswith(f'vidhya.{service}.')
            assert tool.capability.startswith(f'{service}.')
            assert tool.status.value == 'active'


def test_vidhya_service_registry_parameter_metadata_matches_input_schema(registries):
    for registry in registries.values():
        for tool in registry.tools:
            declared_parameters = set(tool.required_parameters + tool.optional_parameters)
            schema_parameters = set(tool.input.schema_summary) - {'context'}

            assert declared_parameters == schema_parameters
            assert set(tool.required_parameters).isdisjoint(tool.optional_parameters)
