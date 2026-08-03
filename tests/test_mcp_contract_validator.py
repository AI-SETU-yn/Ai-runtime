import asyncio

import pytest

from app.mcp_client.discovery.tool_contract_validator import ToolContractValidationError, ToolContractValidator


class Tool:
    def __init__(self, name, required=None, optional=None):
        self.name = name
        self.required_parameters = required or []
        self.optional_parameters = optional or []


class Registry:
    def __init__(self):
        self.server = 'server-alpha'
        self.tools = [Tool('tool.one', required=['record_id'], optional=['include_archived'])]


class Repository:
    registries = [Registry()]


class RegistryService:
    repository = Repository()


class MCPClient:
    def __init__(self, tools):
        self.tools = tools

    async def list_tools(self, server):
        assert server == 'server-alpha'
        return self.tools


def test_mcp_contract_validator_accepts_matching_registry_and_mcp_tools() -> None:
    validator = ToolContractValidator(
        RegistryService(),
        MCPClient(
            [
                {
                    'name': 'tool.one',
                    'inputSchema': {
                        'properties': {'record_id': {}, 'include_archived': {}, 'context': {}},
                        'required': ['record_id', 'context'],
                    },
                }
            ]
        ),
    )

    assert asyncio.run(validator.validate()) == []


def test_mcp_contract_validator_reports_tool_and_parameter_drift() -> None:
    validator = ToolContractValidator(
        RegistryService(),
        MCPClient(
            [
                {
                    'name': 'tool.renamed',
                    'inputSchema': {'properties': {'context': {}}, 'required': ['context']},
                }
            ]
        ),
    )

    mismatches = asyncio.run(validator.validate())

    assert {mismatch.issue for mismatch in mismatches} == {'missing_mcp_tool', 'unexpected_mcp_tool'}


def test_mcp_contract_validator_reports_signature_mismatch() -> None:
    validator = ToolContractValidator(
        RegistryService(),
        MCPClient(
            [
                {
                    'name': 'tool.one',
                    'inputSchema': {
                        'properties': {'record_id': {}, 'context': {}},
                        'required': ['context'],
                    },
                }
            ]
        ),
    )

    mismatches = asyncio.run(validator.validate())

    assert {mismatch.issue for mismatch in mismatches} == {
        'parameter_missing_in_mcp_schema',
        'required_parameter_not_marked_required_in_mcp_schema',
    }


def test_mcp_contract_validator_raises_for_mismatches() -> None:
    validator = ToolContractValidator(RegistryService(), MCPClient([]))

    with pytest.raises(ToolContractValidationError):
        asyncio.run(validator.validate_or_raise_async())
