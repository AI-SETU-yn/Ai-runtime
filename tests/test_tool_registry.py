from pathlib import Path

import pytest
from pydantic import ValidationError

from app.tool_registry.exceptions import (
    DuplicateToolException,
    InvalidToolRegistryException,
    ToolNotFoundException,
)
from app.tool_registry.loader import ToolRegistryLoader
from app.tool_registry.models import Operation, ResponseType, ToolStatus
from app.tool_registry.repository import ToolRegistryRepository
from app.tool_registry.service import ToolRegistryService

VALID_REGISTRY = """\
domain: vidhya
service: academic
server: vidhya-mcp
protocol: mcp
transport: streamable-http
tools:
  - id: vidhya.academic.subject.list.get_all_subjects
    name: academic.get_all_subjects
    entity: subject
    operation: list
    description: Fetches all subjects.
    capability: academic.subject.list
    required_parameters: []
    optional_parameters: []
    response_type: structured
    version: 1.0.0
    status: active
    input:
      schema_summary:
        context: VidhyaRequestContext
    output:
      type: json
"""


def write_registry(root: Path, relative_path: str, content: str) -> Path:
    file_path = root / relative_path
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding='utf-8')
    return file_path


def test_loader_loads_registry(tmp_path: Path):
    registry_file = write_registry(tmp_path, 'vidhya/academic.yaml', VALID_REGISTRY)

    registry = ToolRegistryLoader().load_registry_file(registry_file)

    assert registry.domain == 'vidhya'
    assert registry.service == 'academic'
    assert len(registry.tools) == 1
    assert registry.tools[0].entity == 'subject'
    assert registry.tools[0].operation is Operation.LIST
    assert registry.tools[0].response_type is ResponseType.STRUCTURED
    assert registry.tools[0].status is ToolStatus.ACTIVE


def test_loader_accepts_output_display_metadata(tmp_path: Path):
    registry_file = write_registry(
        tmp_path,
        'vidhya/academic.yaml',
        VALID_REGISTRY.replace(
            'output:\n      type: json',
            (
                'output:\n'
                '      type: json\n'
                '      identifier_fields:\n'
                '      - referenceId\n'
                '      display_fields:\n'
                '      - subjectName\n'
                '      display:\n'
                '        label_template: "{subjectName}"'
            ),
        ),
    )

    registry = ToolRegistryLoader().load_registry_file(registry_file)

    assert registry.tools[0].output.identifier_fields == ['referenceId']
    assert registry.tools[0].output.display_fields == ['subjectName']
    assert registry.tools[0].output.display == {'label_template': '{subjectName}'}


def test_repository_allows_ambiguous_lookup_keys_but_tracks_them(tmp_path: Path):
    write_registry(tmp_path, 'vidhya/academic.yaml', VALID_REGISTRY)
    write_registry(
        tmp_path,
        'vidhya/academic-extra.yaml',
        VALID_REGISTRY.replace(
            'vidhya.academic.subject.list.get_all_subjects',
            'vidhya.academic.subject.list.get_subjects_duplicate',
        ).replace(
            'academic.get_all_subjects',
            'academic.get_subjects_duplicate',
        ),
    )

    repository = ToolRegistryRepository()
    repository.load_from_directory(tmp_path)

    assert len(repository.ambiguous_keys) == 1
    ambiguous_tools = next(iter(repository.ambiguous_keys.values()))
    assert {tool.tool.name for tool in ambiguous_tools} == {
        'academic.get_all_subjects',
        'academic.get_subjects_duplicate',
    }


def test_repository_lookup_raises_for_ambiguous_key(tmp_path: Path):
    write_registry(tmp_path, 'vidhya/academic.yaml', VALID_REGISTRY)
    write_registry(
        tmp_path,
        'vidhya/academic-extra.yaml',
        VALID_REGISTRY.replace(
            'vidhya.academic.subject.list.get_all_subjects',
            'vidhya.academic.subject.list.get_subjects_duplicate',
        ).replace(
            'academic.get_all_subjects',
            'academic.get_subjects_duplicate',
        ),
    )

    repository = ToolRegistryRepository()
    repository.load_from_directory(tmp_path)

    with pytest.raises(DuplicateToolException, match='Multiple tools found for logical lookup key'):
        repository.find_tool('vidhya', 'academic', 'subject', 'list')


def test_repository_detects_duplicate_tool_id(tmp_path: Path):
    write_registry(tmp_path, 'vidhya/academic.yaml', VALID_REGISTRY)
    write_registry(
        tmp_path,
        'vidhya/academic-extra.yaml',
        VALID_REGISTRY.replace('entity: subject', 'entity: holiday').replace('operation: list', 'operation: read'),
    )

    repository = ToolRegistryRepository()

    with pytest.raises(DuplicateToolException, match='Duplicate tool id detected'):
        repository.load_from_directory(tmp_path)


def test_repository_detects_duplicate_tool_name(tmp_path: Path):
    write_registry(tmp_path, 'vidhya/academic.yaml', VALID_REGISTRY)
    write_registry(
        tmp_path,
        'vidhya/academic-extra.yaml',
        VALID_REGISTRY.replace(
            'vidhya.academic.subject.list.get_all_subjects',
            'vidhya.academic.holiday.read.get_holiday',
        ).replace('entity: subject', 'entity: holiday').replace('operation: list', 'operation: read'),
    )

    repository = ToolRegistryRepository()

    with pytest.raises(DuplicateToolException, match='Duplicate tool name detected'):
        repository.load_from_directory(tmp_path)


def test_loader_rejects_invalid_yaml(tmp_path: Path):
    invalid_file = write_registry(
        tmp_path,
        'vidhya/broken.yaml',
        'domain: vidhya\ntools:\n  - id: only-id-with-missing-fields\n',
    )

    with pytest.raises(InvalidToolRegistryException):
        ToolRegistryLoader().load_registry_file(invalid_file)


def test_loader_rejects_empty_registry(tmp_path: Path):
    registry_file = write_registry(
        tmp_path,
        'vidhya/empty.yaml',
        'domain: vidhya\nservice: academic\nserver: vidhya-mcp\nprotocol: mcp\ntransport: streamable-http\ntools: []\n',
    )

    with pytest.raises(InvalidToolRegistryException):
        ToolRegistryLoader().load_registry_file(registry_file)


def test_service_lookup_returns_resolved_tool(tmp_path: Path):
    write_registry(tmp_path, 'vidhya/academic.yaml', VALID_REGISTRY)
    service = ToolRegistryService()
    service.initialize(tmp_path)

    resolved_tool = service.find_tool('vidhya', 'academic', 'subject', 'list')

    assert resolved_tool.domain == 'vidhya'
    assert resolved_tool.service == 'academic'
    assert resolved_tool.server == 'vidhya-mcp'
    assert resolved_tool.tool.name == 'academic.get_all_subjects'
    assert resolved_tool.tool.capability == 'academic.subject.list'


def test_service_lookup_raises_when_not_found(tmp_path: Path):
    write_registry(tmp_path, 'vidhya/academic.yaml', VALID_REGISTRY)
    service = ToolRegistryService()
    service.initialize(tmp_path)

    with pytest.raises(ToolNotFoundException):
        service.find_tool('vidhya', 'academic', 'holiday', 'list')


def test_tool_definition_is_immutable(tmp_path: Path):
    registry_file = write_registry(tmp_path, 'vidhya/academic.yaml', VALID_REGISTRY)
    registry = ToolRegistryLoader().load_registry_file(registry_file)

    with pytest.raises(ValidationError):
        registry.tools[0].name = 'changed'


def test_resolved_tool_is_immutable(tmp_path: Path):
    write_registry(tmp_path, 'vidhya/academic.yaml', VALID_REGISTRY)
    service = ToolRegistryService()
    service.initialize(tmp_path)
    resolved_tool = service.find_tool('vidhya', 'academic', 'subject', 'list')

    with pytest.raises(ValidationError):
        resolved_tool.server = 'changed'
