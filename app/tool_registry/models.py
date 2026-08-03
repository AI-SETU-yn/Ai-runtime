"""Pydantic models for registry YAML files."""

from __future__ import annotations

from enum import StrEnum

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator


class Operation(StrEnum):
    """Supported planner operations for tool lookup."""

    CREATE = 'create'
    READ = 'read'
    UPDATE = 'update'
    DELETE = 'delete'
    LIST = 'list'
    ASSIGN = 'assign'
    UNASSIGN = 'unassign'


class ResponseType(StrEnum):
    """Supported response types returned by tool executions."""

    STRUCTURED = 'structured'
    TEXT = 'text'
    APPROVAL = 'approval'
    DOCUMENT = 'document'


class ToolStatus(StrEnum):
    """Lifecycle status for a registry tool definition."""

    ACTIVE = 'active'
    INACTIVE = 'inactive'
    DEPRECATED = 'deprecated'


class InputDefinition(BaseModel):
    """Describes the input section for a tool definition.

    Attributes:
        schema_summary: Mapping of planner-facing input fields to their declared types.
    """

    model_config = ConfigDict(extra='forbid', frozen=True)

    schema_summary: dict[str, str] = Field(default_factory=dict)


class OutputDefinition(BaseModel):
    """Describes the output section for a tool definition.

    Attributes:
        type: Declared output type for the tool.
        identifier_fields: Output fields that uniquely identify a record.
        display_fields: Output fields that should be used when showing choices.
        display: Optional display metadata for prompt/rendering consumers.
    """

    model_config = ConfigDict(extra='forbid', frozen=True)

    type: str
    identifier_fields: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices('identifier_fields', 'identifierFields'),
    )
    display_fields: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices('display_fields', 'displayFields'),
    )
    display: dict[str, object] = Field(default_factory=dict)


class ToolReferenceDefinition(BaseModel):
    """Registry reference to another tool that can provide dependency data."""

    model_config = ConfigDict(extra='forbid', frozen=True, populate_by_name=True)

    domain: str | None = None
    service: str | None = None
    entity: str | None = None
    operation: str | None = None
    tool: str | None = None


class ParameterDependencyDefinition(BaseModel):
    """Declarative binding from a required parameter to another tool's output."""

    model_config = ConfigDict(extra='forbid', frozen=True, populate_by_name=True)

    parameter: str
    source: ToolReferenceDefinition
    path: str = Field(validation_alias=AliasChoices('path', 'binding_path', 'bindingPath'))

    @model_validator(mode='after')
    def validate_dependency(self) -> 'ParameterDependencyDefinition':
        if not self.parameter.strip():
            raise ValueError('parameter must not be empty')
        if not self.path.strip():
            raise ValueError('path must not be empty')
        if not any((self.source.tool, self.source.entity, self.source.operation)):
            raise ValueError('source must include tool or entity/operation')
        return self


class ToolDefinition(BaseModel):
    """Represents a single executable tool entry from the registry.

    Attributes:
        id: Unique stable tool identifier.
        name: Human-readable registry name.
        entity: Business entity addressed by the tool.
        operation: Exact operation used for planner lookup.
        description: Business description of the tool.
        capability: Internal capability label from the registry.
        required_parameters: Required planner-facing parameters.
        optional_parameters: Optional planner-facing parameters.
        response_type: Declared response category.
        version: Registry version for this tool definition.
        status: Lifecycle status.
        input: Input definition metadata.
        output: Output definition metadata.
    """

    model_config = ConfigDict(extra='forbid', frozen=True)

    id: str
    name: str
    entity: str
    operation: Operation
    description: str
    capability: str
    required_parameters: list[str] = Field(default_factory=list)
    optional_parameters: list[str] = Field(default_factory=list)
    parameter_dependencies: list[ParameterDependencyDefinition] = Field(default_factory=list)
    response_type: ResponseType
    adapter: str | None = None
    model: str | None = None
    version: str
    status: ToolStatus
    input: InputDefinition
    output: OutputDefinition

    @model_validator(mode='after')
    def validate_non_empty_fields(self) -> 'ToolDefinition':
        """Ensure required string fields are populated with non-blank values."""
        for field_name in ('id', 'name', 'entity', 'description', 'capability', 'version'):
            value = getattr(self, field_name)
            if not value or not value.strip():
                raise ValueError(f'{field_name} must not be empty')
        return self


class ToolRegistry(BaseModel):
    """Represents a registry YAML document.

    Attributes:
        domain: Business domain for all tools in the document.
        service: Service name for all tools in the document.
        server: MCP server identifier.
        protocol: Declared protocol.
        transport: Declared transport.
        tools: Tool definitions contained in the registry.
    """

    model_config = ConfigDict(extra='forbid', frozen=True)

    domain: str
    service: str
    server: str
    protocol: str
    transport: str
    adapter: str | None = None
    model: str | None = None
    tools: list[ToolDefinition] = Field(default_factory=list)

    @model_validator(mode='after')
    def validate_registry(self) -> 'ToolRegistry':
        """Ensure top-level registry metadata and tool list are valid."""
        for field_name in ('domain', 'service', 'server', 'protocol', 'transport'):
            value = getattr(self, field_name)
            if not value or not value.strip():
                raise ValueError(f'{field_name} must not be empty')
        if not self.tools:
            raise ValueError('tools must not be empty')
        return self


class ResolvedTool(BaseModel):
    """Resolved immutable tool record returned from repository lookups.

    This model combines registry-level metadata with the specific tool definition
    so downstream execution layers do not need a second registry lookup.

    Attributes:
        domain: Business domain of the registry file.
        service: Service name of the registry file.
        server: MCP server identifier.
        protocol: Declared protocol.
        transport: Declared transport.
        tool: Immutable tool definition.
    """

    model_config = ConfigDict(extra='forbid', frozen=True)

    domain: str
    service: str
    server: str
    protocol: str
    transport: str
    adapter: str | None = None
    model: str | None = None
    tool: ToolDefinition
