"""In-memory repository for tool registry lookups."""

from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter
from typing import NamedTuple

from app.tool_registry.exceptions import DuplicateToolException, InvalidToolRegistryException, ToolNotFoundException
from app.tool_registry.loader import ToolRegistryLoader
from app.tool_registry.models import ResolvedTool, ToolDefinition, ToolRegistry

logger = logging.getLogger(__name__)


class ToolLookupKey(NamedTuple):
    """Stable exact-match lookup key used by the planner contract.

    Attributes:
        domain: Planner domain.
        service: Planner service.
        entity: Planner entity.
        operation: Planner operation.
    """

    domain: str
    service: str
    entity: str
    operation: str


class ToolRegistryRepository:
    """Loads registry files once and serves exact in-memory tool lookups.

    The repository owns the startup-time indexing process and guarantees that
    runtime lookups are fulfilled using constant-time dictionary access.
    """

    def __init__(self, loader: ToolRegistryLoader | None = None) -> None:
        self._loader = loader or ToolRegistryLoader()
        self._registries: list[ToolRegistry] = []
        self._index: dict[ToolLookupKey, tuple[ResolvedTool, ...]] = {}
        self._loaded = False

    @property
    def registries(self) -> tuple[ToolRegistry, ...]:
        """Return loaded immutable registry documents."""
        return tuple(self._registries)

    @property
    def loaded(self) -> bool:
        """Indicate whether the repository has already loaded registry files."""
        return self._loaded

    @property
    def ambiguous_keys(self) -> dict[ToolLookupKey, tuple[ResolvedTool, ...]]:
        """Return every logical lookup key that resolves to multiple tools."""
        return {key: tools for key, tools in self._index.items() if len(tools) > 1}

    def load_from_directory(self, root_path: Path) -> None:
        """Load and index all registry files from the given directory exactly once.

        Args:
            root_path: Root directory that contains registry YAML files.

        Raises:
            InvalidToolRegistryException: If no registry files are found.
            DuplicateToolException: If duplicate IDs or names exist.
        """
        if self._loaded:
            logger.info('Tool registry repository already initialized; skipping reload.')
            return

        start = perf_counter()
        registries = self._loader.load_registries(root_path)
        if not registries:
            raise InvalidToolRegistryException(f'No registry files found in: {root_path}')

        index: dict[ToolLookupKey, list[ResolvedTool]] = {}
        seen_ids: dict[str, str] = {}
        seen_names: dict[str, str] = {}
        tool_count = 0

        for registry in registries:
            for tool in registry.tools:
                tool_count += 1
                self._validate_unique_tool_id(tool, registry, seen_ids)
                self._validate_unique_tool_name(tool, registry, seen_names)

                key = ToolLookupKey(
                    domain=registry.domain,
                    service=registry.service,
                    entity=tool.entity,
                    operation=tool.operation.value,
                )
                resolved_tool = ResolvedTool(
                    domain=registry.domain,
                    service=registry.service,
                    server=registry.server,
                    protocol=registry.protocol,
                    transport=registry.transport,
                    adapter=tool.adapter or registry.adapter,
                    model=tool.model or registry.model,
                    tool=tool,
                )
                index.setdefault(key, []).append(resolved_tool)

        immutable_index = {key: tuple(tools) for key, tools in index.items()}
        self._registries = registries
        self._index = immutable_index
        self._loaded = True
        duration_ms = (perf_counter() - start) * 1000
        ambiguous_keys = self.ambiguous_keys
        logger.info(
            'Tool registry startup completed registry_count=%s tool_count=%s lookup_key_count=%s ambiguous_key_count=%s duration_ms=%.2f',
            len(self._registries),
            tool_count,
            len(self._index),
            len(ambiguous_keys),
            duration_ms,
        )
        for key, tools in ambiguous_keys.items():
            logger.warning(
                'Ambiguous logical lookup key loaded domain=%s service=%s entity=%s operation=%s tool_names=%s',
                key.domain,
                key.service,
                key.entity,
                key.operation,
                [tool.tool.name for tool in tools],
            )

    def find_tool(self, domain: str, service: str, entity: str, operation: str) -> ResolvedTool:
        """Return a resolved tool using exact dictionary lookup.

        Args:
            domain: Planner domain.
            service: Planner service.
            entity: Planner entity.
            operation: Planner operation.

        Returns:
            A resolved immutable tool record.

        Raises:
            ToolNotFoundException: If no exact lookup key exists.
            DuplicateToolException: If multiple tools share the same logical lookup key.
        """
        key = ToolLookupKey(domain=domain, service=service, entity=entity, operation=operation)
        tools = self._index.get(key)
        if tools is None:
            logger.warning(
                'Tool lookup failed domain=%s service=%s entity=%s operation=%s',
                domain,
                service,
                entity,
                operation,
            )
            raise ToolNotFoundException(
                f'No tool found for domain={domain}, service={service}, entity={entity}, operation={operation}'
            )
        if len(tools) > 1:
            logger.warning(
                'Tool lookup ambiguous domain=%s service=%s entity=%s operation=%s tool_names=%s',
                domain,
                service,
                entity,
                operation,
                [tool.tool.name for tool in tools],
            )
            raise DuplicateToolException(
                'Multiple tools found for logical lookup key '
                f'domain={domain}, service={service}, entity={entity}, operation={operation}'
            )
        tool = tools[0]
        logger.info(
            'Tool lookup succeeded domain=%s service=%s entity=%s operation=%s tool_id=%s tool_name=%s',
            domain,
            service,
            entity,
            operation,
            tool.tool.id,
            tool.tool.name,
        )
        return tool

    @staticmethod
    def _validate_unique_tool_id(
        tool: ToolDefinition,
        registry: ToolRegistry,
        seen_ids: dict[str, str],
    ) -> None:
        """Validate that each tool ID is globally unique across all registries."""
        existing_location = seen_ids.get(tool.id)
        current_location = f'{registry.domain}.{registry.service}.{tool.entity}.{tool.operation.value}'
        if existing_location is not None:
            logger.error('Duplicate tool id detected id=%s existing=%s current=%s', tool.id, existing_location, current_location)
            raise DuplicateToolException(f'Duplicate tool id detected: {tool.id}')
        seen_ids[tool.id] = current_location

    @staticmethod
    def _validate_unique_tool_name(
        tool: ToolDefinition,
        registry: ToolRegistry,
        seen_names: dict[str, str],
    ) -> None:
        """Validate that each tool name is globally unique across all registries."""
        existing_location = seen_names.get(tool.name)
        current_location = f'{registry.domain}.{registry.service}.{tool.entity}.{tool.operation.value}'
        if existing_location is not None:
            logger.error(
                'Duplicate tool name detected name=%s existing=%s current=%s',
                tool.name,
                existing_location,
                current_location,
            )
            raise DuplicateToolException(f'Duplicate tool name detected: {tool.name}')
        seen_names[tool.name] = current_location
