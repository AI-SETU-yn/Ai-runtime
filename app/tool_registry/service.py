"""Application service for tool registry access."""

from __future__ import annotations

from pathlib import Path

from app.tool_registry.models import ResolvedTool
from app.tool_registry.repository import ToolRegistryRepository


class ToolRegistryService:
    """Facade that coordinates registry loading and exact lookups.

    The service preserves the current application flow while delegating storage
    and indexing responsibilities to the repository.
    """

    def __init__(self, repository: ToolRegistryRepository | None = None) -> None:
        self._repository = repository or ToolRegistryRepository()

    @property
    def repository(self) -> ToolRegistryRepository:
        """Expose the backing repository for diagnostics and tests."""
        return self._repository

    def initialize(self, root_path: Path) -> None:
        """Load registry files into memory exactly once.

        Args:
            root_path: Directory containing registry YAML files.
        """
        self._repository.load_from_directory(root_path)

    def find_tool(self, domain: str, service: str, entity: str, operation: str) -> ResolvedTool:
        """Resolve a tool using exact planner output fields.

        Args:
            domain: Planner domain.
            service: Planner service.
            entity: Planner entity.
            operation: Planner operation.

        Returns:
            An immutable resolved tool record for downstream execution.
        """
        return self._repository.find_tool(domain, service, entity, operation)
