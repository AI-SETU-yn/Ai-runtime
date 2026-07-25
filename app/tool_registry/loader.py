"""YAML loader for tool registry files."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml
from pydantic import ValidationError

from app.tool_registry.exceptions import InvalidToolRegistryException
from app.tool_registry.models import ToolRegistry

logger = logging.getLogger(__name__)


class ToolRegistryLoader:
    """Loads and validates tool registry YAML files from disk.

    The loader is responsible for reading YAML documents, validating them against
    immutable Pydantic models, and returning registry objects that can be indexed
    by the repository during application startup.
    """

    def load_registry_file(self, file_path: Path) -> ToolRegistry:
        """Load one YAML file into a validated registry model.

        Args:
            file_path: Absolute or relative path to a registry YAML file.

        Returns:
            A validated immutable ``ToolRegistry`` instance.

        Raises:
            InvalidToolRegistryException: If the file cannot be read, parsed, or validated.
        """
        try:
            raw_content = file_path.read_text(encoding='utf-8')
        except OSError as exc:
            raise InvalidToolRegistryException(f'Unable to read registry file: {file_path}') from exc

        try:
            payload = yaml.safe_load(raw_content)
        except yaml.YAMLError as exc:
            raise InvalidToolRegistryException(f'Invalid YAML in registry file: {file_path}') from exc

        if not isinstance(payload, dict):
            raise InvalidToolRegistryException(f'Registry file must contain a YAML object: {file_path}')

        try:
            registry = ToolRegistry.model_validate(payload)
        except ValidationError as exc:
            raise InvalidToolRegistryException(
                f'Registry file schema validation failed: {file_path}'
            ) from exc

        logger.info(
            'Registry loaded domain=%s service=%s tools=%s path=%s',
            registry.domain,
            registry.service,
            len(registry.tools),
            file_path,
        )
        return registry

    def load_registries(self, root_path: Path) -> list[ToolRegistry]:
        """Load every YAML registry file under the configured root directory.

        Args:
            root_path: Root directory that contains registry YAML files.

        Returns:
            A list of validated immutable registry models.

        Raises:
            InvalidToolRegistryException: If the directory path itself is invalid.
        """
        if not root_path.exists():
            logger.warning('Tool registry directory does not exist: %s', root_path)
            return []

        if not root_path.is_dir():
            raise InvalidToolRegistryException(f'Tool registry path is not a directory: {root_path}')

        registries: list[ToolRegistry] = []
        yaml_files = sorted(root_path.rglob('*.yaml')) + sorted(root_path.rglob('*.yml'))

        seen_paths: set[Path] = set()
        unique_yaml_files: list[Path] = []
        for file_path in yaml_files:
            if file_path not in seen_paths:
                seen_paths.add(file_path)
                unique_yaml_files.append(file_path)

        for file_path in unique_yaml_files:
            registries.append(self.load_registry_file(file_path))

        logger.info('Registry file scan completed root=%s registry_count=%s', root_path, len(registries))
        return registries
