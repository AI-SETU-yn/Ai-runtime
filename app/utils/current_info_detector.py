"""Small heuristic for questions that likely need fresh public information."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re

import yaml
from pydantic import BaseModel, ConfigDict, Field

from app.config.settings import get_settings


class CurrentInfoDetectorConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')

    keywords: list[str] = Field(default_factory=list)
    year_context_keywords: list[str] = Field(default_factory=list)


_YEAR_PATTERN = re.compile(r'\b(?:19|20)\d{2}\b')


@lru_cache(maxsize=8)
def load_current_info_config(path: str) -> CurrentInfoDetectorConfig:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding='utf-8')) or {}
    return CurrentInfoDetectorConfig.model_validate(payload)


def get_current_info_config() -> CurrentInfoDetectorConfig:
    return load_current_info_config(str(get_settings().current_info_config_path))


def is_current_info_query(
    user_question: str,
    config: CurrentInfoDetectorConfig | None = None,
) -> bool:
    """Return whether a non-tool question appears to require current data.

    This function deliberately makes no routing decision. The caller must first
    respect the Planner's ``requires_tool`` decision so ERP/tool work wins.
    """
    if not user_question or not user_question.strip():
        return False
    config = config or get_current_info_config()
    lowered = user_question.casefold()
    if any(keyword.casefold() in lowered for keyword in config.keywords):
        return True
    return bool(_YEAR_PATTERN.search(lowered)) and any(
        word.casefold() in lowered for word in config.year_context_keywords
    )
