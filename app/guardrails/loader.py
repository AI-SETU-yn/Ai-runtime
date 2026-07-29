from __future__ import annotations

from pathlib import Path

import yaml

from app.guardrails.models import GuardrailsConfig


class GuardrailsLoader:
    def load(self, path: Path) -> GuardrailsConfig:
        raw = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
        return GuardrailsConfig.model_validate(raw)
