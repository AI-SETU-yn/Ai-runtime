from __future__ import annotations

import json
from typing import Any


def pretty_json(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=True, default=str, sort_keys=True)
