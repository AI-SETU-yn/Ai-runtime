from __future__ import annotations

from typing import Protocol

from app.security.models import SecurityDecision


class SecurityClassifier(Protocol):
    async def classify(self, message: str) -> SecurityDecision:
        ...