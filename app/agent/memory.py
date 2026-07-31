from __future__ import annotations

from typing import Protocol

from app.agent.models import AgentState, Observation


class MemoryInterface(Protocol):
    async def load(self, state: AgentState) -> dict[str, object]:
        ...

    async def save_observation(self, state: AgentState, observation: Observation) -> None:
        ...


class RequestScopedMemory:
    """Request-local memory placeholder for future conversation or semantic stores."""

    async def load(self, state: AgentState) -> dict[str, object]:
        return dict(state.memory_context)

    async def save_observation(self, state: AgentState, observation: Observation) -> None:
        state.memory_context.setdefault('observation_ids', []).append(observation.observation_id)
