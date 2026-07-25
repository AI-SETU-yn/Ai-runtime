"""Health endpoint probing."""
import time
import httpx
from ..models.response import HealthState, HealthStatus
from .session import ConnectionSession


class HealthMonitor:
    async def check(self, session: ConnectionSession) -> HealthStatus:
        started = time.perf_counter()
        try:
            response = await (await session.client()).get(session.server.endpoint_path, headers={"Accept": "text/event-stream"})
            state = HealthState.HEALTHY if response.is_success or response.status_code == 405 else HealthState.UNHEALTHY
            details = {"status_code": response.status_code}
        except httpx.HTTPError as exc:
            state, details = HealthState.UNHEALTHY, {"error": type(exc).__name__}
        return HealthStatus(server=session.server.name, status=state, latency=time.perf_counter() - started, details=details)
