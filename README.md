# AI Runtime

AI Runtime is the enterprise orchestration service behind `/chat`. The current runtime is driven by `GeneralAgent`, which wraps the existing planner, registry, tool executor, MCP client, and response generator without changing the public API contract.

## Request Flow

```text
POST /chat
  |
  v
ChatService
  |
  v
WorkflowManager
  |
  v
GeneralAgent
  |
  v
Reason -> PlannerService -> Model Gateway /planner
  |
  v
Act -> ToolExecutorService -> Tool Registry -> MCP Client -> MCP Server -> Backend Services
  |
  v
Observe -> Agent observations
  |
  v
Decide -> continue or finalize
  |
  v
ResponseGeneratorNode -> Model Gateway /generate
  |
  v
ChatResponse
```

## Graph

The LangGraph workflow is agent-owned:

```text
START
  |
  v
reason
  |
  v
act
  |
  v
observe
  |
  v
decide
  |\
  | \ continue
  |  `----> reason
  |
  `----> response_generator -> END
```

`RuntimeState` is converted to `AgentState` at graph entry. `AgentState` preserves the original runtime fields and adds observations, reasoning history, tool history, memory context, iteration count, current plan, and decision metadata.

## Preserved Components

The General Agent reuses the existing enterprise components:

- `/chat` API
- `ChatService`
- `PlannerService`
- `PlannerRegistryValidator`
- Tool Registry YAML schema
- `ToolExecutorService`
- MCP Client
- MCP Server integrations
- Response generation
- Authentication and runtime context

## Current Agent Scope

The first agent migration keeps behavior backward-compatible. Decisions are deterministic, observations are request-scoped, and tool execution remains sequential.

## Runtime Hardening

The V1 runtime keeps the same public `/chat`, planner, MCP, and tool-registry contracts while applying these production safeguards:

- Runtime services are built through dependency injection and reused as process-local singletons.
- Tool registry and MCP server configuration are loaded once during startup.
- The LangGraph workflow is created lazily on first use and then reused.
- MCP and Model Gateway HTTP clients are closed during application shutdown.
- MCP retry attempts are driven by `AI_RUNTIME_MCP_MAX_RETRIES`.
- Planner and generation requests use separate configurable Model Gateway timeouts.
- Request logging, planner logging, Model Gateway logging, MCP logging, and response-generation logging redact JWTs, Authorization values, bearer tokens, and common secret fields.
- Request context is reset after each request to avoid leaking correlation metadata across concurrent work.
- Production settings reject unsafe combinations such as auth bypass, missing JWT secret, or wildcard CORS when `AI_RUNTIME_ENVIRONMENT=prod`.

## Configuration Files

```text
app/config/settings.py              Runtime environment settings
app/config/guardrails.yaml          Input and output guardrail rules
app/mcp_client/config/servers.yaml  MCP server URLs and transport settings
tool-registry/                      Planner and tool-executor registry source of truth
```

The top-level `config/` folder is not used by the runtime.
