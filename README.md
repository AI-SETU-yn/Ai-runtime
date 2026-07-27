# Yn AI Setu AI Runtime v2

## What This Is

AI Runtime v2 is the FastAPI service that receives chat requests, builds a request-scoped runtime context, plans the user intent through the Model Gateway, resolves tool-backed intents through the tool registry, invokes MCP tools, and asks the Model Gateway to generate the final answer.

Implemented scope:
- `POST /chat`
- `GET /health`
- `GET /ready`
- JWT validation with Core Gateway-compatible aliases
- optional local development auth bypass
- request/correlation/conversation/trace IDs
- LangGraph workflow
- planner node
- tool registry
- MCP client and tool execution
- model gateway client
- structured request/runtime/tool logs
- unit tests

Not implemented yet:
- approvals
- memory
- streaming responses
- guardrails
- multi-agent orchestration
- LiteLLM integration
- LoRA integration

## Folder Structure

- `app/api`: FastAPI routes
- `app/config`: top-level app settings and environment configuration
- `app/conversation`: request, correlation, and conversation context management
- `app/exceptions`: domain exceptions and handlers
- `app/graph`: LangGraph state, graph builder, and node modules
- `app/mcp_client`: MCP server config, connection pooling, transport, execution, and response parsing
- `app/middleware`: HTTP middleware for request context and timing
- `app/model_gateway`: gateway-facing client, config model, and gateway exceptions
- `app/models`: shared API, runtime, planner, response, and error models
- `app/planner`: planner service, parser, and prompts
- `app/prompts`: generic prompt templates and prompt builder
- `app/security`: JWT validation and runtime-context construction
- `app/services`: service layer and dependency assembly
- `app/tool_executor`: registry-backed MCP execution orchestration
- `app/tool_registry`: YAML registry loader, repository, models, and lookup service
- `app/utils`: logging helpers
- `tool-registry`: business tool definitions consumed by the runtime

## Startup Flow

1. `create_app()` builds the FastAPI app and registers CORS, request context middleware, exception handlers, and runtime routes.
2. On lifespan startup, `get_settings()` reads `.env` and `AI_RUNTIME_*` variables.
3. Logging is configured, and local auth bypass is warned if enabled.
4. `ToolRegistryService.initialize()` loads YAML files from `AI_RUNTIME_TOOL_REGISTRY_PATH`, defaulting to `tool-registry`.
5. `get_mcp_client()` creates the MCP client, loads MCP servers from `AI_RUNTIME_MCP_SERVERS_CONFIG_PATH`, and registers each server.
6. `ModelGatewayClient`, `PlannerService`, `ToolExecutorService`, and `WorkflowManager` are assembled and stored on `app.state`.
7. `/ready` returns `ok` once startup has completed.

## Chat Flow

1. Client calls `POST /chat` with `{"message": "..."}` and optionally `conversation_id`.
2. `RequestContextMiddleware` creates or propagates:
   - `request_id`
   - `correlation_id`
   - `conversation_id`
3. Auth runs:
   - local mode: `AI_RUNTIME_BYPASS_AUTH=true` creates a development `RuntimeContext`
   - production mode: bearer JWT is validated using `AI_RUNTIME_JWT_SECRET` / `AI_RUNTIME_JWT_ALGORITHM`
   - Core Gateway aliases `JWT_SECRET` and `JWT_ALGORITHM` are also accepted
4. `ChatService` creates a `RuntimeState` with the user message, runtime context, IDs, and trace ID.
5. `WorkflowManager` runs the LangGraph workflow.
6. `PlannerNode` calls `ModelGatewayClient.plan()` and parses a `PlannerOutput`.
7. `ToolExecutorNode` checks `planner_output.requires_tool`:
   - if `false`, it returns `None`
   - if `true`, it resolves the tool from `tool-registry`, builds MCP arguments/context, and invokes the selected MCP server
8. `ResponseGeneratorNode` builds a grounded response prompt using the planner output and optional tool result.
9. `ModelGatewayClient.generate()` returns the final answer.
10. `ChatService` returns `ChatResponse` with answer plus conversation metadata.

## LangGraph Workflow

The workflow is static:

- `START`
- `planner`
- `tool_executor`
- `response_generator`
- `END`

Dynamic behavior lives in planner output. Tool execution is skipped for non-tool intents but remains in the graph so the response generator always receives the same state shape.

## MCP Config

The canonical default MCP server file is:

```text
app/mcp_client/config/servers.yaml
```

`Settings.mcp_servers_config_path` points there by default. Override it only when needed:

```env
AI_RUNTIME_MCP_SERVERS_CONFIG_PATH=/path/to/servers.yaml
```

The server names in `servers.yaml` must match the `server` field in tool registry YAML files. For example, `tool-registry/vidhya/academic.yaml` uses `vidhya-mcp`, so the MCP config must contain a `vidhya-mcp` server entry.

## RuntimeState

`RuntimeState` is the shared graph object.

It contains:
- `conversation_id`
- `request_id`
- `correlation_id`
- `runtime_context`
- `user_question`
- `planner_output`
- `tool_execution_result`
- `model_response`
- `final_response`
- `trace_id`

## Run Locally

```bash
python -m venv .venv
. .venv/Scripts/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Test

```bash
pytest
```
