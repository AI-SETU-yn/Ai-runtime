# Yn AI Setu AI Runtime - Phase 1

## What this is

Production-oriented Phase 1 foundation for the Yn AI Setu AI Runtime, refactored into a more scalable enterprise architecture without changing existing API behavior.

Implemented scope:
- `POST /chat`
- `GET /health`
- `GET /ready`
- JWT validation
- optional local development auth bypass
- request-scoped runtime context
- static LangGraph workflow foundation
- planner node
- model gateway client
- structured request logging
- correlation/request/conversation IDs
- unit tests

Not implemented in this phase:
- MCP
- tool registry
- tool execution
- approvals
- memory
- streaming responses
- guardrails
- multi-agent orchestration
- LiteLLM integration
- LoRA integration

## Folder structure

- `app/api`: FastAPI routes
- `app/config`: settings and environment configuration
- `app/conversation`: request, correlation, and conversation context management
- `app/exceptions`: domain exceptions and handlers
- `app/graph`: LangGraph state, graph builder, and node modules
- `app/middleware`: HTTP middleware for request context and timing
- `app/model_gateway`: gateway-facing client, config model, and gateway exceptions
- `app/models`: domain-separated API, runtime, planner, response, and error models
- `app/planner`: planner service, parser, prompts, and planner-specific models
- `app/prompts`: generic prompt templates and prompt builder
- `app/security`: JWT validation and runtime-context construction
- `app/services`: service layer and dependency assembly
- `app/utils`: logging configuration

## Architecture

The runtime is intentionally layered:

1. API layer receives requests and validates transport-level input.
2. Security layer validates JWT and builds a typed `RuntimeContext`.
3. Conversation layer provides request, correlation, and conversation identifiers.
4. Service layer constructs `RuntimeState` and orchestrates the workflow.
5. LangGraph executes a static workflow: planner, then response generator.
6. Model gateway client calls only the Model Gateway abstraction.

The runtime does not know about Ollama, vLLM, OpenAI, or any specific provider. That boundary stays behind the Model Gateway.

## Authentication modes

### Local Development Mode

Set:

```env
AI_RUNTIME_BYPASS_AUTH=true
```

Behavior:
- JWT validation is bypassed intentionally for local development only
- protected endpoints work from Swagger without clicking `Authorize`
- a dummy `RuntimeContext` is created automatically
- startup logs a warning that the service is running in local development mode

### Production Mode

Set:

```env
AI_RUNTIME_BYPASS_AUTH=false
```

Behavior:
- normal JWT validation runs exactly as before
- missing bearer token returns `401 Unauthorized`
- valid JWT is required for protected endpoints

### Local development against cloud Auth Service

1. Copy `.env.example` to `.env`
2. Set:

```env
AI_RUNTIME_BYPASS_AUTH=false
AI_RUNTIME_JWT_SECRET=<same secret used by Auth Service or Core API Gateway JWT_SECRET>
AI_RUNTIME_JWT_ALGORITHM=HS256
```

3. Restart the AI Runtime

With that setup:
- missing token returns `401`
- invalid token returns `401`
- valid JWTs signed by the existing Auth Service secret are accepted

## Request flow

1. `POST /chat`
2. `RequestContextMiddleware` creates or propagates `requestId`, `correlationId`, and `conversationId`
3. Authentication path depends on configuration:
   - bypass enabled: local development `RuntimeContext` is created
   - bypass disabled: JWT bearer token is validated using `AI_RUNTIME_JWT_SECRET`/`AI_RUNTIME_JWT_ALGORITHM`; runtime also accepts Core Gateway-compatible aliases `JWT_SECRET`/`JWT_ALGORITHM`
4. `RuntimeContext` is attached to the request
5. `ChatService` builds `RuntimeState`
6. `WorkflowManager` runs the static LangGraph workflow
7. `PlannerNode` generates `PlannerOutput` and `executionPlan=[]`
8. `ResponseGeneratorNode` builds the response prompt and calls `ModelGatewayClient`
9. Final response is returned with conversation metadata

## LangGraph workflow

The workflow is intentionally static:

- `START`
- `Planner`
- `Response Generator`
- `END`

Dynamic behavior belongs only in planner output metadata. The planner can mark `requires_tool=true`, but no tool execution happens in Phase 1.

## RuntimeState

`RuntimeState` is the single communication object shared by all graph nodes.

It contains:
- `conversation_id`
- `request_id`
- `correlation_id`
- `runtime_context`
- `user_question`
- `planner_output`
- `execution_plan`
- `model_response`
- `final_response`
- `trace_id`

This keeps node communication explicit and prepares the runtime for later MCP integration without changing the graph contract.

## Future extension points

- `execution_plan` can be consumed later by an MCP client without changing the current workflow shape.
- `model_gateway/` is isolated so LiteLLM or other downstream orchestration can be introduced without changing runtime business logic.
- `conversation/` can later support richer session policies without spreading context logic across middleware and services.
- `graph/nodes/` allows additional internal node refactors later while preserving the same top-level workflow.

## Reuse from Core API Gateway

The runtime intentionally mirrors these gateway decisions without editing the gateway project:
- shared-secret JWT validation pattern
- environment-driven configuration
- health/readiness style

The runtime improves on that baseline by introducing typed runtime context, centralized errors, separated graph nodes, conversation context abstractions, and domain-based models.

## Run locally

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

