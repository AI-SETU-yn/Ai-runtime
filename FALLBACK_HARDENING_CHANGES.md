# AI Runtime fallback-hardening changes

## Summary

The AI Runtime keeps its existing agent workflow:

```text
Reason -> Act -> Observe -> Decide -> Final response
```

Fallback hardening and optional web search were added without replacing the
agent architecture or changing ERP/tool execution behavior.

## Routing behavior

```text
Planner requires_tool=true
  -> existing ERP/tool execution
  -> You.com is never called

Planner requires_tool=false + general question
  -> existing GENERAL_LLM path

Planner requires_tool=false + current-information question
  -> optional You.com web search
  -> normalized usable evidence -> existing LLM response generation
  -> web failure/unusable evidence -> existing GENERAL_LLM path

LLM failure or blank response
  -> safe fallback response

Planner gateway failure
  -> general.chat with requires_tool=false
```

## Added components

- `app/utils/current_info_detector.py` — identifies likely current-information
  questions and never overrides the Planner's ERP/tool decision.
- `app/web_search/client.py` — optional HTTPX web client with retry, timeout,
  invalid-data, and unusable-result handling plus prompt-size sanitization.
- `app/graph/nodes/current_info_router.py` — invokes web search only for
  non-tool current-information requests and otherwise falls back to GENERAL_LLM.
- `tests/test_current_info_fallback.py` — focused fallback and You.com tests.

## Modified integration points

- `app/config/settings.py`: optional web-search settings, disabled by default.
- `app/exceptions/errors.py`: `WebSearchError` and `WebSearchTimeoutError`.
- `app/services/dependencies.py`: `WebSearchClient` dependency wiring.
- `app/graph/graph.py` and `app/agent/agent.py`: final-response integration
  without changing the Reason/Act/Observe/Decide graph topology.
- `app/graph/nodes/response_generator.py`: safe fallback for Model Gateway
  failures and blank responses.
- `app/planner/planner.py`: neutral `general.chat` fallback on Planner failure.

## You.com compatibility

The adapter supports You.com's response format:

```json
{
  "results": {
    "web": [
      {
        "title": "...",
        "description": "...",
        "snippets": ["..."],
        "url": "..."
      }
    ]
  },
  "metadata": {}
}
```

It extracts bounded, sanitized `title`, `url`, `description`, and `snippets`.
When description is empty, usable snippets are retained. Missing, empty,
metadata-only, malformed, invalid, or error responses fall back to GENERAL_LLM.

## Configuration

```env
AI_RUNTIME_WEB_SEARCH_ENABLED=true
AI_RUNTIME_WEB_SEARCH_URL=https://api.you.com/v1/search
AI_RUNTIME_WEB_SEARCH_API_KEY=your-you-com-api-key
```

Additional settings control result count, timeouts, retries, result length, and
total web-context length. No additional HTTP dependency is required.

## Validation

- `/health` and `/ready`: passed (`200`).
- Live You.com verification: passed; normalized `results.web` evidence was
  returned successfully.
- Current-information query verification (`who is current cheif minster of
  telangana`): passed with three normalized results.
- Final routing audit: passed for general, current-information, ERP-priority,
  web-failure, LLM-failure, and Planner-failure paths.
- Full test suite: `91 passed`.
- Syntax/import/startup checks: passed.
- `git diff --check`: passed.

## Intentionally unchanged

- ERP/tool execution and ToolExecutor behavior.
- Agent graph topology and lifecycle.
- Guardrails, authentication, authorization, and retry behavior.
- Model Gateway code and Model Gateway configuration.
