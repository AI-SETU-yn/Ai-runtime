import json

from app.prompts.templates import PLANNER_PROMPT, RESPONSE_SYSTEM_PROMPT


class PromptBuilder:
    def build_planner_prompt(self, message: str) -> str:
        return f"{PLANNER_PROMPT}\n\nUser message:\n{message}"

    def build_response_prompt(
        self,
        message: str,
        planner_intent: str,
        requires_tool: bool,
        tool_execution_result: dict[str, object] | None = None,
    ) -> str:
        tool_results = json.dumps(tool_execution_result or {}, ensure_ascii=True, default=str)
        return (
            f"{RESPONSE_SYSTEM_PROMPT}\n\n"
            f"Planner intent: {planner_intent}\n"
            f"Requires tool: {requires_tool}\n"
            f"Tool execution result: {tool_results}\n\n"
            f"User message:\n{message}"
        )
