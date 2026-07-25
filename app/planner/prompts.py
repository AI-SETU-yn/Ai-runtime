from app.prompts.templates import PLANNER_PROMPT


class PlannerPromptProvider:
    def get_prompt_template(self) -> str:
        return PLANNER_PROMPT
