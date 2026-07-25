import logging

from app.graph.state import RuntimeState
from app.model_gateway.client import ModelGatewayClient
from app.prompts.builder import PromptBuilder
from app.utils.json_logging import pretty_json

logger = logging.getLogger(__name__)


class ResponseGeneratorNode:
    def __init__(self, prompt_builder: PromptBuilder, model_gateway_client: ModelGatewayClient) -> None:
        self._prompt_builder = prompt_builder
        self._model_gateway_client = model_gateway_client

    async def __call__(self, state: RuntimeState):
        planner_output = state.planner_output
        assert planner_output is not None
        prompt = self._prompt_builder.build_response_prompt(
            state.user_question,
            planner_output.intent,
            planner_output.requires_tool,
            state.tool_execution_result,
        )
        logger.info('response_generator_prompt_json\n%s', pretty_json({
            'user_question': state.user_question,
            'planner_intent': planner_output.intent,
            'requires_tool': planner_output.requires_tool,
            'domain': planner_output.domain,
            'service': planner_output.service,
            'entity': planner_output.entity,
            'operation': planner_output.operation,
            'parameters': planner_output.parameters,
            'tool_execution_result': state.tool_execution_result,
            'grounded_prompt': prompt,
        }))
        answer = await self._model_gateway_client.generate(
            prompt,
            metadata={
                'intent': planner_output.intent,
                'requires_tool': planner_output.requires_tool,
                'domain': planner_output.domain,
                'service': planner_output.service,
                'entity': planner_output.entity,
                'operation': planner_output.operation,
                'parameters': planner_output.parameters,
                'tool_execution_result': state.tool_execution_result,
                'conversation_id': state.conversation_id,
                'trace_id': state.trace_id,
            },
        )
        logger.info('final_response_json\n%s', pretty_json({'final_response': answer}))
        return {
            'model_response': answer,
            'final_response': answer,
        }
