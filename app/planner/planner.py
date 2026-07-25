import logging
from time import perf_counter

from app.exceptions.errors import PlannerError
from app.model_gateway.client import ModelGatewayClient
from app.planner.parser import PlannerOutputParser
from app.planner.prompts import PlannerPromptProvider
from app.prompts.builder import PromptBuilder
from app.utils.json_logging import pretty_json

logger = logging.getLogger(__name__)


class PlannerService:
    def __init__(
        self,
        prompt_builder: PromptBuilder,
        prompt_provider: PlannerPromptProvider,
        output_parser: PlannerOutputParser,
        model_gateway_client: ModelGatewayClient,
    ) -> None:
        self._prompt_builder = prompt_builder
        self._prompt_provider = prompt_provider
        self._output_parser = output_parser
        self._model_gateway_client = model_gateway_client

    async def plan(self, message: str):
        prompt = self._prompt_builder.build_planner_prompt(message)
        logger.info('planner_prompt_built')
        logger.info('planner_request_json\n%s', pretty_json({'query': message, 'prompt': prompt}))

        started = perf_counter()
        planner_response = await self._model_gateway_client.plan(message)
        latency_ms = round((perf_counter() - started) * 1000, 2)

        logger.info('planner_response_received latency_ms=%s', latency_ms)
        logger.info('planner_response_json\n%s', pretty_json(planner_response))
        output = self._output_parser.parse(
            intent=planner_response.get('intent'),
            domain=planner_response.get('domain'),
            service=planner_response.get('service'),
            entity=planner_response.get('entity'),
            operation=planner_response.get('operation'),
            tool=planner_response.get('tool'),
            parameters=planner_response.get('parameters'),
            requires_tool=planner_response.get('requiresTool'),
            raw_response=planner_response.get('rawResponse'),
            adapter=planner_response.get('adapter'),
            model=planner_response.get('model'),
        )
        if not output.intent:
            raise PlannerError('Planner could not determine an intent.')
        if output.requires_tool and not all((output.domain, output.service, output.entity, output.operation)):
            raise PlannerError('Planner marked the request as tool-backed but did not return a complete execution target.')
        logger.info('planner_output_json\n%s', pretty_json(output.model_dump()))
        return output
