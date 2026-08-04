import logging
from time import perf_counter

from app.exceptions.errors import PlannerError, PlannerValidationError
from app.model_gateway.adapter_resolver import ModelGatewayAdapterResolver
from app.model_gateway.client import ModelGatewayClient
from app.model_gateway.exceptions import ModelGatewayError
from app.models.planner import PlannerOutput
from app.planner.parser import PlannerOutputParser
from app.planner.prompts import PlannerPromptProvider
from app.planner.registry_validator import PlannerRegistryValidator
from app.prompts.builder import PromptBuilder
from app.utils.json_logging import pretty_json
from app.utils.redaction import redact_sensitive

logger = logging.getLogger(__name__)


class PlannerService:
    def __init__(
        self,
        prompt_builder: PromptBuilder,
        prompt_provider: PlannerPromptProvider,
        output_parser: PlannerOutputParser,
        model_gateway_client: ModelGatewayClient,
        registry_validator: PlannerRegistryValidator | None = None,
        adapter_resolver: ModelGatewayAdapterResolver | None = None,
    ) -> None:
        self._prompt_builder = prompt_builder
        self._prompt_provider = prompt_provider
        self._output_parser = output_parser
        self._model_gateway_client = model_gateway_client
        self._registry_validator = registry_validator
        self._adapter_resolver = adapter_resolver or ModelGatewayAdapterResolver()

    async def plan(self, message: str):
        try:
            output = await self._request_plan(message, retry=False)
        except PlannerError:
            logger.exception('planner_invalid_output_retrying_once')
            output = await self._request_plan(message, retry=True)
        if self._should_retry_for_incomplete_execution_plan(output):
            logger.warning('planner_validation_retry_json\n%s', pretty_json({
                'message': message,
                'reason': 'registry_proved_execution_plan_incomplete',
                'initial_output': output.model_dump(),
            }))
            retried_output = await self._request_plan(message, retry=True)
            if self._should_retry_for_incomplete_execution_plan(retried_output):
                logger.error('planner_validation_failed_json\n%s', pretty_json({
                    'message': message,
                    'initial_output': output.model_dump(),
                    'retried_output': retried_output.model_dump(),
                }))
                raise PlannerValidationError(
                    'Planner returned an incomplete execution plan after retry.'
                )
            output = retried_output
        return output

    async def _request_plan(self, message: str, *, retry: bool):
        registry_context = self._registry_validator.registry_prompt_context() if self._registry_validator else None
        prompt = self._prompt_builder.build_planner_prompt(
            message,
            registry_context=registry_context,
            base_prompt=self._prompt_provider.get_prompt_template(),
        )
        if retry:
            prompt = (
                f'{prompt}\n\n'
                'Planner validation retry: return a complete execution_plan with every required tool step. '
                'Include dependency-only helper steps when registry metadata requires them. '
                'Keep legacy single-tool responses only when the request truly resolves to one tool.'
            )
        logger.info('planner_prompt_built')
        logger.info('planner_request_json\n%s', pretty_json(redact_sensitive({'query': message, 'prompt': prompt, 'retry': retry})))

        started = perf_counter()
        try:
            planner_response = await self._call_model_gateway_plan(message, prompt)
        except ModelGatewayError:
            logger.exception('planner_gateway_failed_using_planner_failure')
            return self._planner_failure_output('planner_gateway_unreachable')
        latency_ms = round((perf_counter() - started) * 1000, 2)

        logger.info('planner_response_received latency_ms=%s', latency_ms)
        logger.info('planner_response_json\n%s', pretty_json(redact_sensitive(planner_response)))
        output = self._output_parser.parse(
            intent=planner_response.get('intent'),
            domain=planner_response.get('domain'),
            service=planner_response.get('service'),
            entity=planner_response.get('entity'),
            operation=planner_response.get('operation'),
            tool=planner_response.get('tool'),
            parameters=planner_response.get('parameters'),
            execution_plan=planner_response.get('execution_plan') or planner_response.get('executionPlan'),
            tasks=planner_response.get('tasks'),
            requires_tool=planner_response.get('requiresTool')
            if 'requiresTool' in planner_response
            else planner_response.get('requires_tool'),
            raw_response=planner_response.get('rawResponse') or planner_response.get('raw_response'),
            adapter=planner_response.get('adapter'),
            model=planner_response.get('model'),
        )
        original_output = output
        logger.info('planner_original_output_json\n%s', pretty_json(redact_sensitive(original_output.model_dump())))
        if self._registry_validator and output.requires_tool:
            validation = self._registry_validator.normalize_and_validate(output)
            logger.info('planner_registry_validation_json\n%s', pretty_json(redact_sensitive({
                'original_output': original_output.model_dump(),
                'normalized_output': validation.output.model_dump(),
                'normalized': validation.normalized,
                'normalization_reasons': validation.reasons,
                'registry_lookup_result': 'success' if validation.resolved_tool else 'failure',
                'resolved_tool': validation.resolved_tool.model_dump() if validation.resolved_tool else None,
                'failure_reason': validation.failure_reason,
            })))
            if validation.failure_reason:
                if self._is_registry_miss(validation.failure_reason):
                    logger.warning('planner_registry_miss_using_planner_failure_json\n%s', pretty_json(redact_sensitive({
                        'message': message,
                        'failure_reason': validation.failure_reason,
                        'original_output': original_output.model_dump(),
                        'normalized_output': validation.output.model_dump(),
                    })))
                    return self._planner_failure_output('registry_target_not_found')
                raise PlannerError(validation.failure_reason)
            output = validation.output
        if not output.intent:
            raise PlannerError('Planner could not determine an intent.')
        if output.requires_tool and not all((output.domain, output.service, output.entity, output.operation)):
            raise PlannerError('Planner marked the request as tool-backed but did not return a complete execution target.')
        logger.info('planner_output_json\n%s', pretty_json(redact_sensitive(output.model_dump())))
        return output

    async def _call_model_gateway_plan(self, message: str, prompt: str) -> dict:
        adapter = self._adapter_resolver.resolve_for_planning()
        try:
            return await self._model_gateway_client.plan(message, prompt=prompt, adapter=adapter)
        except TypeError as exc:
            if 'adapter' not in str(exc):
                raise
            logger.debug('planner_gateway_client_adapter_argument_not_supported')
            return await self._model_gateway_client.plan(message, prompt=prompt)

    @staticmethod
    def _planner_failure_output(reason: str) -> PlannerOutput:
        return PlannerOutput(
            intent='planner.failure',
            requires_tool=False,
            raw_response=f'planner_failure:{reason}',
        )

    @staticmethod
    def _is_registry_miss(failure_reason: str) -> bool:
        return (
            'does not match any unique registered tool target' in failure_reason
            or 'No tool found for logical lookup key' in failure_reason
        )

    def _should_retry_for_incomplete_execution_plan(self, output: PlannerOutput) -> bool:
        if not output.requires_tool:
            return False
        if not output.execution_plan:
            return False
        if len(output.execution_plan) != 1:
            return False
        if not self._registry_validator:
            return False
        return self._registry_validator.plan_requires_additional_steps(output)
