import pytest

from app.graph.nodes.response_generator import ResponseGeneratorNode
from app.graph.state import RuntimeState
from app.model_gateway.exceptions import ModelGatewayError
from app.models.planner import PlannerOutput
from app.models.runtime import RuntimeContext


class CapturingGateway:
    def __init__(self, *answers: str) -> None:
        self.answers = list(answers) or ['Model grounded answer']
        self.requests: list[dict] = []

    async def generate(self, **kwargs) -> str:
        self.requests.append(kwargs)
        return self.answers.pop(0) if self.answers else 'Model grounded answer'


class FailingGateway:
    async def generate(self, **kwargs) -> str:
        raise ModelGatewayError('down')


def build_state(
    message: str,
    *,
    entity: str,
    records: list[dict],
    answer_data_key: str = 'items',
) -> RuntimeState:
    return RuntimeState(
        conversation_id='conv-1',
        request_id='req-1',
        correlation_id='corr-1',
        runtime_context=RuntimeContext(subject='user-1', user_id='user-1'),
        user_question=message,
        planner_output=PlannerOutput(
            intent=f'generic.{entity}.list',
            requires_tool=True,
            domain='generic',
            service='generic',
            entity=entity,
            operation='list',
        ),
        tool_execution_result={'success': True, 'data': {answer_data_key: records}},
    )


@pytest.mark.asyncio
async def test_response_generator_dispatches_structured_enterprise_data():
    gateway = CapturingGateway('There are two records: Alpha and Beta.')
    node = ResponseGeneratorNode(gateway)

    result = await node.__call__(
        build_state(
            'List records',
            entity='custom_record',
            records=[{'displayName': 'Alpha'}, {'displayName': 'Beta'}],
        )
    )

    assert result['final_response'] == 'There are two records: Alpha and Beta.'
    assert len(gateway.requests) == 1
    request = gateway.requests[0]
    assert request['response_type'] == 'enterprise'
    assert request['messages'] == [{'role': 'user', 'content': 'List records'}]
    assert request['tool_result']['data']['items'][0]['displayName'] == 'Alpha'
    assert request['conversation']['plannerIntent'] == 'generic.custom_record.list'
    assert request['generation_policy']['hallucination'] == 'forbid'


@pytest.mark.asyncio
async def test_response_generator_only_uses_minimal_runtime_fallback_when_gateway_fails():
    node = ResponseGeneratorNode(FailingGateway())

    result = await node.__call__(
        build_state(
            'List records',
            entity='custom_record',
            records=[{'displayName': 'Alpha', 'referenceId': 'hidden-id'}],
        )
    )

    assert result['final_response'] == 'The AI service is temporarily unavailable.'


@pytest.mark.asyncio
async def test_response_generator_retries_ungrounded_gateway_answer_without_formatting():
    gateway = CapturingGateway('The answer is Gamma.', 'Alpha and Beta are available.')
    node = ResponseGeneratorNode(gateway)

    result = await node.__call__(
        build_state(
            'List records',
            entity='custom_record',
            records=[{'displayName': 'Alpha'}, {'displayName': 'Beta'}],
        )
    )

    assert result['final_response'] == 'Alpha and Beta are available.'
    assert [request['response_type'] for request in gateway.requests] == ['enterprise', 'validation_retry']
    assert gateway.requests[1]['metadata']['previous_response'] == 'The answer is Gamma.'


@pytest.mark.asyncio
async def test_response_generator_dispatches_clarification_to_gateway():
    gateway = CapturingGateway('Which value should I use?')
    node = ResponseGeneratorNode(gateway)
    state = build_state('List records', entity='custom_record', records=[])
    state.tool_execution_result = {
        'status': 'requires_input',
        'awaiting_input': True,
        'clarification': {'missing_parameters': ['record_id'], 'options': [{'label': 'Alpha', 'value': 'a'}]},
    }

    result = await node.__call__(state)

    assert result['final_response'] == 'Which value should I use?'
    assert gateway.requests[0]['response_type'] == 'clarification'
    assert gateway.requests[0]['missing_parameters'] == ['record_id']
    assert gateway.requests[0]['tool_result']['clarification']['options'][0]['value'] == 'a'


@pytest.mark.asyncio
async def test_response_generator_dispatches_tool_failure_to_gateway():
    gateway = CapturingGateway('I cannot retrieve that data right now.')
    node = ResponseGeneratorNode(gateway)
    state = build_state('List records', entity='custom_record', records=[])
    state.tool_execution_result = {
        'status': 'error',
        'success': False,
        'tool_name': 'generic.tool',
        'error': {'message': 'Downstream service returned HTTP 401'},
    }

    result = await node.__call__(state)

    assert result['final_response'] == 'I cannot retrieve that data right now.'
    assert gateway.requests[0]['response_type'] == 'tool_failure'
    assert gateway.requests[0]['tool_result']['error']['message'] == 'Downstream service returned HTTP 401'


@pytest.mark.asyncio
async def test_multi_step_response_context_contains_all_visible_step_results():
    gateway = CapturingGateway('Alpha and Beta are available.')
    node = ResponseGeneratorNode(gateway)
    state = RuntimeState(
        conversation_id='conv-1',
        request_id='req-1',
        correlation_id='corr-1',
        runtime_context=RuntimeContext(subject='user-1', user_id='user-1'),
        user_question='List two datasets',
        planner_output=PlannerOutput(
            intent='generic.multi.list',
            requires_tool=True,
            domain='generic',
            service='generic',
            entity='dataset',
            operation='list',
            execution_plan=[
                {
                    'step_id': 'step_1',
                    'intent': 'generic.dataset_a.list',
                    'domain': 'generic',
                    'service': 'generic',
                    'entity': 'dataset_a',
                    'operation': 'list',
                    'question': 'List dataset A',
                    'visible_in_response': True,
                },
                {
                    'step_id': 'step_2',
                    'intent': 'generic.dataset_b.list',
                    'domain': 'generic',
                    'service': 'generic',
                    'entity': 'dataset_b',
                    'operation': 'list',
                    'question': 'List dataset B',
                    'visible_in_response': True,
                },
            ],
        ),
        tool_execution_result={
            'success': True,
            'steps': [
                {'step_id': 'step_1', 'result': {'success': True, 'data': {'items': [{'name': 'Alpha'}]}}},
                {'step_id': 'step_2', 'result': {'success': True, 'data': {'items': [{'name': 'Beta'}]}}},
            ],
            'data': {'items': [{'name': 'Beta'}]},
        },
    )

    result = await node.__call__(state)

    assert result['final_response'] == 'Alpha and Beta are available.'
    request = gateway.requests[0]
    assert request['response_type'] == 'multi_tool'
    assert request['tool_result']['steps'][0]['result']['data']['items'][0]['name'] == 'Alpha'
    assert request['tool_result']['steps'][1]['result']['data']['items'][0]['name'] == 'Beta'


@pytest.mark.asyncio
async def test_multi_step_response_context_hides_invisible_helper_steps():
    gateway = CapturingGateway('Beta is available.')
    node = ResponseGeneratorNode(gateway)
    state = RuntimeState(
        conversation_id='conv-1',
        request_id='req-1',
        correlation_id='corr-1',
        runtime_context=RuntimeContext(subject='user-1', user_id='user-1'),
        user_question='List dependent result',
        planner_output=PlannerOutput(
            intent='generic.dataset_b.list',
            requires_tool=True,
            domain='generic',
            service='generic',
            entity='dataset_b',
            operation='list',
            execution_plan=[
                {
                    'step_id': 'step_1',
                    'intent': 'generic.dataset_a.list',
                    'domain': 'generic',
                    'service': 'generic',
                    'entity': 'dataset_a',
                    'operation': 'list',
                    'visible_in_response': False,
                },
                {
                    'step_id': 'step_2',
                    'intent': 'generic.dataset_b.list',
                    'domain': 'generic',
                    'service': 'generic',
                    'entity': 'dataset_b',
                    'operation': 'list',
                    'question': 'List dataset B',
                    'visible_in_response': True,
                },
            ],
        ),
        tool_execution_result={
            'success': True,
            'steps': [
                {'step_id': 'step_1', 'result': {'success': True, 'data': {'items': [{'name': 'Alpha'}]}}},
                {'step_id': 'step_2', 'result': {'success': True, 'data': {'items': [{'name': 'Beta'}]}}},
            ],
            'data': {'items': [{'name': 'Beta'}]},
        },
    )

    await node.__call__(state)

    request = gateway.requests[0]
    assert request['tool_result']['steps'][0]['result']['data']['items'][0]['name'] == 'Alpha'
    assert request['tool_result']['steps'][1]['result']['data']['items'][0]['name'] == 'Beta'
    assert request['conversation']['executionPlan'][0]['visibleInResponse'] is False
