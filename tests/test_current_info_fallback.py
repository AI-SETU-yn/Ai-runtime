import pytest
import httpx

from app.exceptions.errors import WebSearchError, WebSearchTimeoutError
from app.graph.nodes.current_info_router import CurrentInfoRouterNode
from app.graph.nodes.response_generator import ResponseGeneratorNode
from app.graph.state import RuntimeState
from app.model_gateway.exceptions import ModelGatewayError
from app.models.planner import PlannerOutput
from app.models.runtime import RuntimeContext
from app.planner.planner import PlannerService
from app.prompts.builder import PromptBuilder
from app.web_search.client import WebSearchClient


def make_state(question: str, *, requires_tool: bool = False) -> RuntimeState:
    return RuntimeState(
        conversation_id='conv-1',
        request_id='req-1',
        correlation_id='corr-1',
        runtime_context=RuntimeContext(subject='user-1', user_id='user-1'),
        user_question=question,
        planner_output=PlannerOutput(intent='general.chat', requires_tool=requires_tool),
    )


class StubSearchClient:
    def __init__(self, result=None, error: Exception | None = None, configured: bool = True):
        self.result = result
        self.error = error
        self.configured = configured
        self.calls = 0

    @property
    def is_configured(self):
        return self.configured

    async def search(self, query):
        self.calls += 1
        if self.error:
            raise self.error
        return self.result


@pytest.mark.asyncio
async def test_erp_priority_never_calls_web_search():
    client = StubSearchClient({'results': [{'title': 'Weather', 'content': 'Sunny'}]})
    updates = await CurrentInfoRouterNode(client)(
        make_state("What is today's attendance for Class 10?", requires_tool=True)
    )
    assert updates == {}
    assert client.calls == 0


@pytest.mark.asyncio
async def test_current_information_uses_normalized_web_evidence_before_llm():
    client = StubSearchClient({'results': [{'title': 'Weather', 'content': 'USA is sunny today.'}]})
    router = CurrentInfoRouterNode(client)
    state = make_state("What is today's weather in the USA?")
    updates = await router(state)
    assert client.calls == 1
    assert updates['tool_execution_result']['data']['results'][0]['content'] == 'USA is sunny today.'

    captured = {}

    class Gateway:
        async def generate(self, prompt, *, metadata=None):
            captured['prompt'] = prompt
            return 'It is sunny.'

    response = await ResponseGeneratorNode(PromptBuilder(), Gateway())(
        state.model_copy(update=updates)
    )
    assert response['final_response'] == 'It is sunny.'
    assert 'USA is sunny today.' in captured['prompt']


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'error',
    [
        WebSearchTimeoutError('timed out'),
        WebSearchError('HTTP 500'),
        WebSearchError('HTTP 401'),
        WebSearchError('HTTP 429'),
    ],
)
async def test_web_failures_fall_through_to_general_llm(error):
    client = StubSearchClient(error=error)
    updates = await CurrentInfoRouterNode(client)(make_state("What is today's weather in the USA?"))
    assert updates == {}
    assert client.calls == 1


@pytest.mark.parametrize(
    'payload',
    [None, {}, [], '', {'error': 'provider failed'}, {'request_id': 'metadata-only'}, {'results': [{}]}],
)
def test_web_client_rejects_unusable_provider_payloads(payload):
    client = object.__new__(WebSearchClient)
    client._max_results = 5
    client._max_result_characters = 1_000
    client._max_context_characters = 4_000
    with pytest.raises(WebSearchError):
        client._normalize_response(payload)


@pytest.mark.asyncio
async def test_web_client_maps_invalid_json_to_web_error(monkeypatch):
    client = object.__new__(WebSearchClient)
    client._enabled = True
    client._url = 'https://search.example.test'
    client._api_key = ''
    client._max_results = 5
    client._max_result_characters = 1_000
    client._max_context_characters = 4_000
    client._max_retries = 0
    client._timeout = httpx.Timeout(1.0)

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            raise ValueError('invalid JSON')

    class AsyncClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, *args, **kwargs):
            return Response()

    monkeypatch.setattr('app.web_search.client.httpx.AsyncClient', AsyncClient)
    with pytest.raises(WebSearchError, match='invalid JSON'):
        await client.search('latest news')


def test_web_client_sanitizes_and_bounds_result_content():
    client = object.__new__(WebSearchClient)
    client._max_results = 1
    client._max_result_characters = 20
    client._max_context_characters = 20
    result = client._normalize_response({'results': [{'title': '<b>News</b>', 'content': '  useful\n text ' * 5}]})
    assert result == {'results': [{'title': 'News', 'content': 'useful text usef'}]}


def test_web_client_normalizes_you_com_results_web_with_description():
    client = object.__new__(WebSearchClient)
    client._max_results = 5
    client._max_result_characters = 1_000
    client._max_context_characters = 4_000

    result = client._normalize_response({
        'results': {
            'web': [{
                'title': '<b>Weather</b>',
                'description': '  Sunny\nwith clouds. ',
                'snippets': ['unused snippet'],
                'url': 'https://example.test/weather',
            }],
        },
        'metadata': {},
    })

    assert result == {'results': [{
        'title': 'Weather',
        'url': 'https://example.test/weather',
        'description': 'Sunny with clouds.',
        'snippets': ['unused snippet'],
    }]}


def test_web_client_normalizes_you_com_snippets_when_description_is_empty():
    client = object.__new__(WebSearchClient)
    client._max_results = 5
    client._max_result_characters = 1_000
    client._max_context_characters = 4_000

    result = client._normalize_response({
        'results': {
            'web': [{
                'title': 'Weather',
                'description': ' ',
                'snippets': [' <p>Warm today.</p> ', 'Humidity is low.'],
                'url': 'https://example.test/weather',
            }],
        },
        'metadata': {},
    })

    assert result['results'][0]['title'] == 'Weather'
    assert 'description' not in result['results'][0]
    assert result['results'][0]['snippets'] == ['Warm today.', 'Humidity is low.']


@pytest.mark.parametrize(
    'payload',
    [
        {'results': {'web': []}, 'metadata': {}},
        {'results': {}, 'metadata': {}},
        {'metadata': {}},
    ],
)
def test_web_client_rejects_missing_or_empty_you_com_web_results(payload):
    client = object.__new__(WebSearchClient)
    client._max_results = 5
    client._max_result_characters = 1_000
    client._max_context_characters = 4_000
    with pytest.raises(WebSearchError):
        client._normalize_response(payload)


@pytest.mark.asyncio
@pytest.mark.parametrize('answer', ['', '   '])
async def test_blank_llm_response_uses_safe_fallback(answer):
    class Gateway:
        async def generate(self, prompt, *, metadata=None):
            return answer

    result = await ResponseGeneratorNode(PromptBuilder(), Gateway())(make_state('Explain machine learning.'))
    assert result['final_response'].startswith("I couldn't come up with an answer")


@pytest.mark.asyncio
async def test_llm_failure_uses_safe_fallback():
    class Gateway:
        async def generate(self, prompt, *, metadata=None):
            raise ModelGatewayError('down')

    result = await ResponseGeneratorNode(PromptBuilder(), Gateway())(make_state('Explain machine learning.'))
    assert result['final_response'].startswith("I couldn't come up with an answer")


@pytest.mark.asyncio
async def test_planner_gateway_failure_returns_general_chat():
    class Gateway:
        async def plan(self, message, *, prompt=None):
            raise ModelGatewayError('down')

    service = PlannerService(PromptBuilder(), type('Prompts', (), {'get_prompt_template': lambda self: 'planner'})(), type('Parser', (), {})(), Gateway())
    result = await service.plan('What is the latest news?')
    assert result.intent == 'general.chat'
    assert result.requires_tool is False
