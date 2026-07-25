import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import jwt
import pytest
from fastapi.testclient import TestClient

from app.config.settings import get_settings
from app.main import create_app
from app.model_gateway.client import ModelGatewayClient
from app.planner.parser import PlannerOutputParser
from app.planner.planner import PlannerService
from app.planner.prompts import PlannerPromptProvider
from app.prompts.builder import PromptBuilder
from app.security.auth import JwtService, RuntimeContextFactory
from app.services.dependencies import get_model_gateway_client

TEST_SECRET = '01234567890123456789012345678901'
TEST_REGISTRY = """\
domain: vidhya
service: academic
server: vidhya-mcp
protocol: mcp
transport: streamable-http
tools:
  - id: vidhya.academic.subject.list.get_all_subjects
    name: academic.get_all_subjects
    entity: subject
    operation: list
    description: Fetches all subjects.
    capability: academic.subject.list
    required_parameters: []
    optional_parameters: []
    response_type: structured
    version: 1.0.0
    status: active
    input:
      schema_summary:
        context: VidhyaRequestContext
    output:
      type: json
"""
TEST_MCP_SERVERS = """\
servers:
  vidhya-mcp:
    url: http://localhost:8001
    transport: streamable-http
    endpoint_path: /mcp
"""
os.environ['AI_RUNTIME_JWT_SECRET'] = TEST_SECRET
os.environ['AI_RUNTIME_JWT_ALGORITHM'] = 'HS256'


class StubModelGatewayClient(ModelGatewayClient):
    def __init__(self):
        pass

    async def generate(self, prompt: str, *, metadata=None) -> str:
        return 'stubbed response'

    async def plan(self, query: str) -> dict[str, object]:
        if 'subject' in query.lower():
            return {
                'intent': 'academic.subject.list',
                'domain': 'vidhya',
                'service': 'academic',
                'entity': 'subject',
                'operation': 'list',
                'tool': None,
                'parameters': {},
                'requiresTool': True,
                'rawResponse': '{"intent":"academic.subject.list","domain":"vidhya","service":"academic","entity":"subject","operation":"list","tool":null,"parameters":{},"requiresTool":true}',
                'adapter': 'academic',
                'model': 'test-model',
            }
        return {
            'intent': 'general.chat',
            'domain': None,
            'service': None,
            'entity': None,
            'operation': None,
            'tool': None,
            'parameters': {},
            'requiresTool': False,
            'rawResponse': '{"intent":"general.chat","tool":null,"parameters":{},"requiresTool":false}',
            'adapter': 'academic',
            'model': 'test-model',
        }


def create_test_registry() -> str:
    root = Path(tempfile.mkdtemp(prefix='ai_runtime_registry_'))
    registry_file = root / 'vidhya' / 'academic.yaml'
    registry_file.parent.mkdir(parents=True, exist_ok=True)
    registry_file.write_text(TEST_REGISTRY, encoding='utf-8')
    return str(root)


def create_test_mcp_servers() -> str:
    root = Path(tempfile.mkdtemp(prefix='ai_runtime_mcp_'))
    config_file = root / 'mcp_servers.yaml'
    config_file.write_text(TEST_MCP_SERVERS, encoding='utf-8')
    return str(config_file)


def create_test_client(*, bypass_auth: bool):
    os.environ['AI_RUNTIME_BYPASS_AUTH'] = 'true' if bypass_auth else 'false'
    os.environ['AI_RUNTIME_TOOL_REGISTRY_PATH'] = create_test_registry()
    os.environ['AI_RUNTIME_MCP_SERVERS_CONFIG_PATH'] = create_test_mcp_servers()
    os.environ['AI_RUNTIME_MODEL_GATEWAY_ADAPTER'] = 'academic'
    get_settings.cache_clear()
    app = create_app()
    app.dependency_overrides[get_model_gateway_client] = lambda: StubModelGatewayClient()
    return TestClient(app)


@pytest.fixture()
def token():
    return jwt.encode(
        {
            'sub': 'user-1',
            'userId': 'user-1',
            'organizationId': 'org-1',
            'branchId': 'branch-1',
            'appId': 'vidhya',
            'tenantId': 'tenant-1',
            'locale': 'en-IN',
            'applicationIds': ['hrms'],
            'roles': ['hr_manager'],
            'permissions': ['chat:use'],
            'sessionId': 'session-1',
        },
        TEST_SECRET,
        algorithm='HS256',
    )


def test_health_endpoint():
    with create_test_client(bypass_auth=True) as client:
        response = client.get('/health')
    assert response.status_code == 200
    assert response.json()['status'] == 'ok'


def test_ready_endpoint():
    with create_test_client(bypass_auth=True) as client:
        response = client.get('/ready')
    assert response.status_code == 200
    assert response.json()['status'] == 'ok'


def test_jwt_validation(token: str):
    service = JwtService(type('S', (), {'jwt_secret': TEST_SECRET, 'jwt_algorithm': 'HS256'})())
    claims = service.validate_token(token)
    assert claims['sub'] == 'user-1'


def test_runtime_context_factory():
    context = RuntimeContextFactory.from_claims(
        {
            'sub': 'user-1',
            'userId': 'user-2',
            'organizationId': 'org-1',
            'branchId': 'branch-1',
            'appId': 'vidhya',
            'tenantId': 'tenant-1',
            'locale': 'en-IN',
            'applicationIds': ['hrms'],
            'roles': ['hr_manager'],
            'permissions': ['chat:use'],
            'sessionId': 'session-1',
        },
        'raw-jwt-token',
    )
    assert context.user_id == 'user-2'
    assert context.tenant_id == 'tenant-1'
    assert context.branch_id == 'branch-1'
    assert context.app_id == 'vidhya'
    assert context.locale == 'en-IN'
    assert context.jwt == 'raw-jwt-token'


def test_local_development_runtime_context_factory():
    context = RuntimeContextFactory.for_local_development()
    assert context.user_id == 'developer'
    assert context.tenant_id == 'yn'
    assert context.organization_id == 'yntec'
    assert context.app_id == 'vidhya'
    assert context.roles == ['SUPER_ADMIN']
    assert context.permissions == ['*']


@pytest.mark.asyncio
async def test_planner():
    planner = PlannerService(PromptBuilder(), PlannerPromptProvider(), PlannerOutputParser(), StubModelGatewayClient())
    result = await planner.plan('Show all subjects')
    assert result.intent == 'academic.subject.list'
    assert result.requires_tool is True
    assert result.domain == 'vidhya'
    assert result.service == 'academic'
    assert result.entity == 'subject'
    assert result.operation == 'list'
    assert result.tool is None
    assert result.parameters == {}


def test_chat_service_flow_with_bypass_enabled():
    with create_test_client(bypass_auth=True) as client:
        response = client.post('/chat', json={'message': 'Hello there'})
    assert response.status_code == 200
    body = response.json()
    assert body['answer'] == 'stubbed response'
    assert body['metadata']['trace_id']


def test_chat_requires_auth_when_bypass_disabled():
    with create_test_client(bypass_auth=False) as client:
        response = client.post('/chat', json={'message': 'Hello there'})
    assert response.status_code == 401
    assert response.json()['code'] == 'UNAUTHORIZED'


def test_chat_rejects_invalid_jwt_when_bypass_disabled():
    with create_test_client(bypass_auth=False) as client:
        response = client.post(
            '/chat',
            headers={'Authorization': 'Bearer invalid.token.value'},
            json={'message': 'Hello there'},
        )
    assert response.status_code == 401
    assert response.json()['code'] == 'UNAUTHORIZED'


def test_chat_succeeds_with_valid_jwt_when_bypass_disabled(token: str):
    with create_test_client(bypass_auth=False) as client:
        response = client.post(
            '/chat',
            headers={'Authorization': f'Bearer {token}'},
            json={'message': 'Hello there'},
        )
    assert response.status_code == 200
    assert response.json()['answer'] == 'stubbed response'


def test_model_gateway_client_request_contract():
    captured = []

    class DummyResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class DummyClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, json):
            captured.append({'url': url, 'json': json})
            if url.endswith('/planner'):
                return DummyResponse(
                    {
                        'intent': 'academic.subject.list',
                        'domain': 'vidhya',
                        'service': 'academic',
                        'entity': 'subject',
                        'operation': 'list',
                        'tool': None,
                        'parameters': {},
                        'requiresTool': True,
                        'rawResponse': '{}',
                        'adapter': 'academic',
                        'model': 'test-model',
                    }
                )
            return DummyResponse({'response': 'gateway ok'})

    import app.model_gateway.client as mg_client_module

    original_client = mg_client_module.httpx.AsyncClient
    mg_client_module.httpx.AsyncClient = DummyClient
    try:
        settings = type(
            'Settings',
            (),
            {
                'model_gateway_url': 'http://localhost:9000',
                'model_gateway_chat_path': '/generate',
                'model_gateway_planner_path': '/planner',
                'model_gateway_adapter': 'academic',
                'model_gateway_timeout_seconds': 30.0,
                'model_gateway_connect_timeout_seconds': 5.0,
                'model_gateway_read_timeout_seconds': 30.0,
                'model_gateway_max_retries': 0,
            },
        )()
        client = ModelGatewayClient(settings)

        import asyncio
        result = asyncio.run(client.generate('hello prompt', metadata={'trace_id': 't1'}))
        planner_result = asyncio.run(client.plan('show subjects'))
    finally:
        mg_client_module.httpx.AsyncClient = original_client

    assert result == 'gateway ok'
    assert planner_result['entity'] == 'subject'
    assert planner_result['operation'] == 'list'
    assert captured[0]['url'] == 'http://localhost:9000/generate'
    assert captured[0]['json'] == {'adapter': 'academic', 'prompt': 'hello prompt'}
    assert captured[1]['url'] == 'http://localhost:9000/planner'
    assert captured[1]['json'] == {'adapter': 'academic', 'query': 'show subjects'}


@pytest.mark.asyncio
async def test_model_gateway_client_error_mapping():
    class Settings:
        model_gateway_timeout_seconds = 0.01
        model_gateway_connect_timeout_seconds = 0.01
        model_gateway_read_timeout_seconds = 0.01
        model_gateway_max_retries = 0
        model_gateway_url = 'http://127.0.0.1:9'
        model_gateway_chat_path = '/generate'
        model_gateway_planner_path = '/planner'
        model_gateway_adapter = 'academic'

    client = ModelGatewayClient(Settings())
    with pytest.raises(Exception):
        await client.generate('hi')



def test_chat_accepts_gateway_forwarded_headers_matching_jwt(token: str):
    with create_test_client(bypass_auth=False) as client:
        response = client.post(
            '/chat',
            headers={
                'Authorization': f'Bearer {token}',
                'X-USER-Id': 'user-1',
                'X-ORG-Id': 'org-1',
                'X-BRANCH-Id': 'branch-1',
                'X-APP-Id': 'vidhya',
                'X-Request-Id': 'req-1',
                'X-Correlation-Id': 'corr-1',
                'X-Trace-Id': 'trace-1',
            },
            json={'message': 'Hello there'},
        )
    assert response.status_code == 200
    assert response.json()['metadata']['request_id'] == 'req-1'
    assert response.json()['metadata']['correlation_id'] == 'corr-1'
    assert response.json()['metadata']['trace_id'] == 'trace-1'


def test_chat_rejects_gateway_header_mismatch(token: str):
    with create_test_client(bypass_auth=False) as client:
        response = client.post(
            '/chat',
            headers={
                'Authorization': f'Bearer {token}',
                'X-ORG-Id': 'other-org',
            },
            json={'message': 'Hello there'},
        )
    assert response.status_code == 401
    assert response.json()['code'] == 'UNAUTHORIZED'


def test_chat_rejects_missing_required_jwt_claim():
    token = jwt.encode(
        {
            'sub': 'user-1',
            'organizationId': 'org-1',
            'branchId': 'branch-1',
        },
        TEST_SECRET,
        algorithm='HS256',
    )
    with create_test_client(bypass_auth=False) as client:
        response = client.post(
            '/chat',
            headers={'Authorization': f'Bearer {token}'},
            json={'message': 'Hello there'},
        )
    assert response.status_code == 401
    assert response.json()['code'] == 'UNAUTHORIZED'


def test_chat_rejects_expired_jwt():
    token = jwt.encode(
        {
            'sub': 'user-1',
            'organizationId': 'org-1',
            'branchId': 'branch-1',
            'appId': 'vidhya',
            'exp': datetime.now(timezone.utc) - timedelta(minutes=1),
        },
        TEST_SECRET,
        algorithm='HS256',
    )
    with create_test_client(bypass_auth=False) as client:
        response = client.post(
            '/chat',
            headers={'Authorization': f'Bearer {token}'},
            json={'message': 'Hello there'},
        )
    assert response.status_code == 401
    assert response.json()['code'] == 'UNAUTHORIZED'


def test_chat_rejects_invalid_signature():
    token = jwt.encode(
        {
            'sub': 'user-1',
            'organizationId': 'org-1',
            'branchId': 'branch-1',
            'appId': 'vidhya',
        },
        'wrong-secret-012345678901234567890123',
        algorithm='HS256',
    )
    with create_test_client(bypass_auth=False) as client:
        response = client.post(
            '/chat',
            headers={'Authorization': f'Bearer {token}'},
            json={'message': 'Hello there'},
        )
    assert response.status_code == 401
    assert response.json()['code'] == 'UNAUTHORIZED'


def test_tool_executor_builds_vidhya_context_from_runtime_context():
    from app.models.runtime import RuntimeContext
    from app.tool_executor.service import ToolExecutorService

    context = RuntimeContext(
        subject='user-1',
        user_id='user-1',
        organization_id='org-1',
        branch_id='branch-1',
        app_id='vidhya',
        tenant_id='tenant-1',
        locale='en-IN',
        application_ids=['vidhya'],
        roles=['admin'],
        permissions=['academic:read'],
        session_id='session-1',
        jwt='jwt-token',
    )
    arguments = ToolExecutorService._build_arguments({}, context, 'req-1', 'corr-1', 'trace-1')
    metadata = ToolExecutorService._build_context(context, 'req-1', 'corr-1', 'trace-1')

    assert arguments['context']['organization_id'] == 'org-1'
    assert arguments['context']['branch_id'] == 'branch-1'
    assert arguments['context']['user_id'] == 'user-1'
    assert arguments['context']['tenant_id'] == 'tenant-1'
    assert metadata['jwt'] == 'jwt-token'
    assert metadata['organization_id'] == 'org-1'
    assert metadata['permissions'] == ['academic:read']


