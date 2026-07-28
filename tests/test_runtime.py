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
PLANNER_REGISTRY = """\
domain: vidhya
service: academic
server: vidhya-mcp
protocol: mcp
transport: streamable-http
tools:
  - id: vidhya.academic.academic_year.list
    name: academic.get_all_academic_years_by_branch_id
    entity: academic_year
    operation: list
    description: Fetches academic years.
    capability: academic.academic_year.list
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
  - id: vidhya.academic.holiday.list
    name: academic.get_all_holi_days
    entity: holiday
    operation: list
    description: Fetches holidays.
    capability: academic.holiday.list
    required_parameters:
    - academic_year_id
    optional_parameters: []
    parameter_dependencies:
    - parameter: academic_year_id
      source:
        entity: academic_year
        operation: list
      path: $.data[?(@.isCurrentAcademicYear==true)].referenceId
    response_type: structured
    version: 1.0.0
    status: active
    input:
      schema_summary:
        academic_year_id: str
        context: VidhyaRequestContext
    output:
      type: json
  - id: vidhya.academic.subject.create
    name: academic.create_subject
    entity: subject
    operation: create
    description: Creates a subject.
    capability: academic.subject.create
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
os.environ['AI_RUNTIME_JWT_SECRET'] = TEST_SECRET
os.environ['AI_RUNTIME_JWT_ALGORITHM'] = 'HS256'


class StubModelGatewayClient(ModelGatewayClient):
    def __init__(self):
        pass

    async def generate(self, prompt: str, *, metadata=None) -> str:
        return 'stubbed response'

    async def plan(self, query: str, *, prompt: str | None = None) -> dict[str, object]:
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


def create_planner_registry_service():
    from app.planner.registry_validator import PlannerRegistryValidator
    from app.tool_registry.service import ToolRegistryService

    root = Path(tempfile.mkdtemp(prefix='ai_runtime_planner_registry_'))
    registry_file = root / 'vidhya' / 'academic.yaml'
    registry_file.parent.mkdir(parents=True, exist_ok=True)
    registry_file.write_text(PLANNER_REGISTRY, encoding='utf-8')
    service = ToolRegistryService()
    service.initialize(root)
    return service, PlannerRegistryValidator(service)


class StaticPlannerGateway:
    def __init__(self, response: dict[str, object]) -> None:
        self._response = response

    async def plan(self, query: str, *, prompt: str | None = None) -> dict[str, object]:
        return self._response


async def run_registry_aware_planner(response: dict[str, object]):
    _, validator = create_planner_registry_service()
    planner = PlannerService(
        PromptBuilder(),
        PlannerPromptProvider(),
        PlannerOutputParser(),
        StaticPlannerGateway(response),
        validator,
    )
    return await planner.plan('test query')


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


@pytest.mark.asyncio
async def test_registry_aware_planner_accepts_valid_output():
    result = await run_registry_aware_planner(
        {
            'intent': 'academic.academic_year.list',
            'domain': 'vidhya',
            'service': 'academic',
            'entity': 'academic_year',
            'operation': 'list',
            'parameters': {},
            'requiresTool': True,
            'adapter': 'academic',
        }
    )

    assert result.intent == 'academic.academic_year.list'
    assert result.service == 'academic'
    assert result.entity == 'academic_year'
    assert result.operation == 'list'


@pytest.mark.asyncio
async def test_registry_aware_planner_derives_missing_service():
    result = await run_registry_aware_planner(
        {
            'intent': 'academic.holiday.list',
            'domain': 'vidhya',
            'service': None,
            'entity': 'holiday',
            'operation': 'list',
            'parameters': {},
            'requiresTool': True,
            'adapter': 'academic',
        }
    )

    assert result.service == 'academic'
    assert result.intent == 'academic.holiday.list'


@pytest.mark.asyncio
async def test_registry_aware_planner_derives_missing_intent():
    result = await run_registry_aware_planner(
        {
            'intent': None,
            'domain': 'vidhya',
            'service': 'academic',
            'entity': 'holiday',
            'operation': 'list',
            'parameters': {},
            'requiresTool': True,
            'adapter': 'academic',
        }
    )

    assert result.intent == 'academic.holiday.list'


@pytest.mark.asyncio
async def test_registry_aware_planner_resolves_entity_alias():
    result = await run_registry_aware_planner(
        {
            'intent': 'academic.academic.list',
            'domain': 'vidhya',
            'service': 'academic',
            'entity': 'academic',
            'operation': 'list',
            'parameters': {},
            'requiresTool': True,
            'adapter': 'academic',
        }
    )

    assert result.entity == 'academic_year'
    assert result.intent == 'academic.academic_year.list'


@pytest.mark.asyncio
async def test_registry_aware_planner_normalizes_entity_plural_spacing():
    result = await run_registry_aware_planner(
        {
            'intent': 'academic.academic years.list',
            'domain': 'vidhya',
            'service': 'academic',
            'entity': 'academic years',
            'operation': 'list',
            'parameters': {},
            'requiresTool': True,
            'adapter': 'academic',
        }
    )

    assert result.entity == 'academic_year'
    assert result.intent == 'academic.academic_year.list'


@pytest.mark.asyncio
async def test_registry_aware_planner_rejects_invalid_entity():
    with pytest.raises(Exception) as exc_info:
        await run_registry_aware_planner(
            {
                'intent': 'academic.fee.list',
                'domain': 'vidhya',
                'service': 'academic',
                'entity': 'fee',
                'operation': 'list',
                'parameters': {},
                'requiresTool': True,
                'adapter': 'academic',
            }
        )

    assert 'does not match any unique registered tool target' in str(exc_info.value)


@pytest.mark.asyncio
async def test_registry_aware_planner_rejects_invalid_operation():
    with pytest.raises(Exception) as exc_info:
        await run_registry_aware_planner(
            {
                'intent': 'academic.holiday.remove',
                'domain': 'vidhya',
                'service': 'academic',
                'entity': 'holiday',
                'operation': 'remove',
                'parameters': {},
                'requiresTool': True,
                'adapter': 'academic',
            }
        )

    assert 'does not match any unique registered tool target' in str(exc_info.value)


@pytest.mark.asyncio
async def test_registry_aware_planner_rejects_unknown_entity():
    with pytest.raises(Exception):
        await run_registry_aware_planner(
            {
                'intent': 'academic.transport.list',
                'domain': 'vidhya',
                'service': 'academic',
                'entity': 'transport',
                'operation': 'list',
                'parameters': {},
                'requiresTool': True,
                'adapter': 'academic',
            }
        )


@pytest.mark.asyncio
async def test_registry_aware_planner_rejects_unknown_service():
    with pytest.raises(Exception):
        await run_registry_aware_planner(
            {
                'intent': 'finance.holiday.list',
                'domain': 'vidhya',
                'service': 'finance',
                'entity': 'holiday',
                'operation': 'list',
                'parameters': {},
                'requiresTool': True,
                'adapter': 'academic',
            }
        )


@pytest.mark.asyncio
async def test_registry_aware_planner_parses_trailing_llm_text_after_json():
    result = await run_registry_aware_planner(
        {
            'rawResponse': (
                '{"domain":"vidhya","service":null,"entity":"holiday","operation":"list",'
                '"intent":null,"requiresTool":true,"adapter":"academic","parameters":{}}'
                '\nThe selected tool lists holidays.'
            ),
            'requiresTool': True,
            'adapter': 'academic',
        }
    )

    assert result.service == 'academic'
    assert result.entity == 'holiday'
    assert result.intent == 'academic.holiday.list'


@pytest.mark.asyncio
async def test_registry_aware_planner_rejects_malformed_planner_output():
    with pytest.raises(Exception) as exc_info:
        await run_registry_aware_planner(
            {
                'rawResponse': 'Here is the JSON: {not-valid-json',
                'requiresTool': True,
                'adapter': 'academic',
            }
        )

    assert 'did not include any registry target fields' in str(exc_info.value)


@pytest.mark.asyncio
async def test_registry_aware_planner_validates_multi_step_execution_plan():
    result = await run_registry_aware_planner(
        {
            'intent': 'academic.holiday.list',
            'domain': 'vidhya',
            'service': 'academic',
            'entity': 'holiday',
            'operation': 'list',
            'parameters': {},
            'requiresTool': True,
            'adapter': 'academic',
            'executionPlan': [
                {
                    'step_id': 'step_1',
                    'domain': 'vidhya',
                    'service': 'academic',
                    'entity': 'academic year',
                    'operation': 'list',
                    'parameters': {},
                },
                {
                    'step_id': 'step_2',
                    'domain': 'vidhya',
                    'service': 'academic',
                    'entity': 'holiday',
                    'operation': 'list',
                    'parameters': {},
                    'depends_on': ['step_1'],
                    'parameter_bindings': {
                        'academic_year_id': {
                            'from_step': 'step_1',
                            'path': '$.data[?(@.isCurrentAcademicYear==true)].referenceId',
                        }
                    },
                },
            ],
        }
    )

    assert result.intent == 'academic.holiday.list'
    assert result.entity == 'holiday'
    assert result.execution_plan[0].entity == 'academic_year'
    assert result.execution_plan[1].entity == 'holiday'
    assert result.execution_plan[1].parameter_bindings['academic_year_id']['from_step'] == 'step_1'


@pytest.mark.asyncio
async def test_registry_aware_planner_expands_single_step_dependency_from_registry_metadata():
    result = await run_registry_aware_planner(
        {
            'intent': 'academic.holiday.list',
            'domain': 'vidhya',
            'service': 'academic',
            'entity': 'holiday',
            'operation': 'list',
            'parameters': {},
            'requiresTool': True,
            'adapter': 'academic',
        }
    )

    assert result.intent == 'academic.holiday.list'
    assert len(result.execution_plan) == 2
    assert result.execution_plan[0].intent == 'academic.academic_year.list'
    assert result.execution_plan[0].entity == 'academic_year'
    assert result.execution_plan[0].operation == 'list'
    assert result.execution_plan[1].intent == 'academic.holiday.list'
    assert result.execution_plan[1].entity == 'holiday'
    assert result.execution_plan[1].depends_on == ['step_1']
    assert result.execution_plan[1].parameter_bindings == {
        'academic_year_id': {
            'from_step': 'step_1',
            'path': '$.data[?(@.isCurrentAcademicYear==true)].referenceId',
        }
    }


@pytest.mark.asyncio
async def test_planner_and_executor_run_dependency_plan_end_to_end():
    from app.mcp_client.models.response import InvocationStatus, ToolResponse
    from app.models.runtime import RuntimeContext
    from app.tool_executor.service import ToolExecutorService

    registry_service, validator = create_planner_registry_service()
    planner = PlannerService(
        PromptBuilder(),
        PlannerPromptProvider(),
        PlannerOutputParser(),
        StaticPlannerGateway(
            {
                'intent': 'academic.holiday.list',
                'domain': 'vidhya',
                'service': 'academic',
                'entity': 'holiday',
                'operation': 'list',
                'parameters': {},
                'requiresTool': True,
                'adapter': 'academic',
            }
        ),
        validator,
    )
    calls = []

    class FakeMCPClient:
        async def invoke_tool(self, server, tool_name, arguments, *, context, trace_id, request_id):
            calls.append({'tool_name': tool_name, 'arguments': arguments})
            if tool_name == 'academic.get_all_academic_years_by_branch_id':
                text = (
                    '{"data":['
                    '{"referenceId":"year-old","isCurrentAcademicYear":false},'
                    '{"referenceId":"year-current","isCurrentAcademicYear":true}'
                    ']}'
                )
            else:
                text = '{"data":[{"holidayName":"Founders Day"}]}'
            return ToolResponse(
                success=True,
                status=InvocationStatus.SUCCESS,
                execution_time=0.1,
                tool_results={'content': [{'type': 'text', 'text': text}]},
            )

    planner_output = await planner.plan('Give me the holidays for the current academic year.')
    result = await ToolExecutorService(registry_service, FakeMCPClient()).execute(
        planner_output=planner_output,
        runtime_context=RuntimeContext(subject='user-1', user_id='user-1'),
        request_id='req-1',
        correlation_id='corr-1',
        trace_id='trace-1',
    )

    assert planner_output.execution_plan
    assert result['success'] is True
    assert result['final_step_id'] == 'step_2'
    assert [call['tool_name'] for call in calls] == [
        'academic.get_all_academic_years_by_branch_id',
        'academic.get_all_holi_days',
    ]
    assert calls[1]['arguments']['academic_year_id'] == 'year-current'
    assert result['data'] == {'content': [{'type': 'text', 'text': '{"data":[{"holidayName":"Founders Day"}]}'}]}


@pytest.mark.asyncio
async def test_tool_executor_runs_multi_step_plan_and_binds_previous_output():
    from app.mcp_client.models.response import InvocationStatus, ToolResponse
    from app.models.planner import ExecutionPlanStep, PlannerOutput
    from app.models.runtime import RuntimeContext
    from app.tool_executor.service import ToolExecutorService

    registry_service, _ = create_planner_registry_service()
    calls = []

    class FakeMCPClient:
        async def invoke_tool(self, server, tool_name, arguments, *, context, trace_id, request_id):
            calls.append({'server': server, 'tool_name': tool_name, 'arguments': arguments})
            if tool_name == 'academic.get_all_academic_years_by_branch_id':
                    return ToolResponse(
                        success=True,
                        status=InvocationStatus.SUCCESS,
                        execution_time=0.1,
                        tool_results={
                        'content': [
                            {
                                'type': 'text',
                                'text': (
                                    '{"data":['
                                    '{"referenceId":"year-old","isCurrentAcademicYear":false},'
                                    '{"referenceId":"year-current","isCurrentAcademicYear":true}'
                                    ']}'
                                ),
                            }
                        ],
                        'isError': False,
                    },
                )
            return ToolResponse(
                success=True,
                status=InvocationStatus.SUCCESS,
                execution_time=0.1,
                tool_results={'content': [{'type': 'text', 'text': '{"data":[{"holidayName":"Founders Day"}]}'}]},
            )

    planner_output = PlannerOutput(
        intent='academic.holiday.list',
        requires_tool=True,
        domain='vidhya',
        service='academic',
        entity='holiday',
        operation='list',
        execution_plan=[
            ExecutionPlanStep(
                step_id='step_1',
                intent='academic.academic_year.list',
                domain='vidhya',
                service='academic',
                entity='academic_year',
                operation='list',
            ),
            ExecutionPlanStep(
                step_id='step_2',
                intent='academic.holiday.list',
                domain='vidhya',
                service='academic',
                entity='holiday',
                operation='list',
                depends_on=['step_1'],
                parameter_bindings={
                    'academic_year_id': {
                        'from_step': 'step_1',
                        'path': '$.data[?(@.isCurrentAcademicYear==true)].referenceId',
                    }
                },
            ),
        ],
    )

    result = await ToolExecutorService(registry_service, FakeMCPClient()).execute(
        planner_output=planner_output,
        runtime_context=RuntimeContext(subject='user-1', user_id='user-1'),
        request_id='req-1',
        correlation_id='corr-1',
        trace_id='trace-1',
    )

    assert result['success'] is True
    assert result['final_step_id'] == 'step_2'
    assert len(result['steps']) == 2
    assert calls[0]['tool_name'] == 'academic.get_all_academic_years_by_branch_id'
    assert calls[1]['tool_name'] == 'academic.get_all_holi_days'
    assert calls[1]['arguments']['academic_year_id'] == 'year-current'


@pytest.mark.asyncio
async def test_tool_executor_keeps_single_step_backward_compatibility():
    from app.mcp_client.models.response import InvocationStatus, ToolResponse
    from app.models.planner import PlannerOutput
    from app.models.runtime import RuntimeContext
    from app.tool_executor.service import ToolExecutorService

    registry_service, _ = create_planner_registry_service()
    calls = []

    class FakeMCPClient:
        async def invoke_tool(self, server, tool_name, arguments, *, context, trace_id, request_id):
            calls.append({'server': server, 'tool_name': tool_name, 'arguments': arguments})
            return ToolResponse(
                success=True,
                status=InvocationStatus.SUCCESS,
                execution_time=0.1,
                tool_results={'content': [{'type': 'text', 'text': '{"data":[]}'}]},
            )

    result = await ToolExecutorService(registry_service, FakeMCPClient()).execute(
        planner_output=PlannerOutput(
            intent='academic.academic_year.list',
            requires_tool=True,
            domain='vidhya',
            service='academic',
            entity='academic_year',
            operation='list',
        ),
        runtime_context=RuntimeContext(subject='user-1', user_id='user-1'),
        request_id='req-1',
        correlation_id='corr-1',
        trace_id='trace-1',
    )

    assert result['success'] is True
    assert result['tool_name'] == 'academic.get_all_academic_years_by_branch_id'
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_tool_executor_validates_single_step_required_parameters_before_mcp_call():
    from app.models.planner import PlannerOutput
    from app.models.runtime import RuntimeContext
    from app.tool_executor.service import ToolExecutorService

    registry_service, _ = create_planner_registry_service()
    calls = []

    class FakeMCPClient:
        async def invoke_tool(self, *args, **kwargs):
            calls.append(args)

    result = await ToolExecutorService(registry_service, FakeMCPClient()).execute(
        planner_output=PlannerOutput(
            intent='academic.holiday.list',
            requires_tool=True,
            domain='vidhya',
            service='academic',
            entity='holiday',
            operation='list',
            parameters={'academic_year_id': None},
        ),
        runtime_context=RuntimeContext(subject='user-1', user_id='user-1'),
        request_id='req-1',
        correlation_id='corr-1',
        trace_id='trace-1',
    )

    assert result['success'] is False
    assert result['error']['code'] == 'DEPENDENCY_RESOLUTION_ERROR'
    assert result['error']['data']['code'] == 'MISSING_REQUIRED_PARAMETERS'
    assert result['error']['data']['required_parameters'] == ['academic_year_id']
    assert calls == []


@pytest.mark.asyncio
async def test_tool_executor_runs_three_step_dependency_plan():
    from app.mcp_client.models.response import InvocationStatus, ToolResponse
    from app.models.planner import ExecutionPlanStep, PlannerOutput
    from app.models.runtime import RuntimeContext
    from app.tool_executor.service import ToolExecutorService

    registry_service, _ = create_planner_registry_service()
    calls = []

    class FakeMCPClient:
        async def invoke_tool(self, server, tool_name, arguments, *, context, trace_id, request_id):
            calls.append({'tool_name': tool_name, 'arguments': arguments})
            if len(calls) == 1:
                text = '{"data":[{"referenceId":"year-current","isCurrentAcademicYear":true}]}'
            elif len(calls) == 2:
                text = '{"data":[{"referenceId":"holiday-derived"}]}'
            else:
                text = '{"data":[{"holidayName":"Final Holiday"}]}'
            return ToolResponse(
                success=True,
                status=InvocationStatus.SUCCESS,
                execution_time=0.1,
                tool_results={'content': [{'type': 'text', 'text': text}]},
            )

    planner_output = PlannerOutput(
        intent='academic.holiday.list',
        requires_tool=True,
        execution_plan=[
            ExecutionPlanStep(
                step_id='step_1',
                intent='academic.academic_year.list',
                domain='vidhya',
                service='academic',
                entity='academic_year',
                operation='list',
            ),
            ExecutionPlanStep(
                step_id='step_2',
                intent='academic.holiday.list',
                domain='vidhya',
                service='academic',
                entity='holiday',
                operation='list',
                parameter_bindings={
                    'academic_year_id': {
                        'from_step': 'step_1',
                        'path': '$.data[?(@.isCurrentAcademicYear==true)].referenceId',
                    }
                },
            ),
            ExecutionPlanStep(
                step_id='step_3',
                intent='academic.holiday.list',
                domain='vidhya',
                service='academic',
                entity='holiday',
                operation='list',
                depends_on=['step_2'],
                parameter_bindings={
                    'academic_year_id': {
                        'from_step': 'step_2',
                        'path': '$.data[0].referenceId',
                    }
                },
            ),
        ],
    )

    result = await ToolExecutorService(registry_service, FakeMCPClient()).execute(
        planner_output=planner_output,
        runtime_context=RuntimeContext(subject='user-1', user_id='user-1'),
        request_id='req-1',
        correlation_id='corr-1',
        trace_id='trace-1',
    )

    assert result['success'] is True
    assert result['final_step_id'] == 'step_3'
    assert [call['tool_name'] for call in calls] == [
        'academic.get_all_academic_years_by_branch_id',
        'academic.get_all_holi_days',
        'academic.get_all_holi_days',
    ]
    assert calls[1]['arguments']['academic_year_id'] == 'year-current'
    assert calls[2]['arguments']['academic_year_id'] == 'holiday-derived'


@pytest.mark.asyncio
async def test_tool_executor_returns_dependency_error_for_missing_dependency():
    from app.models.planner import ExecutionPlanStep, PlannerOutput
    from app.models.runtime import RuntimeContext
    from app.tool_executor.service import ToolExecutorService

    registry_service, _ = create_planner_registry_service()
    calls = []

    class FakeMCPClient:
        async def invoke_tool(self, *args, **kwargs):
            calls.append(args)

    result = await ToolExecutorService(registry_service, FakeMCPClient()).execute(
        planner_output=PlannerOutput(
            intent='academic.holiday.list',
            requires_tool=True,
            execution_plan=[
                ExecutionPlanStep(
                    step_id='step_2',
                    intent='academic.holiday.list',
                    domain='vidhya',
                    service='academic',
                    entity='holiday',
                    operation='list',
                    parameter_bindings={'academic_year_id': {'from_step': 'missing_step', 'path': '$.referenceId'}},
                )
            ],
        ),
        runtime_context=RuntimeContext(subject='user-1', user_id='user-1'),
        request_id='req-1',
        correlation_id='corr-1',
        trace_id='trace-1',
    )

    assert result['success'] is False
    assert result['error']['code'] == 'DEPENDENCY_RESOLUTION_ERROR'
    assert result['error']['data']['code'] == 'MISSING_DEPENDENCY'
    assert calls == []


@pytest.mark.asyncio
async def test_tool_executor_returns_dependency_error_for_circular_dependency():
    from app.models.planner import ExecutionPlanStep, PlannerOutput
    from app.models.runtime import RuntimeContext
    from app.tool_executor.service import ToolExecutorService

    registry_service, _ = create_planner_registry_service()
    calls = []

    class FakeMCPClient:
        async def invoke_tool(self, *args, **kwargs):
            calls.append(args)

    result = await ToolExecutorService(registry_service, FakeMCPClient()).execute(
        planner_output=PlannerOutput(
            intent='academic.holiday.list',
            requires_tool=True,
            execution_plan=[
                ExecutionPlanStep(
                    step_id='step_1',
                    intent='academic.academic_year.list',
                    domain='vidhya',
                    service='academic',
                    entity='academic_year',
                    operation='list',
                    depends_on=['step_2'],
                ),
                ExecutionPlanStep(
                    step_id='step_2',
                    intent='academic.holiday.list',
                    domain='vidhya',
                    service='academic',
                    entity='holiday',
                    operation='list',
                    depends_on=['step_1'],
                    parameters={'academic_year_id': 'year-current'},
                ),
            ],
        ),
        runtime_context=RuntimeContext(subject='user-1', user_id='user-1'),
        request_id='req-1',
        correlation_id='corr-1',
        trace_id='trace-1',
    )

    assert result['success'] is False
    assert result['error']['data']['code'] == 'CIRCULAR_DEPENDENCY'
    assert calls == []


@pytest.mark.asyncio
async def test_tool_executor_returns_dependency_error_for_invalid_binding_path():
    from app.mcp_client.models.response import InvocationStatus, ToolResponse
    from app.models.planner import ExecutionPlanStep, PlannerOutput
    from app.models.runtime import RuntimeContext
    from app.tool_executor.service import ToolExecutorService

    registry_service, _ = create_planner_registry_service()
    calls = []

    class FakeMCPClient:
        async def invoke_tool(self, server, tool_name, arguments, *, context, trace_id, request_id):
            calls.append(tool_name)
            return ToolResponse(
                success=True,
                status=InvocationStatus.SUCCESS,
                execution_time=0.1,
                tool_results={'content': [{'type': 'text', 'text': '{"data":[{"referenceId":"year-current"}]}'}]},
            )

    result = await ToolExecutorService(registry_service, FakeMCPClient()).execute(
        planner_output=PlannerOutput(
            intent='academic.holiday.list',
            requires_tool=True,
            execution_plan=[
                ExecutionPlanStep(
                    step_id='step_1',
                    intent='academic.academic_year.list',
                    domain='vidhya',
                    service='academic',
                    entity='academic_year',
                    operation='list',
                ),
                ExecutionPlanStep(
                    step_id='step_2',
                    intent='academic.holiday.list',
                    domain='vidhya',
                    service='academic',
                    entity='holiday',
                    operation='list',
                    parameter_bindings={'academic_year_id': {'from_step': 'step_1', 'path': '$.data[0].missing'}},
                ),
            ],
        ),
        runtime_context=RuntimeContext(subject='user-1', user_id='user-1'),
        request_id='req-1',
        correlation_id='corr-1',
        trace_id='trace-1',
    )

    assert result['success'] is False
    assert result['error']['data']['code'] == 'INVALID_PARAMETER_BINDING'
    assert calls == ['academic.get_all_academic_years_by_branch_id']


@pytest.mark.asyncio
async def test_tool_executor_returns_dependency_error_for_missing_required_parameter():
    from app.models.planner import ExecutionPlanStep, PlannerOutput
    from app.models.runtime import RuntimeContext
    from app.tool_executor.service import ToolExecutorService

    registry_service, _ = create_planner_registry_service()
    calls = []

    class FakeMCPClient:
        async def invoke_tool(self, *args, **kwargs):
            calls.append(args)

    result = await ToolExecutorService(registry_service, FakeMCPClient()).execute(
        planner_output=PlannerOutput(
            intent='academic.holiday.list',
            requires_tool=True,
            execution_plan=[
                ExecutionPlanStep(
                    step_id='step_1',
                    intent='academic.holiday.list',
                    domain='vidhya',
                    service='academic',
                    entity='holiday',
                    operation='list',
                )
            ],
        ),
        runtime_context=RuntimeContext(subject='user-1', user_id='user-1'),
        request_id='req-1',
        correlation_id='corr-1',
        trace_id='trace-1',
    )

    assert result['success'] is False
    assert result['error']['data']['code'] == 'MISSING_REQUIRED_PARAMETERS'
    assert calls == []


def test_chat_service_flow_with_bypass_enabled():
    with create_test_client(bypass_auth=True) as client:
        response = client.post('/chat', json={'message': 'Hello there'})
    assert response.status_code == 200
    body = response.json()
    assert body['answer'] == 'stubbed response'
    assert body['metadata']['trace_id']



def test_bypass_auth_uses_bearer_token_claims(token: str):
    from app.security.auth import _build_bypass_runtime_context
    from starlette.requests import Request

    scope = {
        'type': 'http',
        'method': 'POST',
        'path': '/chat',
        'headers': [],
    }
    request = Request(scope)
    request.state.request_id = 'req-1'
    request.state.correlation_id = 'corr-1'
    credentials = type('Credentials', (), {'scheme': 'Bearer', 'credentials': token})()
    settings = type(
        'S',
        (),
        {
            'local_subject': 'developer',
            'local_user_id': 'developer',
            'local_organization_id': 'yntec',
            'local_branch_id': 'main',
            'local_app_id': 'vidhya',
            'local_tenant_id': 'yn',
            'local_locale': 'en-IN',
            'local_session_id': 'local-session',
            'local_application_ids': ['hrms', 'vidhya'],
            'local_roles': ['SUPER_ADMIN'],
            'local_permissions': ['*'],
        },
    )()

    context = _build_bypass_runtime_context(request, credentials, settings)

    assert context.user_id == 'user-1'
    assert context.organization_id == 'org-1'
    assert context.branch_id == 'branch-1'
    assert context.session_id == 'session-1'
    assert context.jwt == token

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
    captured_timeouts = []

    class DummyResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class DummyClient:
        def __init__(self, *args, **kwargs):
            captured_timeouts.append(kwargs['timeout'])

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
        planner_result = asyncio.run(client.plan('show subjects', prompt='registry prompt'))
    finally:
        mg_client_module.httpx.AsyncClient = original_client

    assert result == 'gateway ok'
    assert planner_result['entity'] == 'subject'
    assert planner_result['operation'] == 'list'
    assert captured[0]['url'] == 'http://localhost:9000/generate'
    assert captured[0]['json'] == {'adapter': 'academic', 'prompt': 'hello prompt'}
    assert captured[1]['url'] == 'http://localhost:9000/planner'
    assert captured[1]['json'] == {'adapter': 'academic', 'query': 'show subjects'}
    assert captured_timeouts[0].read == 90.0
    assert captured_timeouts[1].read == 30.0


def test_model_gateway_client_can_opt_into_planner_prompt_forwarding():
    captured = []

    class DummyResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {'intent': 'general.chat', 'requiresTool': False}

    class DummyClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, json):
            captured.append(json)
            return DummyResponse()

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
                'model_gateway_send_planner_prompt': True,
            },
        )()
        client = ModelGatewayClient(settings)

        import asyncio
        asyncio.run(client.plan('show subjects', prompt='registry prompt'))
    finally:
        mg_client_module.httpx.AsyncClient = original_client

    assert captured == [{'adapter': 'academic', 'query': 'show subjects', 'prompt': 'registry prompt'}]


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


def test_chat_accepts_jwt_without_optional_app_id():
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
    assert response.status_code == 200
    assert response.json()['answer'] == 'stubbed response'


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
    assert arguments['context']['authorization'] == 'Bearer jwt-token'
    assert metadata['jwt'] == 'jwt-token'
    assert metadata['organization_id'] == 'org-1'
    assert metadata['permissions'] == ['academic:read']


@pytest.mark.asyncio
async def test_response_generator_skips_model_gateway_when_tool_failed():
    from app.graph.nodes.response_generator import ResponseGeneratorNode
    from app.graph.state import RuntimeState
    from app.models.planner import PlannerOutput
    from app.models.runtime import RuntimeContext

    class FailingModelGateway:
        async def generate(self, prompt: str, *, metadata=None) -> str:
            raise AssertionError('model gateway should not be called when tool execution failed')

    state = RuntimeState(
        conversation_id='conv-1',
        request_id='req-1',
        correlation_id='corr-1',
        runtime_context=RuntimeContext(subject='user-1', user_id='user-1'),
        user_question='Show all academic years',
        planner_output=PlannerOutput(
            intent='academic.academic_year.list',
            requires_tool=True,
            domain='vidhya',
            service='academic',
            entity='academic_year',
            operation='list',
        ),
        tool_execution_result={
            'success': False,
            'status': 'error',
            'error': {
                'message': 'MCP tool returned isError=true',
                'data': {
                    'content': [
                        {
                            'type': 'text',
                            'text': 'Downstream service returned HTTP 401 Unauthorized.',
                        }
                    ],
                    'isError': True,
                },
            },
        },
    )

    result = await ResponseGeneratorNode(PromptBuilder(), FailingModelGateway()).__call__(state)

    assert 'Downstream service returned HTTP 401 Unauthorized.' in result['final_response']
    assert result['model_response'] == result['final_response']


def test_response_prompt_uses_compact_business_data_from_mcp_text_json():
    tool_execution_result = {
        'tool_name': 'academic.get_all_academic_years_by_branch_id',
        'server': 'vidhya-mcp',
        'status': 'success',
        'success': True,
        'response_type': 'structured',
        'registry_lookup_latency_ms': 0.44,
        'tool_execution_latency_ms': 1409.18,
        'data': {
            'content': [
                {
                    'type': 'text',
                    'text': (
                        '{"statusCode":200,"message":"Fetched academic years",'
                        '"errors":[],"error":false,"data":['
                        '{"academicYear":"2026-2027","monthlyDueDate":15},'
                        '{"academicYear":"2024-2025","monthlyDueDate":12}'
                        ']}'
                    ),
                }
            ],
            'isError': False,
        },
        'error': None,
    }

    prompt = PromptBuilder().build_response_prompt(
        'Show all academic years',
        'academic.academic_year.list',
        True,
        tool_execution_result,
    )

    assert (
        'Enterprise data: [{"academicYear":"2026-2027","monthlyDueDate":15},'
        '{"academicYear":"2024-2025","monthlyDueDate":12}]'
    ) in prompt
    assert 'Tool execution result' not in prompt
    assert 'tool_execution_latency_ms' not in prompt
    assert 'academic.get_all_academic_years_by_branch_id' not in prompt
    assert 'statusCode' not in prompt


def test_response_prompt_handles_already_structured_enterprise_data():
    tool_execution_result = {
        'success': True,
        'data': {
            'subjects': [
                {'name': 'Math', 'code': 'MATH'},
                {'name': 'Science', 'code': 'SCI'},
            ],
            'server': 'should-not-be-forwarded',
        },
        'server': 'vidhya-mcp',
    }

    prompt = PromptBuilder().build_response_prompt(
        'Show subjects',
        'academic.subject.list',
        True,
        tool_execution_result,
    )

    assert (
        'Enterprise data: {"subjects":[{"name":"Math","code":"MATH"},'
        '{"name":"Science","code":"SCI"}]}'
    ) in prompt
    assert 'should-not-be-forwarded' not in prompt
    assert 'vidhya-mcp' not in prompt


def test_response_prompt_preserves_business_status_fields():
    tool_execution_result = {
        'tool_name': 'academic.get_all_holidays',
        'status': 'success',
        'success': True,
        'data': {
            'content': [
                {
                    'type': 'text',
                    'text': '{"data":[{"holidayName":"Founders Day","status":"ACTIVE"}]}',
                }
            ]
        },
    }

    prompt = PromptBuilder().build_response_prompt(
        'Show holidays',
        'academic.holiday.list',
        True,
        tool_execution_result,
    )

    assert '"status":"ACTIVE"' in prompt
    assert '"status":"success"' not in prompt


def test_response_prompt_uses_final_plan_result_without_intermediate_steps():
    tool_execution_result = {
        'success': True,
        'status': 'success',
        'data': {
            'content': [
                {
                    'type': 'text',
                    'text': '{"data":[{"holidayName":"Founders Day"}]}',
                }
            ]
        },
        'steps': [
            {
                'step_id': 'step_1',
                'result': {
                    'data': {
                        'content': [
                            {
                                'type': 'text',
                                'text': '{"data":[{"referenceId":"year-current","isCurrentAcademicYear":true}]}',
                            }
                        ]
                    }
                },
            }
        ],
    }

    prompt = PromptBuilder().build_response_prompt(
        'Give me the holidays for the current academic year.',
        'academic.holiday.list',
        True,
        tool_execution_result,
    )

    assert 'Enterprise data: [{"holidayName":"Founders Day"}]' in prompt
    assert 'year-current' not in prompt
    assert 'isCurrentAcademicYear' not in prompt





def test_settings_accepts_core_gateway_jwt_secret_alias(monkeypatch):
    from app.config.settings import Settings

    monkeypatch.delenv('AI_RUNTIME_JWT_SECRET', raising=False)
    monkeypatch.setenv('JWT_SECRET', TEST_SECRET)
    settings = Settings()
    assert settings.jwt_secret == TEST_SECRET


def test_settings_accepts_model_gateway_operation_timeout_overrides(monkeypatch):
    from app.config.settings import Settings

    monkeypatch.setenv('AI_RUNTIME_MODEL_GATEWAY_PLANNER_TIMEOUT_SECONDS', '31')
    monkeypatch.setenv('AI_RUNTIME_MODEL_GATEWAY_PLANNER_READ_TIMEOUT_SECONDS', '32')
    monkeypatch.setenv('AI_RUNTIME_MODEL_GATEWAY_GENERATE_TIMEOUT_SECONDS', '91')
    monkeypatch.setenv('AI_RUNTIME_MODEL_GATEWAY_GENERATE_READ_TIMEOUT_SECONDS', '92')
    monkeypatch.setenv('AI_RUNTIME_MODEL_GATEWAY_SEND_PLANNER_PROMPT', 'true')

    settings = Settings()

    assert settings.model_gateway_planner_timeout_seconds == 31
    assert settings.model_gateway_planner_read_timeout_seconds == 32
    assert settings.model_gateway_generate_timeout_seconds == 91
    assert settings.model_gateway_generate_read_timeout_seconds == 92
    assert settings.model_gateway_send_planner_prompt is True


def test_tool_executor_keeps_identity_values_inside_context_only():
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
    )

    arguments = ToolExecutorService._build_arguments({}, context, 'req-1', 'corr-1', 'trace-1')

    assert 'branch_id' not in arguments
    assert arguments['context']['branch_id'] == 'branch-1'
    assert arguments['context']['organization_id'] == 'org-1'
    assert arguments['context']['user_id'] == 'user-1'
