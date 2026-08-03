import pytest

from app.mcp_client.models.response import InvocationStatus, ToolResponse
from app.models.execution import ExecutionState
from app.models.planner import ExecutionPlanStep, PlannerOutput
from app.models.runtime import RuntimeContext
from app.tool_executor.service import ToolExecutorService
from app.tool_registry.models import InputDefinition, OutputDefinition, ResolvedTool, ToolDefinition


def make_tool(
    name: str,
    entity: str,
    operation: str,
    required: list[str],
    *,
    identifier_fields: list[str] | None = None,
    display_fields: list[str] | None = None,
) -> ResolvedTool:
    return ResolvedTool(
        domain='vidhya',
        service='academic',
        server='vidhya-academic',
        protocol='mcp',
        transport='http',
        tool=ToolDefinition(
            id=f'vidhya.academic.{entity}.{operation}.{name}',
            name=name,
            entity=entity,
            operation=operation,
            description=f'{operation} {entity}',
            capability=f'academic.{entity}.{operation}',
            required_parameters=required,
            optional_parameters=[],
            response_type='structured',
            version='1.0.0',
            status='active',
            input=InputDefinition(schema_summary={parameter: 'str' for parameter in required}),
            output=OutputDefinition(
                type='json',
                identifier_fields=identifier_fields or [],
                display_fields=display_fields or [],
            ),
        ),
    )


class FakeRegistry:
    def __init__(self) -> None:
        self._tools = {
            ('academic_year', 'list'): make_tool(
                'academic.get_all_academic_years_by_branch_id',
                'academic_year',
                'list',
                [],
                identifier_fields=['referenceId'],
                display_fields=['academicYear'],
            ),
            ('holiday', 'list'): make_tool(
                'academic.get_all_holi_days',
                'holiday',
                'list',
                ['academic_year_id'],
            ),
            ('subject', 'list'): make_tool(
                'academic.get_all_subjects',
                'subject',
                'list',
                [],
            ),
        }

    def find_tool(self, domain: str, service: str, entity: str, operation: str):
        return self._tools[(entity, operation)]


class FakeMCPClient:
    def __init__(self) -> None:
        self.calls = []

    async def invoke_tool(self, server, tool_name, arguments, *, context, trace_id, request_id):
        self.calls.append({'tool_name': tool_name, 'arguments': arguments})
        if tool_name == 'academic.get_all_academic_years_by_branch_id':
            text = (
                '{"data":['
                '{"referenceId":"year-old","academicYear":"2024-2025","isCurrentAcademicYear":false},'
                '{"referenceId":"year-current","academicYear":"2025-2026","isCurrentAcademicYear":true}'
                ']}'
            )
        elif tool_name == 'academic.get_all_holi_days':
            text = '{"data":[{"holidayName":"Founders Day"}]}'
        else:
            text = '{"data":[{"subjectName":"Mathematics"}]}'
        return ToolResponse(
            success=True,
            status=InvocationStatus.SUCCESS,
            execution_time=0.1,
            tool_results={'content': [{'type': 'text', 'text': text}]},
        )


def runtime_context() -> RuntimeContext:
    return RuntimeContext(subject='user-1', user_id='user-1')


def test_clarification_options_require_registry_identifier_metadata():
    options = ToolExecutorService._options_from_records(
        [
            {'referenceId': 'record-1', 'id': 'fallback-id', 'displayName': 'Alpha'},
        ],
        {'display_fields': ['displayName']},
    )

    assert options == []


def test_clarification_options_use_registry_identifier_and_display_metadata():
    options = ToolExecutorService._options_from_records(
        [
            {'recordKey': 'record-1', 'displayName': 'Alpha'},
        ],
        {
            'identifier_fields': ['recordKey'],
            'display': {'label_template': '{displayName}'},
        },
    )

    assert options[0].value == 'record-1'
    assert options[0].label == 'Alpha'


@pytest.mark.asyncio
async def test_execution_plan_runs_two_independent_tools_before_final_result():
    mcp = FakeMCPClient()
    planner_output = PlannerOutput(
        intent='academic.multi.list',
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
                intent='academic.subject.list',
                domain='vidhya',
                service='academic',
                entity='subject',
                operation='list',
            ),
        ],
    )

    result = await ToolExecutorService(FakeRegistry(), mcp).execute(
        planner_output=planner_output,
        runtime_context=runtime_context(),
        request_id='req-1',
        correlation_id='corr-1',
        trace_id='trace-1',
        allow_clarification=True,
    )

    assert result['success'] is True
    assert result['final_step_id'] == 'step_2'
    assert [call['tool_name'] for call in mcp.calls] == [
        'academic.get_all_academic_years_by_branch_id',
        'academic.get_all_subjects',
    ]
    assert len(result['steps']) == 2


@pytest.mark.asyncio
async def test_missing_required_parameter_pauses_without_calling_tool():
    mcp = FakeMCPClient()
    planner_output = PlannerOutput(
        intent='academic.holiday.list',
        requires_tool=True,
        domain='vidhya',
        service='academic',
        entity='holiday',
        operation='list',
    )

    result = await ToolExecutorService(FakeRegistry(), mcp).execute(
        planner_output=planner_output,
        runtime_context=runtime_context(),
        request_id='req-1',
        correlation_id='corr-1',
        trace_id='trace-1',
        allow_clarification=True,
    )

    assert result['status'] == 'requires_input'
    assert result['awaiting_input'] is True
    assert result['clarification']['missing_parameters'] == ['academic_year_id']
    assert result['clarification']['question'] == ''
    assert mcp.calls == []


@pytest.mark.asyncio
async def test_single_step_resume_after_clarification_executes_original_tool():
    planner_output = PlannerOutput(
        intent='academic.holiday.list',
        requires_tool=True,
        domain='vidhya',
        service='academic',
        entity='holiday',
        operation='list',
    )
    paused = await ToolExecutorService(FakeRegistry(), FakeMCPClient()).execute(
        planner_output=planner_output,
        runtime_context=runtime_context(),
        request_id='req-1',
        correlation_id='corr-1',
        trace_id='trace-1',
        allow_clarification=True,
    )

    resume_mcp = FakeMCPClient()
    resumed = await ToolExecutorService(FakeRegistry(), resume_mcp).execute(
        planner_output=ExecutionState.model_validate(paused['execution_state']).planner_output,
        runtime_context=runtime_context(),
        request_id='req-2',
        correlation_id='corr-2',
        trace_id='trace-2',
        execution_state=ExecutionState.model_validate(paused['execution_state']),
        clarification_answer='year-current',
        allow_clarification=True,
    )

    assert resumed['success'] is True
    assert [call['tool_name'] for call in resume_mcp.calls] == ['academic.get_all_holi_days']
    assert resume_mcp.calls[0]['arguments']['academic_year_id'] == 'year-current'


@pytest.mark.asyncio
async def test_resume_after_clarification_does_not_rerun_completed_steps():
    first_mcp = FakeMCPClient()
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
                depends_on=['step_1'],
            ),
        ],
    )

    paused = await ToolExecutorService(FakeRegistry(), first_mcp).execute(
        planner_output=planner_output,
        runtime_context=runtime_context(),
        request_id='req-1',
        correlation_id='corr-1',
        trace_id='trace-1',
        allow_clarification=True,
    )

    assert paused['status'] == 'requires_input'
    assert [option['label'] for option in paused['clarification']['options']] == [
        '2024-2025',
        '2025-2026',
    ]
    assert [call['tool_name'] for call in first_mcp.calls] == ['academic.get_all_academic_years_by_branch_id']

    resume_mcp = FakeMCPClient()
    resumed = await ToolExecutorService(FakeRegistry(), resume_mcp).execute(
        planner_output=planner_output,
        runtime_context=runtime_context(),
        request_id='req-2',
        correlation_id='corr-2',
        trace_id='trace-2',
        execution_state=ExecutionState.model_validate(paused['execution_state']),
        clarification_answer='2',
        allow_clarification=True,
    )

    assert resumed['success'] is True
    assert resumed['final_step_id'] == 'step_2'
    assert [call['tool_name'] for call in resume_mcp.calls] == ['academic.get_all_holi_days']
    assert resume_mcp.calls[0]['arguments']['academic_year_id'] == 'year-current'
    assert len(resumed['steps']) == 2
