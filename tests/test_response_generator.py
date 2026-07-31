import pytest
from datetime import date

from app.graph.nodes.response_generator import GroundedResponseComposer, ResponseGeneratorNode
from app.graph.state import RuntimeState
from app.models.planner import PlannerOutput
from app.models.runtime import RuntimeContext
from app.prompts.builder import PromptBuilder
from app.utils.temporal import TemporalIntent, normalize_temporal_intent

BULLET = '\u2022'
FIXED_DATE = date(2026, 7, 30)

ACADEMIC_YEAR_RECORDS = [
    {'academicYear': '2026-2027', 'isCurrentAcademicYear': False},
    {'academicYear': '2024-2025', 'isCurrentAcademicYear': True},
    {'academicYear': '2025-2026', 'isCurrentAcademicYear': False},
]

HOLIDAY_RECORDS = [
    {'holidayName': 'Bakrid', 'referenceId': '1', 'holidayStartDate': '2026-07-01', 'holidayEndDate': '2026-07-01', 'holidayStatus': True},
    {'holidayName': 'Founders Day', 'referenceId': '2', 'holidayStartDate': '2026-08-05', 'holidayEndDate': '2026-08-05', 'holidayStatus': True},
    {'holidayName': 'Quarter Break', 'referenceId': '3', 'holidayStartDate': '2026-07-30', 'holidayEndDate': '2026-07-31', 'holidayStatus': True},
    {'holidayName': 'Summer Break', 'referenceId': '4', 'holidayStartDate': '2026-07-25', 'holidayEndDate': '2026-08-02', 'holidayStatus': True},
    {'holidayName': 'Old Holiday', 'referenceId': '5', 'holidayStartDate': '2026-06-10', 'holidayEndDate': '2026-06-10', 'holidayStatus': True},
    {'holidayName': 'Yesterday Holiday', 'referenceId': '6', 'holidayStartDate': '2026-07-29', 'holidayEndDate': '2026-07-29', 'holidayStatus': True},
    {'holidayName': 'Tomorrow Holiday', 'referenceId': '7', 'holidayStartDate': '2026-07-31', 'holidayEndDate': '2026-07-31', 'holidayStatus': True},
]

CURRENT_ACADEMIC_YEAR = 'The current active year is 2024-2025.'
ALL_ACADEMIC_YEARS = f'Found 3 years.\n\n{BULLET} 2026-2027\n{BULLET} 2024-2025 (Current)\n{BULLET} 2025-2026'
COUNT_ACADEMIC_YEARS = 'There are 3 years configured.'
DETAIL_2025_2026 = f'Year details:\n\n{BULLET} 2025-2026'
DETAIL_2024_2025 = f'Year details:\n\n{BULLET} 2024-2025 (Current)'
MISSING_DETAIL = 'Year 2030-2031 was not found.'
NO_PREVIOUS_ACADEMIC_YEARS = 'There are no previous years.'
THIS_MONTH_HOLIDAYS = f'Found 5 holidays this month.\n\n{BULLET} Bakrid, Start Date: 2026-07-01\n{BULLET} Summer Break, Start Date: 2026-07-25, End Date: 2026-08-02\n{BULLET} Yesterday Holiday, Start Date: 2026-07-29\n{BULLET} Quarter Break, Start Date: 2026-07-30, End Date: 2026-07-31\n{BULLET} Tomorrow Holiday, Start Date: 2026-07-31'
PREVIOUS_MONTH_HOLIDAYS = f'Found 1 holidays the previous month.\n\n{BULLET} Old Holiday, Start Date: 2026-06-10'
NEXT_MONTH_HOLIDAYS = f'Found 2 holidays next month.\n\n{BULLET} Summer Break, Start Date: 2026-07-25, End Date: 2026-08-02\n{BULLET} Founders Day, Start Date: 2026-08-05'
TODAY_HOLIDAYS = f'Found 2 holidays today.\n\n{BULLET} Summer Break, Start Date: 2026-07-25, End Date: 2026-08-02\n{BULLET} Quarter Break, Start Date: 2026-07-30, End Date: 2026-07-31'
TOMORROW_HOLIDAYS = f'Found 3 holidays tomorrow.\n\n{BULLET} Summer Break, Start Date: 2026-07-25, End Date: 2026-08-02\n{BULLET} Quarter Break, Start Date: 2026-07-30, End Date: 2026-07-31\n{BULLET} Tomorrow Holiday, Start Date: 2026-07-31'
YESTERDAY_HOLIDAYS = f'Found 2 holidays yesterday.\n\n{BULLET} Summer Break, Start Date: 2026-07-25, End Date: 2026-08-02\n{BULLET} Yesterday Holiday, Start Date: 2026-07-29'
THIS_WEEK_HOLIDAYS = f'Found 4 holidays this week.\n\n{BULLET} Summer Break, Start Date: 2026-07-25, End Date: 2026-08-02\n{BULLET} Yesterday Holiday, Start Date: 2026-07-29\n{BULLET} Quarter Break, Start Date: 2026-07-30, End Date: 2026-07-31\n{BULLET} Tomorrow Holiday, Start Date: 2026-07-31'
PREVIOUS_WEEK_HOLIDAYS = f'Found 1 holidays the previous week.\n\n{BULLET} Summer Break, Start Date: 2026-07-25, End Date: 2026-08-02'
NEXT_WEEK_HOLIDAYS = f'Found 1 holidays next week.\n\n{BULLET} Founders Day, Start Date: 2026-08-05'
THIS_YEAR_HOLIDAYS = f'Found 7 holidays this year.\n\n{BULLET} Old Holiday, Start Date: 2026-06-10\n{BULLET} Bakrid, Start Date: 2026-07-01\n{BULLET} Summer Break, Start Date: 2026-07-25, End Date: 2026-08-02\n{BULLET} Yesterday Holiday, Start Date: 2026-07-29\n{BULLET} Quarter Break, Start Date: 2026-07-30, End Date: 2026-07-31\n{BULLET} Tomorrow Holiday, Start Date: 2026-07-31\n{BULLET} Founders Day, Start Date: 2026-08-05'
NO_PREVIOUS_YEAR_HOLIDAYS = 'There are no holidays scheduled for the previous year.'
NO_NEXT_YEAR_HOLIDAYS = 'There are no holidays scheduled for next year.'
UPCOMING_HOLIDAYS = f'Found 4 upcoming holidays.\n\n{BULLET} Summer Break, Start Date: 2026-07-25, End Date: 2026-08-02\n{BULLET} Quarter Break, Start Date: 2026-07-30, End Date: 2026-07-31\n{BULLET} Tomorrow Holiday, Start Date: 2026-07-31\n{BULLET} Founders Day, Start Date: 2026-08-05'
PAST_HOLIDAYS = f'Found 3 previous holidays.\n\n{BULLET} Old Holiday, Start Date: 2026-06-10\n{BULLET} Bakrid, Start Date: 2026-07-01\n{BULLET} Yesterday Holiday, Start Date: 2026-07-29'
LATEST_HOLIDAY = 'The latest holiday is Founders Day, Start Date: 2026-08-05.'
ALL_HOLIDAYS_SHORT = f'Found 2 holidays.\n\n{BULLET} Bakrid, Start Date: 2026-07-01\n{BULLET} Founders Day, Start Date: 2026-08-05'


def build_state(message: str, *, entity: str, records: list[dict]) -> RuntimeState:
    return RuntimeState(
        conversation_id='conv-1',
        request_id='req-1',
        correlation_id='corr-1',
        runtime_context=RuntimeContext(subject='user-1', user_id='user-1'),
        user_question=message,
        planner_output=PlannerOutput(
            intent=f'academic.{entity}.list',
            requires_tool=True,
            domain='vidhya',
            service='academic',
            entity=entity,
            operation='list',
        ),
        tool_execution_result={'success': True, 'data': {'items': records}},
    )


class NeverCalledGateway:
    async def generate(self, prompt: str, *, metadata=None) -> str:
        raise AssertionError('model gateway should not be called for deterministic formatting')


def build_node() -> ResponseGeneratorNode:
    return ResponseGeneratorNode(PromptBuilder(), NeverCalledGateway(), current_date_provider=lambda: FIXED_DATE)


def build_composer() -> GroundedResponseComposer:
    return GroundedResponseComposer(current_date_provider=lambda: FIXED_DATE)


@pytest.mark.parametrize(
    ('message', 'expected'),
    [
        ('Show holidays this month', TemporalIntent(scope='month', offset=0)),
        ('Show holidays last month', TemporalIntent(scope='month', offset=-1)),
        ('Show holidays previous month', TemporalIntent(scope='month', offset=-1)),
        ('Show holidays prior month', TemporalIntent(scope='month', offset=-1)),
        ('Show holidays past month', TemporalIntent(scope='month', offset=-1)),
        ('Show holidays next month', TemporalIntent(scope='month', offset=1)),
        ('Show holidays today', TemporalIntent(scope='day', offset=0)),
        ('Show holidays tomorrow', TemporalIntent(scope='day', offset=1)),
        ('Show holidays yesterday', TemporalIntent(scope='day', offset=-1)),
        ('Show holidays this week', TemporalIntent(scope='week', offset=0)),
        ('Show holidays previous week', TemporalIntent(scope='week', offset=-1)),
        ('Show holidays next week', TemporalIntent(scope='week', offset=1)),
        ('Show holidays this year', TemporalIntent(scope='year', offset=0)),
        ('Show holidays previous year', TemporalIntent(scope='year', offset=-1)),
        ('Show holidays next year', TemporalIntent(scope='year', offset=1)),
        ('Show upcoming holidays', TemporalIntent(scope='future', offset=0)),
        ('Show past holidays', TemporalIntent(scope='past', offset=0)),
        ('Show latest holiday', TemporalIntent(scope='latest', offset=0)),
        ('Which academic year is active now?', TemporalIntent(scope='current', offset=0)),
    ],
)
def test_temporal_intent_normalization(message: str, expected: TemporalIntent):
    assert normalize_temporal_intent(message) == expected


@pytest.mark.asyncio
async def test_existing_current_academic_year_behavior_remains_unchanged():
    result = await build_node().__call__(build_state('Which academic year is active now?', entity='academic_year', records=ACADEMIC_YEAR_RECORDS))
    assert result['final_response'] == CURRENT_ACADEMIC_YEAR


@pytest.mark.asyncio
async def test_academic_year_count_and_detail_modes():
    node = build_node()
    count = await node.__call__(build_state('How many academic years are configured?', entity='academic_year', records=ACADEMIC_YEAR_RECORDS))
    detail = await node.__call__(build_state('Show academic year 2025-2026', entity='academic_year', records=ACADEMIC_YEAR_RECORDS))
    current_detail = await node.__call__(build_state('Information about academic year 2024-2025', entity='academic_year', records=ACADEMIC_YEAR_RECORDS))
    missing = await node.__call__(build_state('Academic year details for 2030-2031', entity='academic_year', records=ACADEMIC_YEAR_RECORDS))
    list_with_current = await node.__call__(build_state('Get all academic years and identify the current one.', entity='academic_year', records=ACADEMIC_YEAR_RECORDS))

    assert count['final_response'] == COUNT_ACADEMIC_YEARS
    assert detail['final_response'] == DETAIL_2025_2026
    assert current_detail['final_response'] == DETAIL_2024_2025
    assert missing['final_response'] == MISSING_DETAIL
    assert list_with_current['final_response'] == ALL_ACADEMIC_YEARS


@pytest.mark.asyncio
async def test_existing_this_month_behavior_remains_unchanged():
    result = await build_node().__call__(build_state('Show holidays this month', entity='holiday', records=HOLIDAY_RECORDS))
    assert result['final_response'] == THIS_MONTH_HOLIDAYS


@pytest.mark.asyncio
async def test_previous_month_and_next_month_and_day_filters():
    node = build_node()
    previous_month = await node.__call__(build_state('Show holidays previous month', entity='holiday', records=HOLIDAY_RECORDS))
    next_month = await node.__call__(build_state('Show holidays next month', entity='holiday', records=HOLIDAY_RECORDS))
    today = await node.__call__(build_state('Show holidays today', entity='holiday', records=HOLIDAY_RECORDS))
    tomorrow = await node.__call__(build_state('Show holidays tomorrow', entity='holiday', records=HOLIDAY_RECORDS))
    yesterday = await node.__call__(build_state('Show holidays yesterday', entity='holiday', records=HOLIDAY_RECORDS))

    assert previous_month['final_response'] == PREVIOUS_MONTH_HOLIDAYS
    assert next_month['final_response'] == NEXT_MONTH_HOLIDAYS
    assert today['final_response'] == TODAY_HOLIDAYS
    assert tomorrow['final_response'] == TOMORROW_HOLIDAYS
    assert yesterday['final_response'] == YESTERDAY_HOLIDAYS


@pytest.mark.asyncio
async def test_week_year_future_past_latest_and_empty_results():
    node = build_node()
    this_week = await node.__call__(build_state('Show holidays this week', entity='holiday', records=HOLIDAY_RECORDS))
    previous_week = await node.__call__(build_state('Show holidays previous week', entity='holiday', records=HOLIDAY_RECORDS))
    next_week = await node.__call__(build_state('Show holidays next week', entity='holiday', records=HOLIDAY_RECORDS))
    this_year = await node.__call__(build_state('Show holidays this year', entity='holiday', records=HOLIDAY_RECORDS))
    previous_year = await node.__call__(build_state('Show holidays previous year', entity='holiday', records=HOLIDAY_RECORDS))
    next_year = await node.__call__(build_state('Show holidays next year', entity='holiday', records=HOLIDAY_RECORDS))
    upcoming = await node.__call__(build_state('Show upcoming holidays', entity='holiday', records=HOLIDAY_RECORDS))
    past = await node.__call__(build_state('Show past holidays', entity='holiday', records=HOLIDAY_RECORDS))
    latest = await node.__call__(build_state('Show latest holiday', entity='holiday', records=HOLIDAY_RECORDS))

    assert this_week['final_response'] == THIS_WEEK_HOLIDAYS
    assert previous_week['final_response'] == PREVIOUS_WEEK_HOLIDAYS
    assert next_week['final_response'] == NEXT_WEEK_HOLIDAYS
    assert this_year['final_response'] == THIS_YEAR_HOLIDAYS
    assert previous_year['final_response'] == NO_PREVIOUS_YEAR_HOLIDAYS
    assert next_year['final_response'] == NO_NEXT_YEAR_HOLIDAYS
    assert upcoming['final_response'] == UPCOMING_HOLIDAYS
    assert past['final_response'] == PAST_HOLIDAYS
    assert latest['final_response'] == LATEST_HOLIDAY


@pytest.mark.asyncio
async def test_multi_ask_execution_plan_formats_independent_leaf_steps():
    node = build_node()
    state = RuntimeState(
        conversation_id='conv-1',
        request_id='req-1',
        correlation_id='corr-1',
        runtime_context=RuntimeContext(subject='user-1', user_id='user-1'),
        user_question='Get all academic years and identify the holidays.',
        planner_output=PlannerOutput(
            intent='academic.multi.list',
            requires_tool=True,
            domain='vidhya',
            service='academic',
            entity='academic_year',
            operation='list',
            execution_plan=[
                {
                    'step_id': 'step_1',
                    'intent': 'academic.academic_year.list',
                    'domain': 'vidhya',
                    'service': 'academic',
                    'entity': 'academic_year',
                    'operation': 'list',
                    'parameters': {},
                    'question': 'Get all academic years.',
                    'visible_in_response': True,
                },
                {
                    'step_id': 'step_2',
                    'intent': 'academic.holiday.list',
                    'domain': 'vidhya',
                    'service': 'academic',
                    'entity': 'holiday',
                    'operation': 'list',
                    'parameters': {},
                    'question': 'Identify the holidays.',
                    'visible_in_response': True,
                },
            ],
        ),
        tool_execution_result={
            'success': True,
            'steps': [
                {
                    'step_id': 'step_1',
                    'result': {'success': True, 'data': {'items': ACADEMIC_YEAR_RECORDS}},
                },
                {
                    'step_id': 'step_2',
                    'result': {'success': True, 'data': {'items': HOLIDAY_RECORDS[:2]}},
                },
            ],
            'data': {'items': HOLIDAY_RECORDS[:2]},
        },
    )

    result = await node.__call__(state)

    assert result['final_response'] == f'Academic years:\n{ALL_ACADEMIC_YEARS}\n\nHolidays:\n{ALL_HOLIDAYS_SHORT}'


@pytest.mark.asyncio
async def test_multi_step_dependency_chain_keeps_only_final_leaf_result():
    node = build_node()
    state = RuntimeState(
        conversation_id='conv-1',
        request_id='req-1',
        correlation_id='corr-1',
        runtime_context=RuntimeContext(subject='user-1', user_id='user-1'),
        user_question='Give me the holidays for the current academic year.',
        planner_output=PlannerOutput(
            intent='academic.holiday.list',
            requires_tool=True,
            domain='vidhya',
            service='academic',
            entity='holiday',
            operation='list',
            execution_plan=[
                {
                    'step_id': 'step_1',
                    'intent': 'academic.academic_year.list',
                    'domain': 'vidhya',
                    'service': 'academic',
                    'entity': 'academic_year',
                    'operation': 'list',
                    'parameters': {},
                    'visible_in_response': False,
                },
                {
                    'step_id': 'step_2',
                    'intent': 'academic.holiday.list',
                    'domain': 'vidhya',
                    'service': 'academic',
                    'entity': 'holiday',
                    'operation': 'list',
                    'parameters': {},
                    'question': 'Give me the holidays for the current academic year.',
                    'visible_in_response': True,
                    'depends_on': ['step_1'],
                    'parameter_bindings': {'academic_year_id': {'from_step': 'step_1', 'path': '$.data[0].referenceId'}},
                },
            ],
        ),
        tool_execution_result={
            'success': True,
            'steps': [
                {
                    'step_id': 'step_1',
                    'result': {'success': True, 'data': {'items': ACADEMIC_YEAR_RECORDS}},
                },
                {
                    'step_id': 'step_2',
                    'result': {'success': True, 'data': {'items': HOLIDAY_RECORDS[:2]}},
                },
            ],
            'data': {'items': HOLIDAY_RECORDS[:2]},
            'final_step_id': 'step_2',
        },
    )

    result = await node.__call__(state)

    assert result['final_response'] == ALL_HOLIDAYS_SHORT


def build_matrix_cases() -> list[tuple[str, str, list[dict], str]]:
    cases = []
    for message in [
        'Which academic year is active now?', 'What is the current academic year?', 'Tell me the active academic year', 'Show the selected academic year now',
        'Give me the default academic year', 'Which academic year is current?', 'Current academic year?', 'Active academic year now',
        'What is the default academic year now?', 'Show current academic year', 'Which academic year is selected now?', 'Tell me current active academic year',
        'Give active academic year', 'What is the active year now?', 'Selected academic year?'
    ]:
        cases.append((message, 'academic_year', ACADEMIC_YEAR_RECORDS, CURRENT_ACADEMIC_YEAR))
    for message in [
        'Show all academic years', 'List academic years', 'Display all academic years', 'Enumerate academic years', 'Get all academic years',
        'List all academic years', 'Show every academic year', 'Display academic years', 'Enumerate all academic years', 'Give me all academic years',
        'Show the academic year list', 'Academic year list', 'List all year records', 'Display all year records', 'Show all year records'
    ]:
        cases.append((message, 'academic_year', ACADEMIC_YEAR_RECORDS, ALL_ACADEMIC_YEARS))
    for message in ['How many academic years are configured?', 'Count academic years', 'Total academic years', 'Number of academic years']:
        cases.append((message, 'academic_year', ACADEMIC_YEAR_RECORDS, COUNT_ACADEMIC_YEARS))
    for message, expected in [
        ('Show academic year 2025-2026', DETAIL_2025_2026),
        ('Academic year details for 2025-2026', DETAIL_2025_2026),
        ('Information about academic year 2025-2026', DETAIL_2025_2026),
        ('Academic year 2025-2026', DETAIL_2025_2026),
        ('Show academic year 2024-2025', DETAIL_2024_2025),
        ('Academic year details for 2030-2031', MISSING_DETAIL),
    ]:
        cases.append((message, 'academic_year', ACADEMIC_YEAR_RECORDS, expected))
    for message in [
        'Show all previous academic years', 'List previous academic years', 'Display past academic years', 'Get last academic years', 'Show previous year records',
        'Past academic years?', 'Previous academic years', 'List all past academic years', 'Display all previous academic years', 'Give me previous academic years',
        'Show last academic years', 'Past year records', 'Previous year list', 'All previous year records', 'List prior academic years'
    ]:
        cases.append((message, 'academic_year', ACADEMIC_YEAR_RECORDS, NO_PREVIOUS_ACADEMIC_YEARS))
    for message in [
        'Show holidays this month', 'List holidays this month', 'Display holidays this month', 'Get holidays this month', 'Show all holidays this month',
        'Holiday list this month', 'Any holidays this month?', 'What holidays are this month?', 'Tell me holidays this month', 'Give me holidays this month',
        'This month holidays', 'Display all holidays in this month', 'Enumerate holidays this month', 'Show me holidays for this month', 'Holidays for this month'
    ]:
        cases.append((message, 'holiday', HOLIDAY_RECORDS, THIS_MONTH_HOLIDAYS))
    for message in ['Show holidays previous month', 'List holidays previous month', 'Display prior month holidays', 'Past month holidays', 'Holidays previous month']:
        cases.append((message, 'holiday', HOLIDAY_RECORDS, PREVIOUS_MONTH_HOLIDAYS))
    for message in ['Show holidays next month', 'List holidays next month', 'Display next month holidays', 'Any holidays next month?', 'Show all holidays next month', 'Next month holiday list', 'Give me holidays for next month', 'Enumerate holidays next month', 'What holidays are next month?', 'Tell me next month holidays', 'Display all holidays in next month', 'Holidays next month', 'List all next month holidays', 'Get holidays for next month', 'Show me next month holidays']:
        cases.append((message, 'holiday', HOLIDAY_RECORDS, NEXT_MONTH_HOLIDAYS))
    for message in ['Show holidays today', 'List holidays today', 'Display holidays today', 'Any holidays today?', 'What holidays are today?', 'Show all holidays today', 'Give holidays today', 'Today holidays', 'Tell me holidays today', 'Display all holidays for today', 'Get holidays today', 'Show me holidays on today', 'List today holidays', 'Enumerate holidays today', 'Holidays today please']:
        cases.append((message, 'holiday', HOLIDAY_RECORDS, TODAY_HOLIDAYS))
    for message in ['Show holidays tomorrow', 'List holidays tomorrow', 'Display holidays tomorrow', 'Any holidays tomorrow?', 'What holidays are tomorrow?', 'Show all holidays tomorrow', 'Tomorrow holidays', 'Get holidays tomorrow', 'Enumerate holidays tomorrow', 'Holidays tomorrow please']:
        cases.append((message, 'holiday', HOLIDAY_RECORDS, TOMORROW_HOLIDAYS))
    for message in ['Show holidays yesterday', 'List holidays yesterday', 'Display holidays yesterday', 'Any holidays yesterday?', 'What holidays are yesterday?', 'Show all holidays yesterday', 'Yesterday holidays', 'Get holidays yesterday', 'Enumerate holidays yesterday', 'Holidays yesterday please']:
        cases.append((message, 'holiday', HOLIDAY_RECORDS, YESTERDAY_HOLIDAYS))
    for message in ['Show holidays this week', 'List holidays this week', 'Display holidays this week', 'Any holidays this week?', 'This week holidays', 'Get holidays this week', 'Enumerate holidays this week', 'Show me holidays this week', 'Holidays for this week', 'This week holiday list']:
        cases.append((message, 'holiday', HOLIDAY_RECORDS, THIS_WEEK_HOLIDAYS))
    for message in ['Show holidays previous week', 'List holidays previous week', 'Display prior week holidays', 'Past week holidays', 'Holidays previous week', 'Last week holidays', 'Show last week holidays', 'List past week holidays', 'Any holidays previous week?', 'What holidays were previous week?']:
        cases.append((message, 'holiday', HOLIDAY_RECORDS, PREVIOUS_WEEK_HOLIDAYS))
    for message in ['Show holidays next week', 'List holidays next week', 'Display next week holidays', 'Any holidays next week?', 'Next week holidays', 'Get holidays next week', 'Enumerate holidays next week', 'Show me holidays next week', 'Holidays for next week', 'Next week holiday list']:
        cases.append((message, 'holiday', HOLIDAY_RECORDS, NEXT_WEEK_HOLIDAYS))
    for message in ['Show holidays this year', 'List holidays this year', 'Display holidays this year', 'Any holidays this year?', 'This year holidays', 'Get holidays this year', 'Enumerate holidays this year', 'Show me holidays this year', 'Holidays for this year', 'This year holiday list']:
        cases.append((message, 'holiday', HOLIDAY_RECORDS, THIS_YEAR_HOLIDAYS))
    for message in ['Show holidays previous year', 'List holidays previous year', 'Display prior year holidays', 'Past year holidays', 'Holidays previous year', 'Last year holidays', 'Show last year holidays', 'List past year holidays', 'Any holidays previous year?', 'What holidays were previous year?']:
        cases.append((message, 'holiday', HOLIDAY_RECORDS, NO_PREVIOUS_YEAR_HOLIDAYS))
    for message in ['Show holidays next year', 'List holidays next year', 'Display next year holidays', 'Any holidays next year?', 'Next year holidays', 'Get holidays next year', 'Enumerate holidays next year', 'Show me holidays next year', 'Holidays for next year', 'Next year holiday list']:
        cases.append((message, 'holiday', HOLIDAY_RECORDS, NO_NEXT_YEAR_HOLIDAYS))
    for message in ['Show upcoming holidays', 'List future holidays', 'Display upcoming holidays', 'Any upcoming holidays?', 'What are the upcoming holidays?', 'Show all upcoming holidays', 'Give me future holidays', 'Upcoming holidays', 'Tell me future holidays', 'Display all future holidays', 'Get upcoming holidays', 'Show me future holidays', 'List upcoming holidays', 'Enumerate future holidays', 'Next holidays please']:
        cases.append((message, 'holiday', HOLIDAY_RECORDS, UPCOMING_HOLIDAYS))
    for message in ['Show past holidays', 'List past holidays', 'Display previous holidays', 'Any past holidays?', 'What were the previous holidays?', 'Show all previous holidays', 'Give me past holidays', 'Previous holidays', 'Tell me past holidays', 'Display all past holidays']:
        cases.append((message, 'holiday', HOLIDAY_RECORDS, PAST_HOLIDAYS))
    for message in ['Show latest holiday', 'List latest holiday', 'Display recent holiday', 'Any latest holiday?', 'What is the recent holiday?', 'Give me latest holiday', 'Recent holiday', 'Tell me latest holiday', 'Display newest holiday', 'Which holiday is newest?']:
        cases.append((message, 'holiday', HOLIDAY_RECORDS, LATEST_HOLIDAY))
    for message in ['List all holidays', 'Show all holidays', 'Display all holidays', 'Enumerate holidays', 'Get all holidays', 'Holiday list', 'Show every holiday', 'Display holiday list', 'Enumerate all holidays', 'Give me all holidays']:
        cases.append((message, 'holiday', HOLIDAY_RECORDS[:2], ALL_HOLIDAYS_SHORT))
    return cases


MATRIX_CASES = build_matrix_cases()
assert len(MATRIX_CASES) >= 100


@pytest.mark.parametrize(('message', 'entity', 'records', 'expected'), MATRIX_CASES)
def test_grounded_response_matrix(message: str, entity: str, records: list[dict], expected: str):
    composer = build_composer()
    planner_output = PlannerOutput(intent=f'academic.{entity}.list', requires_tool=True, entity=entity, operation='list')
    assert composer.compose(message, planner_output, records) == expected


@pytest.mark.asyncio
async def test_multi_step_response_uses_step_local_question_for_each_section():
    node = build_node()
    state = RuntimeState(
        conversation_id='conv-1',
        request_id='req-1',
        correlation_id='corr-1',
        runtime_context=RuntimeContext(subject='user-1', user_id='user-1'),
        user_question='How many academic years and list holidays?',
        planner_output=PlannerOutput(
            intent='academic.multi.list',
            requires_tool=True,
            domain='vidhya',
            service='academic',
            entity='academic_year',
            operation='list',
            execution_plan=[
                {
                    'step_id': 'step_1',
                    'intent': 'academic.academic_year.list',
                    'domain': 'vidhya',
                    'service': 'academic',
                    'entity': 'academic_year',
                    'operation': 'list',
                    'parameters': {},
                    'question': 'How many academic years are configured?',
                    'visible_in_response': True,
                },
                {
                    'step_id': 'step_2',
                    'intent': 'academic.holiday.list',
                    'domain': 'vidhya',
                    'service': 'academic',
                    'entity': 'holiday',
                    'operation': 'list',
                    'parameters': {},
                    'question': 'List holidays.',
                    'visible_in_response': True,
                },
            ],
        ),
        tool_execution_result={
            'success': True,
            'steps': [
                {
                    'step_id': 'step_1',
                    'question': 'How many academic years are configured?',
                    'visible_in_response': True,
                    'result': {'success': True, 'data': {'items': ACADEMIC_YEAR_RECORDS}},
                },
                {
                    'step_id': 'step_2',
                    'question': 'List holidays.',
                    'visible_in_response': True,
                    'result': {'success': True, 'data': {'items': HOLIDAY_RECORDS[:2]}},
                },
            ],
            'data': {'items': HOLIDAY_RECORDS[:2]},
        },
    )

    result = await node.__call__(state)

    assert result['final_response'] == 'Academic years:\nThere are 3 years configured.\n\nHolidays:\n' + ALL_HOLIDAYS_SHORT
