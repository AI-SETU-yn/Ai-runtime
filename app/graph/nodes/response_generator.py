import json
import logging
import re
from datetime import date, datetime, timedelta
from dataclasses import dataclass
from typing import Any, Callable, Literal

from app.graph.state import RuntimeState
from app.model_gateway.client import ModelGatewayClient
from app.model_gateway.exceptions import ModelGatewayError
from app.models.planner import PlannerOutput
from app.prompts.builder import PromptBuilder
from app.utils.json_logging import pretty_json
from app.utils.temporal import TemporalIntent, normalize_temporal_intent

logger = logging.getLogger(__name__)

_BULLET = '\u2022'
CollectionResponseMode = Literal['LIST', 'COUNT', 'DETAIL']
AcademicYearResponseMode = Literal['LIST', 'COUNT', 'DETAIL', 'CURRENT', 'PREVIOUS', 'FUTURE', 'LATEST']


@dataclass(frozen=True)
class SpecializedRecordFormatter:
    name: str
    matcher: Callable[[list[Any]], bool]
    formatter: Callable[[str, list[Any]], str]


class FormatterRegistry:
    def __init__(self, formatters: tuple[SpecializedRecordFormatter, ...]) -> None:
        self._formatters = formatters

    def select(self, records: list[Any]) -> SpecializedRecordFormatter | None:
        for formatter in self._formatters:
            if formatter.matcher(records):
                return formatter
        return None


class ResponseGeneratorNode:
    def __init__(
        self,
        prompt_builder: PromptBuilder,
        model_gateway_client: ModelGatewayClient,
        current_date_provider: Callable[[], date] | None = None,
    ) -> None:
        self._prompt_builder = prompt_builder
        self._model_gateway_client = model_gateway_client
        self._grounded_composer = GroundedResponseComposer(current_date_provider=current_date_provider)

    async def __call__(self, state: RuntimeState):
        planner_output = state.planner_output
        assert planner_output is not None
        if self._tool_failed(state.tool_execution_result):
            answer = self._tool_failure_answer(state.tool_execution_result)
            logger.warning('response_generation_skipped_failed_tool_json\n%s', pretty_json({
                'planner_intent': planner_output.intent,
                'tool_execution_result': state.tool_execution_result,
                'final_response': answer,
            }))
            return {'model_response': answer, 'final_response': answer}

        plan_answer = self._compose_execution_plan_answer(state.user_question, planner_output, state.tool_execution_result)
        if plan_answer:
            logger.info('response_generator_execution_plan_answer_json\n%s', pretty_json({
                'planner_intent': planner_output.intent,
                'execution_plan_step_count': len(planner_output.execution_plan),
                'final_response': plan_answer,
            }))
            return {'model_response': plan_answer, 'final_response': plan_answer}

        enterprise_data = self._prompt_builder.extract_enterprise_data(state.tool_execution_result)
        deterministic_answer = self._grounded_composer.compose(state.user_question, planner_output, enterprise_data)
        if deterministic_answer:
            logger.info('response_generator_deterministic_answer_json\n%s', pretty_json({
                'planner_intent': planner_output.intent,
                'entity': planner_output.entity,
                'operation': planner_output.operation,
                'enterprise_data': enterprise_data,
                'final_response': deterministic_answer,
            }))
            return {'model_response': deterministic_answer, 'final_response': deterministic_answer}

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
        try:
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
        except ModelGatewayError:
            answer = self._general_answer_failure_answer()
            logger.warning('response_generator_gateway_failed_using_fallback')

        if not isinstance(answer, str) or not answer.strip():
            answer = self._general_answer_failure_answer()
            logger.warning('response_generator_empty_answer_using_fallback')
        if not self._grounded_composer.validate(state.user_question, answer, planner_output, enterprise_data):
            fallback_answer = self._grounded_composer.fallback(state.user_question, planner_output, enterprise_data)
            logger.warning('response_generator_grounding_validation_failed_json\n%s', pretty_json({
                'planner_intent': planner_output.intent,
                'entity': planner_output.entity,
                'operation': planner_output.operation,
                'enterprise_data': enterprise_data,
                'model_response': answer,
                'fallback_response': fallback_answer,
            }))
            answer = fallback_answer
        logger.info('final_response_json\n%s', pretty_json({'final_response': answer}))
        return {'model_response': answer, 'final_response': answer}

    def _compose_execution_plan_answer(
        self,
        user_question: str,
        planner_output: PlannerOutput,
        tool_execution_result: dict[str, object] | None,
    ) -> str | None:
        if not planner_output.execution_plan or not isinstance(tool_execution_result, dict):
            return None
        steps = tool_execution_result.get('steps')
        if not isinstance(steps, list):
            return None

        execution_steps = {
            step.get('step_id'): step
            for step in steps
            if isinstance(step, dict) and isinstance(step.get('step_id'), str)
        }
        visible_steps = [
            (index, step)
            for index, step in enumerate(planner_output.execution_plan, start=1)
            if step.visible_in_response
        ]
        if len(visible_steps) <= 1:
            return None

        sections: list[str] = []
        for index, step in visible_steps:
            step_id = step.step_id or f'step_{index}'
            execution_step = execution_steps.get(step_id)
            if not isinstance(execution_step, dict):
                continue
            result = execution_step.get('result')
            if not isinstance(result, dict):
                continue
            enterprise_data = self._prompt_builder.extract_enterprise_data(result)
            step_output = PlannerOutput(
                intent=step.intent or planner_output.intent,
                requires_tool=True,
                domain=step.domain,
                service=step.service,
                entity=step.entity,
                operation=step.operation,
                parameters=step.parameters,
            )
            step_question = step.question or execution_step.get('question') or user_question
            rendered = self._grounded_composer.compose(step_question, step_output, enterprise_data)
            if not rendered:
                continue
            sections.append(self._grounded_composer.format_section(step.entity, rendered))

        if not sections:
            return None
        return '\n\n'.join(sections)

    @staticmethod
    def _tool_failed(tool_execution_result: dict[str, object] | None) -> bool:
        if not isinstance(tool_execution_result, dict):
            return False
        if tool_execution_result.get('success') is False or tool_execution_result.get('status') == 'error':
            return True
        data = tool_execution_result.get('data')
        return isinstance(data, dict) and data.get('isError') is True

    @classmethod
    def _tool_failure_answer(cls, tool_execution_result: dict[str, object] | None) -> str:
        detail = cls._extract_tool_error_detail(tool_execution_result)
        if detail:
            return f"I couldn't fetch the requested enterprise data. {detail}"
        return "I couldn't fetch the requested enterprise data because the tool call failed. Please try again after refreshing your session."

    @staticmethod
    def _general_answer_failure_answer() -> str:
        return (
            "I couldn't come up with an answer to that right now. "
            "Please try rephrasing your question or try again in a moment."
        )

    @classmethod
    def _extract_tool_error_detail(cls, value: Any) -> str | None:
        if isinstance(value, dict):
            content = value.get('content')
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and isinstance(item.get('text'), str):
                        return item['text']
            for key in ('content', 'data', 'error', 'message', 'text'):
                detail = cls._extract_tool_error_detail(value.get(key))
                if detail:
                    return detail
        if isinstance(value, list):
            for item in value:
                detail = cls._extract_tool_error_detail(item)
                if detail:
                    return detail
        return value if isinstance(value, str) and value else None


class GroundedResponseComposer:
    _ALL_HINTS = ('all', 'show all', 'display all', 'enumerate', 'get all', 'every')
    _COUNT_HINTS = ('how many', 'total', 'number of', 'count')
    _DETAIL_HINTS = ('details for', 'detail for', 'information about', 'info about', 'show academic year', 'academic year details')
    _ACADEMIC_YEAR_PATTERN = re.compile(r'\b(\d{4}-\d{4})\b')

    def __init__(self, current_date_provider: Callable[[], date] | None = None) -> None:
        self._current_date_provider = current_date_provider or date.today
        self._formatter_registry = FormatterRegistry(self._build_specialized_record_formatters())

    def _build_specialized_record_formatters(self) -> tuple[SpecializedRecordFormatter, ...]:
        return (
            SpecializedRecordFormatter(
                name='academic_year',
                matcher=self._is_academic_year_records,
                formatter=self._format_academic_year_records,
            ),
            SpecializedRecordFormatter(
                name='holiday',
                matcher=self._is_holiday_records,
                formatter=self._format_holiday_records,
            ),
        )

    def compose(self, user_question: str, planner_output: PlannerOutput, enterprise_data: Any) -> str | None:
        if not planner_output.requires_tool or enterprise_data in (None, '', {}, []):
            return None
        records, label = self._extract_records(enterprise_data, planner_output)
        if records is not None:
            return self._format_record_collection(user_question, records, label, planner_output.entity)
        if isinstance(enterprise_data, dict):
            return self._format_object_details(enterprise_data, planner_output.entity)
        return None

    def validate(self, user_question: str, answer: str, planner_output: PlannerOutput, enterprise_data: Any) -> bool:
        expected = self.compose(user_question, planner_output, enterprise_data)
        if expected is None:
            return True
        return answer.strip() == expected.strip()

    def fallback(self, user_question: str, planner_output: PlannerOutput, enterprise_data: Any) -> str:
        deterministic = self.compose(user_question, planner_output, enterprise_data)
        if deterministic:
            return deterministic
        return json.dumps(enterprise_data, ensure_ascii=True, default=str)

    def _extract_records(self, enterprise_data: Any, planner_output: PlannerOutput) -> tuple[list[Any] | None, str]:
        if isinstance(enterprise_data, list):
            return enterprise_data, self._plural_label(planner_output.entity)
        if isinstance(enterprise_data, dict):
            for key, value in enterprise_data.items():
                if isinstance(value, list):
                    return value, self._pluralize_words(key.replace('_', ' '))
        return None, self._plural_label(planner_output.entity)

    def _format_record_collection(self, user_question: str, records: list[Any], label: str, entity: str | None) -> str:
        specialized = self._formatter_registry.select(records)
        if specialized is not None:
            return specialized.formatter(user_question, records)
        return self._format_generic_collection(user_question, records, label, entity)

    def format_section(self, entity: str | None, rendered: str) -> str:
        label = self._humanize_label(self._plural_label(entity))
        return f'{label}:\n{rendered}'

    def _format_generic_collection(self, user_question: str, records: list[Any], label: str, entity: str | None) -> str:
        mode = self._generic_response_mode(user_question)
        if mode == 'COUNT':
            return f'There are {len(records)} {label} configured.'

        if mode == 'DETAIL':
            match = self._find_generic_detail_record(user_question, records)
            if match is not None:
                singular = self._singular_label(entity)
                return self._format_generic_detail(singular, match, entity)

        heading = f'Found {len(records)} {label}.'
        if not records:
            return heading
        lines = [heading, '']
        for record in records:
            lines.append(f'{_BULLET} {self._render_record(record, entity)}')
        return '\n'.join(lines)

    def _generic_response_mode(self, user_question: str) -> CollectionResponseMode:
        lowered = user_question.lower()
        if self._contains_any(lowered, self._COUNT_HINTS):
            return 'COUNT'
        if self._looks_like_detail_request(lowered, user_question):
            return 'DETAIL'
        return 'LIST'

    def _looks_like_detail_request(self, lowered: str, user_question: str) -> bool:
        return bool(self._detail_lookup_token(user_question)) and any(
            phrase in lowered
            for phrase in ('details for', 'detail for', 'information about', 'info about', 'show', 'get', 'find')
        )

    def _find_generic_detail_record(self, user_question: str, records: list[Any]) -> Any | None:
        lookup = self._detail_lookup_token(user_question)
        if not lookup:
            return None
        normalized_lookup = self._normalize_lookup_value(lookup)
        for record in records:
            if not isinstance(record, dict):
                continue
            for field in self._candidate_detail_fields(record):
                value = record.get(field)
                if self._normalize_lookup_value(value) == normalized_lookup:
                    return record
        return None

    def _format_generic_detail(self, singular: str, record: Any, entity: str | None) -> str:
        return f'{self._humanize_label(singular)} details:\n\n{_BULLET} {self._render_record(record, entity)}'

    def _detail_lookup_token(self, user_question: str) -> str | None:
        academic_year = self._requested_academic_year(user_question)
        if academic_year:
            return academic_year
        quoted = re.search(r'"([^"]+)"', user_question)
        if quoted:
            return quoted.group(1).strip()
        tail = re.search(r'(?:details? for|information about|info about|show|get|find)\s+([A-Za-z0-9][A-Za-z0-9_\- ]+)$', user_question, re.IGNORECASE)
        if tail:
            return tail.group(1).strip()
        return None

    def _candidate_detail_fields(self, record: dict[str, Any]) -> tuple[str, ...]:
        priority = ('name', 'title', 'code', 'academicYear', 'holidayName')
        ordered = [field for field in priority if field in record]
        ordered.extend(key for key, value in record.items() if key not in ordered and isinstance(value, (str, int, float)))
        return tuple(ordered)

    @staticmethod
    def _normalize_lookup_value(value: Any) -> str:
        return re.sub(r'\s+', ' ', str(value or '').strip()).lower()

    def _format_academic_year_records(self, user_question: str, records: list[Any]) -> str:
        mode = self._academic_year_response_mode(user_question)
        temporal = self._academic_year_temporal_intent(user_question, mode)
        current_record = self._find_flagged_record(records, 'isCurrentAcademicYear')
        current_years = self._parse_academic_year(self._academic_year_value(current_record)) if current_record else None
        parsed = [item for item in (self._build_academic_year_view(record, current_years) for record in records) if item is not None]
        wants_all = self._contains_any(user_question.lower(), self._ALL_HINTS)

        if mode == 'COUNT':
            return f'There are {len(parsed)} years configured.'

        if mode == 'DETAIL':
            target_year = self._requested_academic_year(user_question)
            if not target_year:
                ordered = self._preserve_original_order(parsed, records)
                return self._format_academic_year_list('years', ordered)
            match = next((item for item in parsed if item['label'] == target_year), None)
            if match is None:
                return f'Year {target_year} was not found.'
            return self._format_academic_year_detail(match)

        if temporal and temporal.scope == 'current':
            if current_record is None:
                return 'There is no active year.'
            return f"The current active year is {self._academic_year_value(current_record)}."

        if temporal and temporal.scope == 'past':
            previous = [item for item in parsed if item['relative'] == 'previous']
            if not previous:
                return 'There are no previous years.'
            return self._format_academic_year_list('previous years', self._sort_academic_year_items(previous))

        if temporal and temporal.scope == 'future':
            future = [item for item in parsed if item['relative'] == 'future']
            if not future:
                return 'There are no upcoming years.'
            ordered = self._sort_academic_year_items(future)
            if wants_all:
                return self._format_academic_year_list('upcoming years', ordered)
            return f"The next year is {ordered[0]['label']}."

        if temporal and temporal.scope == 'latest':
            ordered = self._sort_academic_year_items(parsed)
            if not ordered:
                return 'No years were found.'
            if wants_all:
                return self._format_academic_year_list('years', ordered)
            return f"The latest year is {ordered[-1]['label']}."

        ordered = self._preserve_original_order(parsed, records)
        return self._format_academic_year_list('years', ordered)

    def _format_holiday_records(self, user_question: str, records: list[Any]) -> str:
        temporal = normalize_temporal_intent(user_question)
        filtered = self._filter_holidays(records, temporal)
        if filtered is not None:
            holidays, scope = filtered
            if not holidays:
                return self._empty_holiday_message(scope, temporal)
            wants_all = self._contains_any(user_question.lower(), self._ALL_HINTS)
            if scope == 'latest' and len(holidays) == 1 and not wants_all:
                return f"The latest holiday is {self._render_holiday(holidays[0])}."
            lines = [self._holiday_heading(scope, len(holidays), temporal), '']
            for record in holidays:
                lines.append(f'{_BULLET} {self._render_holiday(record)}')
            return '\n'.join(lines)
        lines = [f'Found {len(records)} holidays.', '']
        for record in records:
            lines.append(f'{_BULLET} {self._render_holiday(record)}')
        return '\n'.join(lines)

    def _filter_holidays(self, records: list[Any], temporal: TemporalIntent | None) -> tuple[list[dict[str, Any]], str] | None:
        if temporal is None:
            return None
        holiday_records = [record for record in records if isinstance(record, dict)]
        current = self._current_date_provider()
        if temporal.scope == 'day':
            target = current + timedelta(days=temporal.offset)
            matches = [r for r in holiday_records if self._holiday_matches_day(r, target)]
            return (self._sort_holidays(matches), 'day')
        if temporal.scope == 'week':
            anchor = current + timedelta(days=temporal.offset * 7)
            matches = [r for r in holiday_records if self._holiday_in_week(r, anchor)]
            return (self._sort_holidays(matches), 'week')
        if temporal.scope == 'month':
            target = self._add_months(current, temporal.offset)
            matches = [r for r in holiday_records if self._holiday_in_same_month(r, target)]
            return (self._sort_holidays(matches), 'month')
        if temporal.scope == 'year':
            target_year = current.year + temporal.offset
            matches = [r for r in holiday_records if self._holiday_in_year(r, target_year)]
            return (self._sort_holidays(matches), 'year')
        if temporal.scope == 'future':
            upcoming = [r for r in holiday_records if self._holiday_end_date(r) and self._holiday_end_date(r) >= current]
            return (self._sort_holidays(upcoming), 'future')
        if temporal.scope == 'past':
            previous = [r for r in holiday_records if self._holiday_end_date(r) and self._holiday_end_date(r) < current]
            return (self._sort_holidays(previous), 'past')
        if temporal.scope == 'latest':
            return (self._sort_holidays(self._latest_holiday_records(holiday_records)), 'latest')
        return None

    def _format_object_details(self, enterprise_data: dict[str, Any], entity: str | None) -> str:
        title = self._singular_label(entity) or 'record'
        lines = [f'Found {title} details.', '']
        for key, value in enterprise_data.items():
            lines.append(f'{_BULLET} {self._humanize_key(key)}: {self._render_value(value)}')
        return '\n'.join(lines)

    def _format_academic_year_list(self, label: str, items: list[dict[str, Any]]) -> str:
        lines = [f'Found {len(items)} {label}.', '']
        for item in items:
            suffix = ' (Current)' if item['current'] else ''
            lines.append(f"{_BULLET} {item['label']}{suffix}")
        return '\n'.join(lines)

    def _format_academic_year_detail(self, item: dict[str, Any]) -> str:
        suffix = ' (Current)' if item['current'] else ''
        return f'Year details:\n\n{_BULLET} {item["label"]}{suffix}'

    def _build_academic_year_view(self, record: Any, current_years: tuple[int, int] | None) -> dict[str, Any] | None:
        if not isinstance(record, dict):
            return None
        label = self._academic_year_value(record)
        years = self._parse_academic_year(label)
        if years is None:
            return None
        relative = 'current' if record.get('isCurrentAcademicYear') is True else self._academic_year_relative(years, current_years)
        return {'label': label, 'years': years, 'current': record.get('isCurrentAcademicYear') is True, 'relative': relative}

    def _academic_year_response_mode(self, user_question: str) -> AcademicYearResponseMode:
        lowered = user_question.lower()
        temporal = normalize_temporal_intent(user_question)
        wants_all = self._contains_any(lowered, self._ALL_HINTS)
        if self._contains_any(lowered, self._COUNT_HINTS):
            return 'COUNT'
        if self._requested_academic_year(user_question) and (
            self._contains_any(lowered, self._DETAIL_HINTS)
            or ('academic year' in lowered and not wants_all)
        ):
            return 'DETAIL'
        if temporal is not None:
            if temporal.scope == 'current':
                return 'LIST' if wants_all else 'CURRENT'
            if temporal.scope == 'past' or (temporal.scope == 'year' and temporal.offset < 0):
                return 'PREVIOUS'
            if temporal.scope == 'future' or (temporal.scope == 'year' and temporal.offset > 0):
                return 'FUTURE'
            if temporal.scope == 'latest':
                return 'LATEST'
        if wants_all:
            return 'LIST'
        return 'LIST'

    def _academic_year_temporal_intent(self, user_question: str, mode: AcademicYearResponseMode) -> TemporalIntent | None:
        if mode == 'CURRENT':
            return TemporalIntent(scope='current')
        if mode == 'PREVIOUS':
            return TemporalIntent(scope='past')
        if mode == 'FUTURE':
            return TemporalIntent(scope='future')
        if mode == 'LATEST':
            return TemporalIntent(scope='latest')
        return None

    def _requested_academic_year(self, user_question: str) -> str | None:
        match = self._ACADEMIC_YEAR_PATTERN.search(user_question)
        if not match:
            return None
        return match.group(1)

    @staticmethod
    def _academic_year_value(record: dict[str, Any] | None) -> str:
        if not isinstance(record, dict):
            return ''
        return str(record.get('academicYear', '')).strip()

    @staticmethod
    def _parse_academic_year(value: str) -> tuple[int, int] | None:
        parts = value.split('-')
        if len(parts) != 2:
            return None
        try:
            return int(parts[0]), int(parts[1])
        except ValueError:
            return None

    def _academic_year_relative(self, years: tuple[int, int], current_years: tuple[int, int] | None) -> str:
        current = self._current_date_provider()
        if current_years is not None:
            if years < current_years:
                return 'previous'
            if years > current_years:
                return 'future'
            return 'current'
        if years[1] < current.year:
            return 'previous'
        if years[0] > current.year:
            return 'future'
        return 'unknown'

    @staticmethod
    def _sort_academic_year_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(items, key=lambda item: item['years'])

    @staticmethod
    def _preserve_original_order(items: list[dict[str, Any]], records: list[Any]) -> list[dict[str, Any]]:
        order = {GroundedResponseComposer._academic_year_value(record): index for index, record in enumerate(records) if isinstance(record, dict)}
        return sorted(items, key=lambda item: order.get(item['label'], 0))

    def _render_record(self, record: Any, entity: str | None) -> str:
        if not isinstance(record, dict):
            return self._render_value(record)
        preferred_keys = self._preferred_keys(entity)
        parts: list[str] = []
        seen: set[str] = set()
        for key in preferred_keys:
            if key in record:
                value = self._render_value(record[key])
                if value:
                    parts.append(f'{self._humanize_key(key)}: {value}')
                    seen.add(key)
        for key, value in record.items():
            if key in seen or self._is_internal_field(key):
                continue
            rendered = self._render_value(value)
            if rendered:
                parts.append(f'{self._humanize_key(key)}: {rendered}')
        return ', '.join(parts) if parts else self._render_value(record)

    def _render_holiday(self, record: dict[str, Any]) -> str:
        name = str(record.get('holidayName') or record.get('name') or 'Holiday').strip()
        start = record.get('holidayStartDate') or record.get('startDate')
        end = record.get('holidayEndDate') or record.get('endDate')
        parts = [name]
        if start:
            parts.append(f'Start Date: {start}')
        if end and end != start:
            parts.append(f'End Date: {end}')
        return ', '.join(parts)

    def _render_value(self, value: Any) -> str:
        if isinstance(value, bool):
            return 'true' if value else 'false'
        if isinstance(value, list):
            return ', '.join(self._render_value(item) for item in value)
        if isinstance(value, dict):
            public_items = {key: item for key, item in value.items() if not self._is_internal_field(key)}
            return json.dumps(public_items or value, ensure_ascii=True, default=str, separators=(',', ':'))
        return str(value)

    @staticmethod
    def _contains_any(text: str, phrases: tuple[str, ...]) -> bool:
        return any(phrase in text for phrase in phrases)

    @staticmethod
    def _find_flagged_record(records: list[Any], flag_name: str) -> dict[str, Any] | None:
        for record in records:
            if isinstance(record, dict) and record.get(flag_name) is True:
                return record
        return None

    @staticmethod
    def _is_academic_year_records(records: list[Any]) -> bool:
        return bool(records) and all(isinstance(item, dict) and 'academicYear' in item for item in records)

    @staticmethod
    def _is_holiday_records(records: list[Any]) -> bool:
        return bool(records) and all(isinstance(item, dict) and any(key in item for key in ('holidayName', 'holidayStartDate', 'holidayEndDate')) for item in records)

    @staticmethod
    def _preferred_keys(entity: str | None) -> list[str]:
        mapping = {
            'subject': ['name', 'code', 'status'],
            'holiday': ['holidayName', 'holidayStartDate', 'holidayEndDate'],
            'section': ['name', 'code', 'status'],
            'class': ['name', 'code', 'status'],
        }
        return mapping.get((entity or '').lower(), ['name', 'title', 'code', 'status'])

    @staticmethod
    def _singular_label(entity: str | None) -> str:
        if not entity:
            return 'record'
        return entity.replace('_', ' ')

    @classmethod
    def _plural_label(cls, entity: str | None) -> str:
        return cls._pluralize_words(cls._singular_label(entity))

    @staticmethod
    def _pluralize_words(label: str) -> str:
        label = label.strip()
        if not label:
            return 'records'
        if label.endswith('s'):
            return label
        return f'{label}s'

    @staticmethod
    def _humanize_label(value: str) -> str:
        normalized = value.replace('_', ' ').strip()
        return normalized[:1].upper() + normalized[1:] if normalized else 'Results'

    @staticmethod
    def _humanize_key(key: str) -> str:
        normalized = key.replace('_', ' ')
        return normalized[:1].upper() + normalized[1:]

    @staticmethod
    def _is_internal_field(key: str) -> bool:
        lowered = key.lower()
        return lowered in {'referenceid', 'academicyearsummaryresponse', 'holidaystatus', 'statusflag', 'metadata', 'iscurrentacademicyear'} or lowered.endswith('id')

    @staticmethod
    def _parse_date(value: Any) -> date | None:
        if not isinstance(value, str) or not value:
            return None
        try:
            return datetime.strptime(value[:10], '%Y-%m-%d').date()
        except ValueError:
            return None

    def _holiday_start_date(self, record: dict[str, Any]) -> date | None:
        return self._parse_date(record.get('holidayStartDate') or record.get('startDate'))

    def _holiday_end_date(self, record: dict[str, Any]) -> date | None:
        return self._parse_date(record.get('holidayEndDate') or record.get('endDate') or record.get('holidayStartDate') or record.get('startDate'))

    def _holiday_in_same_month(self, record: dict[str, Any], target: date) -> bool:
        start = self._holiday_start_date(record)
        end = self._holiday_end_date(record)
        if start is None or end is None:
            return False
        month_start = date(target.year, target.month, 1)
        month_end = self._end_of_month(target)
        return start <= month_end and end >= month_start

    def _holiday_matches_day(self, record: dict[str, Any], target: date) -> bool:
        start = self._holiday_start_date(record)
        end = self._holiday_end_date(record)
        if start is None or end is None:
            return False
        return start <= target <= end

    def _holiday_in_week(self, record: dict[str, Any], anchor: date) -> bool:
        start = self._holiday_start_date(record)
        end = self._holiday_end_date(record)
        if start is None or end is None:
            return False
        week_start = anchor - timedelta(days=anchor.weekday())
        week_end = week_start + timedelta(days=6)
        return start <= week_end and end >= week_start

    def _holiday_in_year(self, record: dict[str, Any], year: int) -> bool:
        start = self._holiday_start_date(record)
        end = self._holiday_end_date(record)
        if start is None or end is None:
            return False
        year_start = date(year, 1, 1)
        year_end = date(year, 12, 31)
        return start <= year_end and end >= year_start

    @staticmethod
    def _add_months(base: date, months: int) -> date:
        month_index = (base.month - 1) + months
        year = base.year + month_index // 12
        month = month_index % 12 + 1
        return date(year, month, 1)

    @staticmethod
    def _end_of_month(target: date) -> date:
        first_next = date(target.year + (1 if target.month == 12 else 0), 1 if target.month == 12 else target.month + 1, 1)
        return first_next - timedelta(days=1)

    def _latest_holiday_records(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        dated = [(record, self._holiday_start_date(record)) for record in records]
        dated = [(record, value) for record, value in dated if value is not None]
        if not dated:
            return []
        latest_date = max(value for _, value in dated)
        return [record for record, value in dated if value == latest_date]

    def _sort_holidays(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(records, key=lambda record: self._holiday_start_date(record) or date.min)

    def _holiday_heading(self, scope: str, count: int, temporal: TemporalIntent) -> str:
        if scope == 'day':
            label = {0: 'today', 1: 'tomorrow', -1: 'yesterday'}.get(temporal.offset, 'that day')
            return f'Found {count} holidays {label}.'
        if scope == 'week':
            label = {0: 'this week', 1: 'next week', -1: 'the previous week'}.get(temporal.offset, 'that week')
            return f'Found {count} holidays {label}.'
        if scope == 'month':
            label = {0: 'this month', 1: 'next month', -1: 'the previous month'}.get(temporal.offset, 'that month')
            return f'Found {count} holidays {label}.'
        if scope == 'year':
            label = {0: 'this year', 1: 'next year', -1: 'the previous year'}.get(temporal.offset, 'that year')
            return f'Found {count} holidays {label}.'
        if scope == 'future':
            return f'Found {count} upcoming holidays.'
        if scope == 'past':
            return f'Found {count} previous holidays.'
        if scope == 'latest':
            return f'Found {count} latest holidays.'
        return f'Found {count} holidays.'

    def _empty_holiday_message(self, scope: str, temporal: TemporalIntent) -> str:
        if scope == 'day':
            return {
                0: 'There are no holidays scheduled for today.',
                1: 'There are no holidays scheduled for tomorrow.',
                -1: 'There were no holidays scheduled for yesterday.',
            }.get(temporal.offset, 'There are no holidays scheduled for that day.')
        if scope == 'week':
            return {
                0: 'There are no holidays scheduled for this week.',
                1: 'There are no holidays scheduled for next week.',
                -1: 'There are no holidays scheduled for the previous week.',
            }.get(temporal.offset, 'There are no holidays scheduled for that week.')
        if scope == 'month':
            return {
                0: 'There are no holidays scheduled for this month.',
                1: 'There are no holidays scheduled for next month.',
                -1: 'There are no holidays scheduled for the previous month.',
            }.get(temporal.offset, 'There are no holidays scheduled for that month.')
        if scope == 'year':
            return {
                0: 'There are no holidays scheduled for this year.',
                1: 'There are no holidays scheduled for next year.',
                -1: 'There are no holidays scheduled for the previous year.',
            }.get(temporal.offset, 'There are no holidays scheduled for that year.')
        if scope == 'future':
            return 'There are no upcoming holidays.'
        if scope == 'past':
            return 'There are no previous holidays.'
        return 'No holidays were found.'
