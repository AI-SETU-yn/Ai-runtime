from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TemporalScope = Literal['day', 'week', 'month', 'year', 'current', 'future', 'past', 'latest']


@dataclass(frozen=True)
class TemporalIntent:
    scope: TemporalScope | None = None
    offset: int = 0


def normalize_temporal_intent(user_query: str) -> TemporalIntent | None:
    lowered = user_query.lower()

    absolute_aliases = {
        'today': TemporalIntent(scope='day', offset=0),
        'tomorrow': TemporalIntent(scope='day', offset=1),
        'yesterday': TemporalIntent(scope='day', offset=-1),
        'this week': TemporalIntent(scope='week', offset=0),
        'next week': TemporalIntent(scope='week', offset=1),
        'last week': TemporalIntent(scope='week', offset=-1),
        'previous week': TemporalIntent(scope='week', offset=-1),
        'prior week': TemporalIntent(scope='week', offset=-1),
        'past week': TemporalIntent(scope='week', offset=-1),
        'this month': TemporalIntent(scope='month', offset=0),
        'next month': TemporalIntent(scope='month', offset=1),
        'last month': TemporalIntent(scope='month', offset=-1),
        'previous month': TemporalIntent(scope='month', offset=-1),
        'prior month': TemporalIntent(scope='month', offset=-1),
        'past month': TemporalIntent(scope='month', offset=-1),
        'this year': TemporalIntent(scope='year', offset=0),
        'next year': TemporalIntent(scope='year', offset=1),
        'last year': TemporalIntent(scope='year', offset=-1),
        'previous year': TemporalIntent(scope='year', offset=-1),
        'prior year': TemporalIntent(scope='year', offset=-1),
        'past year': TemporalIntent(scope='year', offset=-1),
    }
    for phrase, intent in absolute_aliases.items():
        if phrase in lowered:
            return intent

    if any(token in lowered for token in ('current', 'active', 'selected', 'default', 'now')):
        return TemporalIntent(scope='current')
    if any(token in lowered for token in ('upcoming', 'future')) or 'next holidays' in lowered:
        return TemporalIntent(scope='future')
    if any(token in lowered for token in ('latest', 'recent', 'newest')):
        return TemporalIntent(scope='latest')
    if any(token in lowered for token in ('previous', 'past', 'last', 'prior')):
        return TemporalIntent(scope='past')
    return None
