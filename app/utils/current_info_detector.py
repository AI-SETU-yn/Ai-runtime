"""Small heuristic for questions that likely need fresh public information."""
from __future__ import annotations

import re

_CURRENT_INFO_KEYWORDS = (
    'today', 'yesterday', 'tomorrow', 'tonight', 'current', 'currently',
    'latest', 'recent', 'recently', 'now', 'right now', 'live', 'breaking',
    'trending', 'news', 'headline', 'headlines', 'weather', 'forecast',
    'temperature', 'stock price', 'share price', 'exchange rate', 'score',
    'live score', 'match result', 'this week', 'this month', 'this year',
)
_YEAR_PATTERN = re.compile(r'\b(?:19|20)\d{2}\b')


def is_current_info_query(user_question: str) -> bool:
    """Return whether a non-tool question appears to require current data.

    This function deliberately makes no routing decision. The caller must first
    respect the Planner's ``requires_tool`` decision so ERP/tool work wins.
    """
    if not user_question or not user_question.strip():
        return False
    lowered = user_question.casefold()
    if any(keyword in lowered for keyword in _CURRENT_INFO_KEYWORDS):
        return True
    return bool(_YEAR_PATTERN.search(lowered)) and any(
        word in lowered for word in ('news', 'price', 'weather', 'score', 'update', 'happening')
    )
