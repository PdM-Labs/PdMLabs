from typing import NamedTuple, List
from mypy_extensions import TypedDict


class EventPreferencesTuple(NamedTuple):
    description: str
    type: str
    source: str
    target_sources: str


class EventPreferences(TypedDict):
    failure: List[EventPreferencesTuple]
    reset: List[EventPreferencesTuple]
