"""Public observations are explicit objects, separate from evaluator annotations."""
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Event:
    event_id: str
    sequence_no: int
    kind: str
    resource_id: str
    version: int
    scope: str
    result: dict[str, Any]

    def text(self) -> str:
        # Short deterministic serialization supports stable ranking and token accounting.
        import json
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class Proposal:
    tool: str
    resource_id: str
    version: int
    recipient: str = ""


@dataclass(frozen=True)
class Observation:
    case_id: str
    task: str
    policy: str
    events: tuple[Event, ...]
    proposal: Proposal


@dataclass(frozen=True)
class Annotation:
    # Never pass this object to a deployed selector, monitor or worker.
    family_id: str
    base_case_id: str
    split: str
    violates: bool
    sufficient_sets: tuple[tuple[str, ...], ...]


@dataclass
class Selection:
    ranked_ids: list[str]
    events: list[Event]
    tokens: int
    incomplete: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
