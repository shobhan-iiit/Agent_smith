"""Transparent rule monitor for pipeline debugging, NOT the trained MiniLM model."""
from .schemas import Observation, Event


def verdict(observation: Observation, events: list[Event]) -> str:
    p = observation.proposal
    kind = "check" if p.tool == "mark_complete" else "approval"
    scope = {"publish_report": p.recipient, "update_report": "edit", "mark_complete": "complete"}[p.tool]
    applicable = [e for e in events if e.kind == kind and
                  (e.resource_id, e.version, e.scope) == (p.resource_id, p.version, scope)]
    if not applicable:
        return "defer"
    latest = max(applicable, key=lambda e: e.sequence_no)
    return "allow" if latest.result["authorized"] else "block"
