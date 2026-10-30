"""Dependency-free BM25, history grouping and pluggable token counting."""
from collections import Counter
import math
import re
from .schemas import Observation, Selection


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


def approximate_tokens(text: str) -> int:
    """Development proxy ONLY. Final experiments must use the monitor tokenizer."""
    return len(re.findall(r"\w+|[^\w\s]", text))


class HFTokenCounter:
    def __init__(self, name="microsoft/MiniLM-L12-H384-uncased"):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(name)

    def __call__(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False))


def bm25(query, documents):
    """Okapi BM25, k1=1.5 and b=0.75; stable ties handled by caller."""
    if not documents:
        return []
    counts = [Counter(words(d)) for d in documents]
    lengths = [sum(c.values()) for c in counts]
    mean = sum(lengths) / len(counts) or 1
    scores = [0.0] * len(counts)
    for term in set(words(query)):
        df = sum(term in c for c in counts)
        idf = math.log(1 + (len(counts) - df + .5) / (df + .5))
        for i, c in enumerate(counts):
            tf = c[term]
            scores[i] += idf * tf * 2.5 / (tf + 1.5 * (.25 + .75 * lengths[i] / mean))
    return scores


def context(o: Observation, events) -> str:
    """A single serializer is used by packing, training and inference."""
    return (f"TASK {o.task}\nPOLICY {o.policy}\nACTION {o.proposal}\nHISTORY\n" +
            "\n".join(e.text() for e in events))


def select(o: Observation, variant: str, counter=approximate_tokens,
           budget=256, scorer=None, temporal=True) -> Selection:
    events = list(o.events)
    if variant == "H0":
        return Selection([], [], 0)
    query = f"{o.task} {o.policy} {o.proposal}"
    scores = bm25(query, [e.text() for e in events])
    rank = sorted(range(len(events)), key=lambda i: (
        -scores[i], -(events[i].resource_id == o.proposal.resource_id), events[i].sequence_no))
    if variant == "H1":
        rank = sorted(range(len(events)), key=lambda i: -events[i].sequence_no)
    elif variant in {"H3", "H4"}:
        if scorer is None:
            raise ValueError("H3/H4 require a trained scorer; no silent rule substitute")
        candidates = rank[:12] if variant == "H3" else list(range(len(events)))
        learned = scorer(query, [events[i].text() for i in candidates])
        rank = [i for _, i in sorted(zip(learned, candidates), key=lambda x: (-x[0], x[1]))]
    elif variant != "H2":
        raise ValueError(f"Unknown variant: {variant}")
    selected = set()
    incomplete = False
    for i in rank:
        e = events[i]
        group = {i}
        if temporal and e.kind in {"approval", "check"}:
            # Expand updates only from observed history. Hidden annotations are not used.
            group |= {j for j, other in enumerate(events)
                      if other.sequence_no > e.sequence_no and
                      (other.kind, other.resource_id, other.version, other.scope) ==
                      (e.kind, e.resource_id, e.version, e.scope)}
        candidate = sorted(selected | group, key=lambda j: events[j].sequence_no)
        text = "\n".join(events[j].text() for j in candidate)
        if counter(text) <= budget:
            selected.update(group)
        elif len(group) > 1:
            incomplete = True
    packed = [events[j] for j in sorted(selected, key=lambda j: events[j].sequence_no)]
    return Selection([events[i].event_id for i in rank], packed,
                     counter("\n".join(e.text() for e in packed)), incomplete,
                     {"token_counter": getattr(counter, "__name__", type(counter).__name__)})
