"""Bounded, in-memory workflow prototype for the synthetic support fixtures."""

from dataclasses import dataclass, field
import json
from math import log2
from time import perf_counter

import dataset as ds
from experiment import fraction, indexed, rows


@dataclass
class EvidenceTool:
    task: dict
    records_by_arguments: dict
    budget: int = 2
    fail_once: bool = False
    audit: list = field(default_factory=list)

    def call(self, name, args):
        schema = {"resource_id": str, "scope": str, "attribute": str, "version": int}
        action = self.task["proposed_action"]
        records = []
        if len(self.audit) >= self.budget:
            return {"status": "budget_exceeded", "records": records}
        if (name != "lookup_evidence" or type(args) is not dict or set(args) != set(schema)
                or any(type(args[key]) is not kind for key, kind in schema.items())):
            status = "invalid_arguments"
        elif (args["resource_id"], args["scope"], args["version"]) != (
                action["resource_id"], action["scope"], action["arguments"]["version"]):
            status = "unauthorized_scope"
        elif self.fail_once:
            self.fail_once = False
            status = "temporary_error"
        else:
            key = json.dumps(args, sort_keys=True)
            records = self.records_by_arguments.get(key, [])
            status = "ok" if records else "not_found"
        self.audit.append({"tool": name, "arguments": args, "status": status,
                           "event_ids": [event["event_id"] for event in records]})
        return {"status": status, "records": records}


def execute_fixture(task, store, investigate):
    action = task["proposed_action"]
    policy = task["policy"]
    events = {e["event_id"]: e for e in task["visible_history"]}
    if type(action["arguments"].get("version")) is not int or not all(
            isinstance(action.get(key), str) and action[key] for key in ("tool", "scope", "resource_id")):
        raise ValueError("Invalid mock action")
    initial = ds.decide(policy, action, list(events.values()))
    verdict = initial
    tool = EvidenceTool(task, store, budget=task["max_investigation_queries"])
    if investigate and verdict == "defer":
        for requirement in policy["requirements"]:
            field = requirement["field"]
            if ds.current_value(list(events.values()), field, action) is not ds.UNRESOLVED:
                continue
            args = {"attribute": field, "resource_id": action["resource_id"],
                    "scope": action["scope"], "version": action["arguments"]["version"]}
            response = tool.call("lookup_evidence", args)
            events.update((e["event_id"], e) for e in response["records"])
            verdict = ds.decide(policy, action, list(events.values()))
            if verdict != "defer":
                break
    # The simulator applies one action to its local mock ledger.
    effect = dict(action) if verdict == "allow" else None
    return {"task_id": task["task_id"], "initial": initial, "verdict": verdict,
            "executed": effect is not None, "mock_effect": effect, "queries": tool.audit,
            "evidence_ids": [e["event_id"] for e in events.values() if ds.matches(e, action)]}


def evaluate_fixtures(root, out, split):
    tasks = indexed(root, "live/public_tasks.jsonl", "task_id")
    expected = indexed(root, "live/private_expectations.jsonl", "task_id")
    stores = {task_id: {} for task_id in tasks}
    for row in rows(root, "live/private_tool_responses.jsonl"):
        stores[row["task_id"]][json.dumps(row["arguments"], sort_keys=True)] = row["result"]["records"]
    outcomes = {}
    for variant, investigate in [("without_investigation", False), ("bounded_rules", True)]:
        traces = [execute_fixture(task, stores[task_id], investigate) for task_id, task in tasks.items()]
        for trace in traces:
            trace["expected"] = expected[trace["task_id"]]["expected_verdict"]
            trace["variant"] = variant
        safe = sum(t["expected"] == "allow" for t in traces)
        n = len(traces)
        outcomes[variant] = {
            "correct_verdicts": fraction(sum(t["verdict"] == t["expected"] for t in traces), n),
            "safe_mock_execution": fraction(sum(t["executed"] and t["expected"] == "allow" for t in traces), safe),
            "unsafe_mock_execution": fraction(sum(t["executed"] and t["expected"] != "allow" for t in traces), n),
            "query_attempts": sum(len(t["queries"]) for t in traces),
            "deferrals": fraction(sum(t["verdict"] == "defer" for t in traces), n),
        }
        ds.write_jsonl(out / f"{split}_{variant}.jsonl", traces)

    probes = rows(root, "tool_calling/public_cases.jsonl")
    labels = indexed(root, "tool_calling/private_labels.jsonl", "example_id")
    traces = []
    for probe in probes:
        task_id = probe["task_id"]
        expected_probe = labels[probe["example_id"]]
        tool = EvidenceTool(tasks[task_id], stores[task_id], fail_once=expected_probe["injected_server_fault"])
        call = probe["tool_call"]
        first = tool.call(call["name"], call["arguments"])["status"]
        repaired = None
        if first == "temporary_error":
            repaired = tool.call(call["name"], call["arguments"])["status"]
        traces.append(dict(example_id=probe["example_id"], first=first, repaired=repaired,
                           expected=expected_probe["expected_server_status"], queries=tool.audit))
    ds.write_jsonl(out / f"{split}_tool_probes.jsonl", traces)
    transients = [trace for trace in traces if trace["expected"] == "temporary_error"]
    outcomes["tool_probes"] = {
        "first_status_matches": fraction(sum(t["first"] == t["expected"] for t in traces), len(traces)),
        "transient_recovery": fraction(sum(t["repaired"] == "ok" for t in transients), len(transients)),
    }
    return outcomes


def evaluate_policy_rag(root, corpus_file):
    passages = rows(corpus_file.parent, corpus_file.name)
    texts = [p["text"] for p in passages]
    queries = rows(root, "policy_rag/queries.jsonl")
    matches = []
    times = []
    for query in queries:
        start = perf_counter()
        scores = ds.bm25(query["query"], texts)
        ranked = sorted(range(len(passages)), key=lambda i: (-scores[i], i))[:3]
        times.append((perf_counter() - start) * 1000)
        ids = [passages[i]["passage_id"] for i in ranked]
        relevant = query["relevant_passage_ids"][0]
        matches.append(1 / log2(ids.index(relevant) + 2) if relevant in ids else 0)
    return {"passages": len(passages), "no_retrieval_recall3": fraction(0, len(queries)),
            "bm25_recall3": fraction(sum(gain > 0 for gain in matches), len(queries)),
            "bm25_ndcg3": sum(matches) / len(matches),
            "mean_query_ms": sum(times) / len(times)}


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    from zipfile import ZipFile
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("Agent_Smith_Dataset_v1.zip"))
    args = parser.parse_args()
    with ZipFile(args.archive) as archive:
        def read(name):
            matches = [path for path in archive.namelist() if path.endswith("/development/validation/" + name)]
            return [json.loads(line) for line in archive.read(matches[0]).splitlines()]
        tasks = read("live/public_tasks.jsonl")
        evidence = read("live/private_tool_responses.jsonl")
        for task in tasks:
            store = {json.dumps(row["arguments"], sort_keys=True): row["result"]["records"]
                     for row in evidence if row["task_id"] == task["task_id"]}
            baseline = execute_fixture(task, store, False)
            investigated = execute_fixture(task, store, True)
            if baseline["verdict"] != investigated["verdict"]:
                print(json.dumps({"without_investigation": baseline, "with_investigation": investigated}, indent=2))
                break
        else:
            print(json.dumps({"with_investigation": investigated}, indent=2))
