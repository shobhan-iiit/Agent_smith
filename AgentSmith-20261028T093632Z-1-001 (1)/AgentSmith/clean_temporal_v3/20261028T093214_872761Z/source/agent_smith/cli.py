"""Runnable CPU development pipeline. Model-based results are never implied."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from .data import examples, export
from .retrieval import select
from .monitor import verdict
from .metrics import coverage, recall, ndcg, summarize
from .simulator import Controller


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate")
    generate.add_argument("--output", default="data")
    replay = commands.add_parser("replay")
    replay.add_argument("--output", default="results/development")
    replay.add_argument("--variants", nargs="+", choices=["H0", "H1", "H2"], default=["H0", "H1", "H2"])
    replay.add_argument("--budget", type=int, default=256)
    commands.add_parser("demo")
    args = parser.parse_args()
    if args.command == "generate":
        print(json.dumps({"instances": export(args.output), "split": "development"}))
        return
    variants = args.variants if args.command == "replay" else ["H1", "H2"]
    rows = []
    cases = list(examples())
    if args.command == "demo":
        cases = [x for x in cases if x[0].case_id == "publish_report_supersession_0_1_n32"]
    for o, a, state in cases:
        for variant in variants:
            start = time.perf_counter()
            selected = select(o, variant, budget=getattr(args, "budget", 256))
            elapsed = (time.perf_counter() - start) * 1000
            predicted = verdict(o, selected.events)
            result = Controller(state.clone()).act(o.proposal, predicted)
            relevant = set().union(*map(set, a.sufficient_sets))
            rows.append({"case_id": o.case_id, "family_id": a.family_id,
                         "base_case_id": a.base_case_id, "variant": variant,
                         "history_length": len(o.events), "unsafe": a.violates,
                         "monitor": predicted, "controller": result["decision"],
                         "executed": result["executed"], "packed_ids": [e.event_id for e in selected.events],
                         "coverage": coverage([e.event_id for e in selected.events], a.sufficient_sets),
                         "recall5": recall(selected.ranked_ids, relevant, 5),
                         "ndcg5": ndcg(selected.ranked_ids, relevant, 5),
                         "history_tokens": selected.tokens, "selection_ms": elapsed})
    summary = {v: summarize([r for r in rows if r["variant"] == v]) for v in variants}
    payload = {"stage": "development", "monitor": "transparent_rule_debug_baseline",
               "token_counter": "approximate_NOT_final_monitor_tokenizer", "summary": summary}
    if args.command == "replay":
        root = Path(args.output)
        root.mkdir(parents=True, exist_ok=True)
        # Atomic replacement prevents partially written outputs after an interruption.
        content = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
        temporary = root / "predictions.tmp"
        temporary.write_text(content)
        temporary.replace(root / "predictions.jsonl")
        payload["prediction_sha256"] = hashlib.sha256(content.encode()).hexdigest()
        (root / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    else:
        payload["examples"] = rows
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
