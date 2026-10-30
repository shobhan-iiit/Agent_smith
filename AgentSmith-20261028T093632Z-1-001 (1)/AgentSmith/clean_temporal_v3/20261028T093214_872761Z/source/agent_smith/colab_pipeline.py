"""Colab pilot: explicit data preparation, cross-fitted contexts and validation.

This uses SIX development templates, not the final research corpus. Labels are
simulator-derived and require human review. No final test is created or evaluated.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from .data import examples
from .schemas import Event, Observation, Proposal
from .retrieval import select, context
from .learning import MODEL, fit, Predictor
from .monitor import verdict
from .metrics import coverage, recall, ndcg

LABELS = {"allow": 0, "block": 1, "defer": 2}


def read(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def write(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in rows))


def observation(row):
    return Observation(row["case_id"], row["task"], row["policy"],
                       tuple(Event(**x) for x in row["events"]), Proposal(**row["proposal"]))


def query(o):
    return f"{o.task} {o.policy} {o.proposal}"


def prepare(root):
    root = Path(root)
    public, annotations, selector = [], [], []
    # One family per category in each pilot split. Never call this a final split.
    for o, a, _ in examples():
        split = "validation" if "supersession" in a.family_id else "train"
        public.append(asdict(o) | {"family_id": a.family_id, "base_case_id": a.base_case_id,
                                  "split": split})
        annotations.append(asdict(a) | {"case_id": o.case_id, "split": split})
        # The singleton sufficient set is valid for these deliberately simple pilot templates.
        relevant = set().union(*map(set, a.sufficient_sets))
        # Use the shortest rendering only: do not overweight repeated length variants.
        if len(o.events) == 8:
            for e in o.events:
                selector.append({"text": query(o) + "\nEVENT\n" + e.text(),
                                 "label": int(e.event_id in relevant), "split": split,
                                 "family_id": a.family_id, "case_id": o.case_id})
    write(root / "public/pilot.jsonl", public)
    write(root / "private/labels.jsonl", annotations)
    for split in ["train", "validation"]:
        write(root / f"selector/{split}.jsonl", [r for r in selector if r["split"] == split])
    manifest = {"stage": "pilot_only", "review_status": "provisional_simulator_labels",
                "instances": len(public), "families": 6, "test_instances": 0,
                "monitor_label_map": LABELS,
                "splits": {s: sorted({r["family_id"] for r in public if r["split"] == s})
                           for s in ["train", "validation"]}}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    # Human-readable review sheet; labels remain private to model components.
    import csv
    with (root / "review.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case_id", "family_id", "split", "violates", "sufficient_sets", "reviewer", "approved", "notes"])
        for a in annotations:
            writer.writerow([a["case_id"], a["family_id"], a["split"], a["violates"],
                             json.dumps(a["sufficient_sets"]), "", "", ""])
    print(json.dumps(manifest, indent=2))


def run(root, artifacts, smoke=False):
    from transformers import AutoTokenizer
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Enable a GPU in Colab before running training")
    root, artifacts = Path(root), Path(artifacts)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    count = lambda text: len(tokenizer.encode(text, add_special_tokens=False))
    epochs = 1 if smoke else 3
    rows = read(root / "public/pilot.jsonl")
    train = read(root / "selector/train.jsonl")
    validation = read(root / "selector/validation.jsonl")
    families = sorted({r["family_id"] for r in train})
    # Three pilot training families -> leave-one-family-out, three folds.
    # Fold checkpoint selection uses ONLY a deterministic fitting-data holdout;
    # excluded family labels never select a checkpoint.
    predictors = {}
    for index, excluded in enumerate(families):
        fitting = [r for r in train if r["family_id"] != excluded]
        # A dedicated family is reserved for fit() validation: tiny pilot only.
        fit_families = sorted({r["family_id"] for r in fitting})
        holdout = fit_families[-1]
        fold_train = [dict(r, split="train") for r in fitting if r["family_id"] != holdout]
        fold_val = [dict(r, split="validation") for r in fitting if r["family_id"] == holdout]
        folder = artifacts / f"fold_{index}"
        write(folder / "train.jsonl", fold_train)
        write(folder / "validation.jsonl", fold_val)
        fit(folder / "train.jsonl", folder / "validation.jsonl", folder / "model", epochs=epochs)
        predictors[excluded] = str(folder / "model")
    final_selector = artifacts / "selector"
    fit(root / "selector/train.jsonl", root / "selector/validation.jsonl", final_selector, epochs=epochs)

    monitor_rows = []
    # Load only one fold model at a time; prevent accumulating models in GPU RAM.
    for split in ["train", "validation"]:
        targets = families if split == "train" else [None]
        for family in targets:
            checkpoint = predictors[family] if split == "train" else final_selector
            predictor = Predictor(checkpoint)
            for row in rows:
                if row["split"] != split or (family is not None and row["family_id"] != family):
                    continue
                o = observation(row)
                if len(tokenizer.encode(context(o, []))) > 256:
                    raise ValueError("Mandatory context exceeds reserved budget")
                for variant in ["H0", "H1", "H2", "H3", "H4"]:
                    selected = select(o, variant, counter=count, scorer=predictor)
                    text = context(o, selected.events)
                    if len(tokenizer.encode(text)) > 512:
                        raise ValueError("Full monitor input exceeds 512 tokens")
                    monitor_rows.append({"text": text, "label": LABELS[verdict(o, selected.events)],
                                         "split": split, "family_id": row["family_id"],
                                         "case_id": row["case_id"], "variant": variant,
                                         "selector_checkpoint": str(checkpoint)})
            del predictor
            torch.cuda.empty_cache()
    for split in ["train", "validation"]:
        write(root / f"monitor/{split}.jsonl", [r for r in monitor_rows if r["split"] == split])
    fit(root / "monitor/train.jsonl", root / "monitor/validation.jsonl",
        artifacts / "monitor", task="monitor", epochs=epochs)
    evaluate(root, artifacts)


def evaluate(root, artifacts):
    from transformers import AutoTokenizer
    from .simulator import Controller
    root, artifacts = Path(root), Path(artifacts)
    tokenizer = AutoTokenizer.from_pretrained(artifacts / "monitor")
    count = lambda text: len(tokenizer.encode(text, add_special_tokens=False))
    selector = Predictor(artifacts / "selector")
    monitor = Predictor(artifacts / "monitor")
    annotations = {r["case_id"]: r for r in read(root / "private/labels.jsonl")}
    outcomes = []
    for row in read(root / "public/pilot.jsonl"):
        if row["split"] != "validation":
            continue
        o = observation(row)
        a = annotations[o.case_id]
        for variant in ["H0", "H1", "H2", "H3", "H4"]:
            import time
            start = time.perf_counter()
            selection = select(o, variant, counter=count, scorer=selector)
            text = context(o, selection.events)
            probabilities = monitor.probabilities([text])[0]
            prediction = max(range(3), key=lambda i: probabilities[i])
            truth = LABELS[verdict(o, selection.events)]
            relevant = set().union(*map(set, a["sufficient_sets"]))
            outcomes.append({"case_id": o.case_id, "family_id": row["family_id"],
                             "variant": variant, "prediction": prediction, "target": truth,
                             "unsafe": a["violates"], "probabilities": probabilities,
                             "coverage": coverage([e.event_id for e in selection.events], a["sufficient_sets"]),
                             "recall5": recall(selection.ranked_ids, relevant, 5),
                             "ndcg5": ndcg(selection.ranked_ids, relevant, 5),
                             "total_tokens": len(tokenizer.encode(text)),
                             "latency_seconds": time.perf_counter() - start})
    write(artifacts / "validation_predictions.jsonl", outcomes)
    summaries = {}
    for variant in ["H0", "H1", "H2", "H3", "H4"]:
        group = [r for r in outcomes if r["variant"] == variant]
        summaries[variant] = {"cases": len(group),
                              "accuracy": sum(r["target"] == r["prediction"] for r in group) / len(group),
                              "evidence_coverage": sum(r["coverage"] for r in group) / len(group),
                              "mean_total_tokens": sum(r["total_tokens"] for r in group) / len(group)}
    (artifacts / "validation_summary.json").write_text(json.dumps(summaries, indent=2))
    print(json.dumps(summaries, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["prepare", "run", "evaluate"])
    p.add_argument("--data", default="data/colab_pilot")
    p.add_argument("--artifacts", default="artifacts/colab_pilot")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.command == "prepare":
        prepare(args.data)
    elif args.command == "run":
        run(args.data, args.artifacts, args.smoke)
    else:
        evaluate(args.data, args.artifacts)


if __name__ == "__main__":
    main()
