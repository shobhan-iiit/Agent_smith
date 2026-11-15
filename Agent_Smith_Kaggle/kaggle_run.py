"""Train MiniLM models, then freeze and evaluate the synthetic family holdout."""

import argparse
from collections import defaultdict
from dataclasses import replace
import hashlib
import itertools
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from zipfile import ZipFile

import numpy as np
from importlib.metadata import version

import dataset as ds
from experiment import digest, load_split, paired_delta, ranking_metrics, rows, summarize, verify_files, write
from neural import Encoder, LIMIT, MODEL, Predictor, Settings, fit
from sandbox import evaluate_fixtures, evaluate_policy_rag

LABELS = list(ds.LABELS)
VARIANTS = ("H0", "H1", "H2", "H3", "H4")
BUDGET = 256
QWEN = "Qwen/Qwen2.5-1.5B-Instruct"
BASE_FILES = ["config.json", "pytorch_model.bin", "model.safetensors", "vocab.txt",
              "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]


def query(case, policy=True):
    parts = [case["task"], json.dumps(case["proposal"], sort_keys=True)]
    if policy:
        parts.append(ds.render(case, []).split("\nPOLICY ", 1)[1].split("\nACTION ", 1)[0])
    return "\n".join(parts)


def pair_text(case, event, policy=True):
    return query(case, policy) + "\nEVENT\n" + ds.render_event(event)


def selector_rows(cases, labels, policy=True):
    for case in cases:
        if len(case["events"]) != 8:
            continue
        relevant = set(labels[case["case_id"]]["relevance_event_ids"])
        for event in case["events"]:
            yield {"text": pair_text(case, event, policy), "family_id": case["family_id"],
                   "label": int(event["event_id"] in relevant)}


def key(event):
    return tuple(event[k] for k in ("resource_id", "scope", "version", "attribute"))


class PreparedCases:
    def __init__(self, cases, encoder):
        self.cases = cases
        self.encoder = encoder
        self.docs = [[ds.render_event(event) for event in case["events"]] for case in cases]
        flat = list(itertools.chain.from_iterable(self.docs))
        sizes = iter(map(len, encoder.encode(flat, specials=False)))
        self.sizes = [[next(sizes) for _ in docs] for docs in self.docs]
        self.headers = list(map(len, encoder.encode([ds.render(c, []) for c in cases])))
        self.lexical = [self.lexical_order(c, d) for c, d in zip(cases, self.docs)]
        self.timings = {}

    @staticmethod
    def lexical_order(case, docs):
        scores = ds.bm25(query(case), docs)
        return sorted(range(len(docs)), key=lambda i: (-scores[i],
                      case["events"][i]["resource_id"] != case["proposal"]["resource_id"], i))

    def learned_ranks(self, predictor, shortlist=None, policy=True):
        orders = [list(range(len(d))) if shortlist is None else r[:shortlist]
                  for d, r in zip(self.docs, self.lexical)]
        start = perf_counter()
        texts = [pair_text(c, c["events"][i], policy) for c, order in zip(self.cases, orders) for i in order]
        scores = iter(predictor.probabilities(texts)[:, 1])
        ranks = [sorted(order, key=lambda i: (-next(scores), i)) for order in orders]
        self.timings[(shortlist, policy)] = {"pairs": len(texts), "seconds": perf_counter() - start}
        return ranks

    def pack(self, index, order, *, grouping=True, recent=False):
        case = self.cases[index]
        sizes = self.sizes[index]
        available = min(BUDGET, LIMIT - self.headers[index])
        if available < 0:
            raise ValueError("Required task, policy and action exceed the context window")
        groups = defaultdict(set)
        for i, event in enumerate(case["events"]):
            groups[key(event)].add(i)
        selected = set()
        used = 0
        for i in order:
            add = (groups[key(case["events"][i])] if grouping else {i}) - selected
            size = sum(sizes[j] for j in add)
            if used + size <= available:
                selected.update(add)
                used += size
            elif recent:
                break
        events = [case["events"][j] for j in sorted(selected)]
        return events, used

    def contexts(self, labels, scorer, variants=VARIANTS, no_policy=None, full_rank=None):
        result = []
        ranks = {}
        if any(v in variants for v in ("H3", "H3_ungrouped")):
            if full_rank is None:
                ranks["H3"] = self.learned_ranks(scorer, 12)
            else:
                shortlist = [set(order[:12]) for order in self.lexical]
                ranks["H3"] = [[i for i in full if i in candidates]
                               for full, candidates in zip(full_rank, shortlist)]
        if "H4" in variants:
            ranks["H4"] = full_rank if full_rank is not None else self.learned_ranks(scorer)
        if "H3_no_policy" in variants:
            ranks["H3_no_policy"] = self.learned_ranks(no_policy, 12, False)
        for i, case in enumerate(self.cases):
            annotation = labels[case["case_id"]]
            ids = [e["event_id"] for e in case["events"]]
            orders = {"H0": [], "H1": list(reversed(range(len(ids)))), "H2": self.lexical[i]}
            orders.update({v: r[i] for v, r in ranks.items()})
            if "H3" in orders:
                orders["H3_ungrouped"] = orders["H3"]
            for v in variants:
                order = orders[v]
                events, count = self.pack(i, order, grouping=v != "H3_ungrouped", recent=v == "H1")
                selected = [e["event_id"] for e in events]
                ranked = [ids[j] for j in order]
                recall, ndcg = ranking_metrics(ranked, annotation["relevance_event_ids"])
                sets = annotation["sufficient_sets"]
                visible = ds.decide(case["policy"], case["proposal"], events)
                result.append(dict(text=ds.render(case, events), case_id=case["case_id"],
                    family_id=case["family_id"], history_length=len(ids), variant=v,
                    label=ds.LABELS[visible], visible=visible, full=annotation["verdict"],
                    history_tokens=count, selected=selected, ranked=ranked, recall5=recall, ndcg5=ndcg,
                    coverage=any(set(s) <= set(selected) for s in sets) if sets else None,
                    shortlist_recall12=(len(set(ids[j] for j in self.lexical[i][:12]) & set(annotation["relevance_event_ids"]))
                        / len(annotation["relevance_event_ids"])) if annotation["relevance_event_ids"] else None))
        sizes = list(map(len, self.encoder.encode([r["text"] for r in result])))
        histories = list(map(len, self.encoder.encode(
            [r["text"].split("\nHISTORY\n", 1)[1] for r in result], specials=False)))
        for row, size, history in zip(result, sizes, histories):
            if size > LIMIT or history > BUDGET or history != row["history_tokens"]:
                raise ValueError(f"Token-budget or token-accounting violation: {row['case_id']}")
            row["total_tokens"] = size
        return result


def fit_selector(train, labels, val, val_labels, path, settings, encoder, policy=True):
    fit(list(selector_rows(train, labels, policy)), list(selector_rows(val, val_labels, policy)),
        path, ["irrelevant", "relevant"], settings, encoder)


def policy_training_rows(split, corpus):
    texts = [p["text"] for p in corpus]
    passages = {p["passage_id"]: p for p in corpus}
    for row in rows(split, "policy_rag/queries.jsonl"):
        q = row["query"] + " missing evidence "
        scores = ds.bm25(q, texts)
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:9]
        ids = {corpus[i]["passage_id"] for i in order} | set(row["relevant_passage_ids"])
        for passage_id in sorted(ids):
            yield {"text": q + "\nPASSAGE\n" + passages[passage_id]["text"],
                   "family_id": row["family_id"], "label": int(passage_id in row["relevant_passage_ids"])}


def train_seed(train, labels, val, val_labels, folds, output, settings, encoder, ablations):
    for fold in folds:
        dest = output / f"fold_{fold['fold']}"
        training = set(fold["fit_train_families"])
        validating = set(fold["fit_validation_families"])
        excluded = set(fold["out_of_fold_families"])
        if training & validating or training & excluded or validating & excluded:
            raise ValueError("Cross-fitting family leakage")
        fit_selector([c for c in train if c["family_id"] in training], labels,
                     [c for c in train if c["family_id"] in validating], labels, dest, settings, encoder)

    fit_selector(train, labels, val, val_labels, output / "selector", settings, encoder)
    if ablations:
        fit_selector(train, labels, val, val_labels, output / "selector_no_policy", settings, encoder, False)
        fit_selector(train, labels, val, val_labels, output / "selector_frozen", replace(settings, adaptation="head"), encoder)

    for fold in folds:
        path = output / f"fold_contexts_{fold['fold']}.jsonl"
        if path.is_file():
            continue
        selected = set(fold["out_of_fold_families"])
        prepared = PreparedCases([c for c in train if c["family_id"] in selected], encoder)
        predictor = Predictor(output / f"fold_{fold['fold']}")
        records = prepared.contexts(labels, predictor, full_rank=prepared.learned_ranks(predictor))
        write_contexts(path, records)
        print(f"Saved {len(records)} out-of-fold contexts to {path.name}", flush=True)

    path = output / "validation_contexts.jsonl"
    if not path.is_file():
        prepared = PreparedCases(val, encoder)
        predictor = Predictor(output / "selector")
        write_contexts(path, prepared.contexts(val_labels, predictor, full_rank=prepared.learned_ranks(predictor)))
    training_rows = [row for fold in folds for row in rows(output, f"fold_contexts_{fold['fold']}.jsonl")]
    validation_rows = rows(output, path.name)
    if len(training_rows) != len(train) * len(VARIANTS) or len({(r["case_id"], r["variant"]) for r in training_rows}) != len(training_rows):
        raise ValueError("Out-of-fold monitor contexts are missing or duplicated")
    # Empty histories can render identically across families. Exclude exact text overlaps.
    validation_texts = {row["text"] for row in validation_rows}
    fit_rows = [row for row in training_rows if row["text"] not in validation_texts]
    write(output / "monitor_input_counts.json", {"fit": len(fit_rows), "validation": len(validation_rows),
          "removed_exact_validation_overlap": len(training_rows) - len(fit_rows)})
    fit(fit_rows, validation_rows, output / "monitor", LABELS, settings, encoder)
    if ablations:
        fit(fit_rows, validation_rows, output / "monitor_frozen", LABELS, replace(settings, adaptation="head"), encoder)


def write_contexts(path, records):
    temporary = path.with_suffix(".tmp")
    ds.write_jsonl(temporary, records)
    temporary.replace(path)


def results_for(records, predictor, out, split, seed):
    scores = predictor.probabilities([r["text"] for r in records])
    for r, probabilities in zip(records, scores):
        r["probabilities"] = dict(zip(LABELS, map(float, probabilities)))
        r["predicted"] = LABELS[int(probabilities.argmax())]
        r["seed"] = seed
    write_contexts(out / f"{split}_predictions.jsonl", records)


def evaluate(root, split, out, encoder, seed, ablations):
    cases, labels = load_split(root, split)
    prepared = PreparedCases(cases, encoder)
    selector = Predictor(out / "selector")
    monitor = Predictor(out / "monitor")
    variants = VARIANTS + (("H3_ungrouped", "H3_no_policy") if ablations else ())
    no_policy = Predictor(out / "selector_no_policy") if ablations else None
    records = prepared.contexts(labels, selector, variants, no_policy, full_rank=prepared.learned_ranks(selector))
    del selector, no_policy
    results_for(records, monitor, out, split, seed)
    if ablations:
        head_selector = Predictor(out / "selector_frozen")
        head_records = prepared.contexts(labels, head_selector, ("H3",))
        del head_selector
        results_for(head_records, monitor, out, split + "_selector_frozen", seed)
        for record in head_records:
            record["variant"] = "H3_selector_frozen"
        head_monitor = Predictor(out / "monitor_frozen")
        monitor_records = [dict(r, variant="H3_monitor_frozen") for r in records if r["variant"] == "H3"]
        results_for(monitor_records, head_monitor, out, split + "_monitor_frozen", seed)
        records.extend(head_records + monitor_records)

    def aggregate(items):
        summary = summarize([dict(r, selection_ms=0) for r in items])
        summary.pop("selection_ms_p50")
        summary.pop("selection_ms_p95")
        summary["mean_bm25_candidate_recall12"] = float(np.mean(
            [r["shortlist_recall12"] for r in items if r["shortlist_recall12"] is not None]))
        return summary

    metrics = {v: aggregate([r for r in records if r["variant"] == v])
               for v in sorted({r["variant"] for r in records})}
    pairs = rows(root, "private/counterfactual_pairs.jsonl")
    for variant, summary in metrics.items():
        predictions = {r["case_id"]: r["predicted"] for r in records if r["variant"] == variant}
        summary["counterfactual_both_correct"] = sum(predictions[p["safe_case"]] == "allow" and
            predictions[p["unsafe_case"]] == "block" for p in pairs) / len(pairs)
    grouped = [r for r in records if r["variant"] == "H3"]
    write(out / f"{split}_metrics.json", {"variants": metrics,
          "H3_by_history_length": {n: aggregate([r for r in grouped if r["history_length"] == n])
                                   for n in (8, 16, 32)},
          "H3_by_family": {f: aggregate([r for r in grouped if r["family_id"] == f])
                           for f in sorted({r["family_id"] for r in grouped})},
          "paired_accuracy": {f"{v}_vs_{b}": paired_delta(records, b, v)
                              for b, v in (("H1", "H2"), ("H2", "H3"), ("H3", "H4"))},
          "batch_scoring_diagnostics": {str(k): v for k, v in prepared.timings.items()}})
    rule_reference = [dict(case_id=c["case_id"], family_id=c["family_id"], expected=labels[c["case_id"]]["verdict"],
                           predicted=ds.decide(c["policy"], c["proposal"], c["events"])) for c in cases]
    write_contexts(out / f"{split}_full_rules.jsonl", rule_reference)
    print(split, {v: x["full_accuracy"]["rate"] for v, x in metrics.items()}, flush=True)


def safe_extract(archive_path, root):
    root = root.resolve()
    with ZipFile(archive_path) as archive:
        for name in archive.namelist():
            if not (root / name).resolve().is_relative_to(root):
                raise ValueError(f"Invalid archive path: {name}")
        archive.extractall(root)


def checkpoint_hashes(output):
    return {str(p.relative_to(output)): digest(p)
            for p in sorted(output.glob("seed_*/*/*"))
            if p.is_file() and p.name in ("checkpoint.json", "complete.json", "head.safetensors",
                                         "adapter_config.json", "adapter_model.safetensors", "tokenizer.json")}


def main():
    import torch
    from huggingface_hub import snapshot_download

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("train", "evaluate"))
    parser.add_argument("--archive", type=Path, default=Path("Agent_Smith_Dataset_v1.zip"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, nargs="+", default=[17, 42])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--skip-ablations", action="store_true")
    parser.add_argument("--skip-live", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("Enable a Kaggle GPU accelerator before running.")
    if args.epochs < 1 or args.batch < 1 or len(set(args.seed)) != len(args.seed):
        parser.error("Expected positive epochs and batch size and unique seeds")
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    data = args.out / "input" / "Agent_Smith_Dataset_v1"
    if not (data / "development" / "checksums.json").is_file():
        safe_extract(args.archive, data.parent)
    dev = data / "development"
    verify_files(dev)
    config = {"python_hash_seed": os.environ.get("PYTHONHASHSEED"), "dataset_archive_sha256": digest(args.archive),
              "development_sha256": digest(dev / "checksums.json"), "seeds": args.seed,
              "epochs": args.epochs, "batch": args.batch, "ablations": not args.skip_ablations,
              "live": not args.skip_live,
              "software": {name: version(name) for name in ("torch", "transformers", "peft", "tokenizers")},
              "history_tokens": BUDGET, "total_tokens": LIMIT,
              "source_hashes": {name: digest(Path(__file__).with_name(name))
                                for name in ("kaggle_run.py", "neural.py", "dataset.py", "experiment.py", "sandbox.py", "live.py")}}
    run_file = args.out / "run.json"
    if run_file.exists():
        old = json.loads(run_file.read_text())
        base = old.pop("base")
        revision = old.pop("base_revision")
        old_qwen_revision = old.pop("qwen_revision")
        if old != config:
            raise ValueError("Existing run configuration differs. Use a new --out directory.")
        qwen_revision = old_qwen_revision
        if not Path(base).is_dir():
            base = snapshot_download(MODEL, revision=revision, allow_patterns=BASE_FILES)
    elif args.phase == "train":
        base = snapshot_download(MODEL, allow_patterns=BASE_FILES)
        revision = Path(base).name
        from huggingface_hub import model_info
        qwen_revision = model_info(QWEN).sha if not args.skip_live else None
        write(run_file, {**config, "base": base, "base_revision": revision, "qwen_revision": qwen_revision})
    else:
        parser.error("Run the train phase before evaluation.")

    os.environ["AGENT_SMITH_MINILM_PATH"] = base
    encoder = Encoder(base)
    train, labels = load_split(dev / "train", "train")
    val, val_labels = load_split(dev / "validation", "validation")
    train_families = {c["family_id"] for c in train}
    val_families = {c["family_id"] for c in val}
    if train_families & val_families:
        raise ValueError("Training and validation families overlap")
    freeze = args.out / "freeze.json"
    if args.phase == "train":
        folds = json.loads((dev / "crossfit_folds.json").read_text())
        excluded = [f for fold in folds for f in fold["out_of_fold_families"]]
        if len(excluded) != len(set(excluded)) or set(excluded) != train_families:
            raise ValueError("Cross-fit excluded families must partition the training split")
        for seed in args.seed:
            train_seed(train, labels, val, val_labels, folds, args.out / f"seed_{seed}",
                       Settings(base, seed, args.epochs, args.batch), encoder, not args.skip_ablations)
        if not args.skip_live:
            corpus = rows(dev, "policy_corpus.jsonl")
            fit(list(policy_training_rows(dev / "train", corpus)),
                list(policy_training_rows(dev / "validation", corpus)),
                args.out / f"seed_{args.seed[0]}" / "policy_reranker",
                ["irrelevant", "relevant"], Settings(base, args.seed[0], args.epochs, args.batch), encoder)
        write(freeze, {"run_sha256": digest(run_file), "checkpoints": checkpoint_hashes(args.out),
                       "test_archive_sha256": digest(data / "sealed_test" / "Agent_Smith_Heldout_Test.zip")})
        return

    if not freeze.is_file():
        parser.error("Training has not produced a freeze manifest.")
    manifest = json.loads(freeze.read_text())
    test_zip = data / "sealed_test" / "Agent_Smith_Heldout_Test.zip"
    if (manifest["run_sha256"] != digest(run_file) or manifest["checkpoints"] != checkpoint_hashes(args.out)
            or manifest["test_archive_sha256"] != digest(test_zip)):
        raise ValueError("Frozen configuration, model files or test archive changed")
    with TemporaryDirectory() as temp:
        root = Path(temp)
        safe_extract(test_zip, root)
        verify_files(root)
        test = root / "test"
        test_cases, _ = load_split(test, "test")
        if {c["family_id"] for c in test_cases} & (train_families | val_families):
            raise ValueError("Test family overlap")
        for seed in args.seed:
            destination = args.out / f"seed_{seed}"
            evaluate(dev / "validation", "validation", destination, encoder, seed, not args.skip_ablations)
            evaluate(test, "test", destination, encoder, seed, not args.skip_ablations)
        supporting = evaluate_fixtures(test, args.out, "test")
        supporting["policy_rag"] = evaluate_policy_rag(test, dev / "policy_corpus.jsonl")
        write(args.out / "test_sandbox_metrics.json", supporting)
        if not args.skip_live:
            from live import evaluate_live
            qwen = snapshot_download(QWEN, revision=qwen_revision, allow_patterns=[
                "*.safetensors", "*.json", "*.txt", "*.jinja"])
            reference = args.out / f"seed_{args.seed[0]}"
            evaluate_live(test, dev / "policy_corpus.jsonl", args.out, qwen,
                          Predictor(reference / "monitor"), Predictor(reference / "policy_reranker"),
                          Predictor(reference / "selector"))


if __name__ == "__main__":
    main()
