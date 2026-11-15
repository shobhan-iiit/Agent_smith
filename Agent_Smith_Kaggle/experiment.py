"""Train on development families, freeze artifacts, then evaluate the held-out split."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from time import perf_counter
from zipfile import ZipFile

import joblib
import numpy as np
import sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline

import dataset as ds

LABELS = tuple(ds.LABELS)
VARIANTS = ("H0", "H1", "H2", "H3", "H4", "H3_ungrouped", "H3_no_policy")
SEEDS = (17, 42)
BUDGET = 256
TOTAL = 512
SETTINGS = {"variants": VARIANTS, "seeds": SEEDS, "history_tokens": BUDGET,
            "total_tokens": TOTAL, "tokenizer": "dataset.words regex",
            "selector": "balanced logistic, C=1, liblinear",
            "monitor": "balanced logistic, C=1, liblinear OVR",
            "threshold": "argmax", "shortlist": 12}


def rows(root, file):
    return [json.loads(line) for line in (root / file).read_text().splitlines() if line]


def indexed(root, file, key):
    return {r[key]: r for r in rows(root, file)}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def event_key(event):
    return tuple(event[key] for key in ("resource_id", "scope", "version", "attribute"))


def event_features(case, event, use_policy=True):
    action = case["proposal"]
    features = {
        "resource": event["resource_id"] == action["resource_id"],
        "scope": event["scope"] == action["scope"],
        "version": event["version"] == action["arguments"]["version"],
        "position": event["sequence_no"] / len(case["events"]),
        "revision": event["revision"],
        "linked_updates": len(event["superseded_by"]),
        "concurrent": len(event["concurrent_records"]),
        "kind": event["kind"],
        "status": event["status"],
    }
    # A joint feature lets the linear scorer distinguish a complete key match.
    features["key_match"] = ds.matches(event, action)
    if use_policy:
        features["required_field"] = any(
            event["attribute"] == clause["field"] for clause in case["policy"]["requirements"]
        )
    return features


def policy_query(case):
    return " ".join((case["task"], json.dumps(case["proposal"], sort_keys=True),
                     json.dumps(case["policy"]["requirements"], sort_keys=True)))


def pack(case, variant, scorer=None):
    if variant == "H0":
        return [], []
    events = case["events"]
    docs = [ds.render_event(e) for e in events]
    if variant == "H1":
        rank = list(reversed(range(len(events))))
    else:
        lexical = ds.bm25(policy_query(case), docs)
        rank = sorted(range(len(events)), key=lambda i: (
            -lexical[i], events[i]["resource_id"] != case["proposal"]["resource_id"], i))
        if variant.startswith(("H3", "H4")):
            candidates = list(range(len(events))) if variant == "H4" else rank[:12]
            features = [event_features(case, events[i], variant != "H3_no_policy") for i in candidates]
            scores = scorer.predict_proba(features)[:, 1]
            rank = [i for _, i in sorted(zip(-scores, candidates))]
    groups = defaultdict(set)
    for i, event in enumerate(events):
        groups[event_key(event)].add(i)
    chosen = set()
    header = len(ds.words(ds.render(case, [])))
    if header > TOTAL:
        raise ValueError("Mandatory input exceeds the budget")
    for i in rank:
        group = {i} if variant == "H3_ungrouped" else groups[event_key(events[i])]
        if group <= chosen:
            continue
        combined = chosen | group
        history = "\n".join(docs[j] for j in sorted(combined))
        if len(ds.words(history)) <= min(BUDGET, TOTAL - header):
            chosen = combined
        elif variant == "H1":
            break
    return [events[i] for i in sorted(chosen)], [events[i]["event_id"] for i in rank]


def monitor_features(case, events):
    """Represent observed values without evaluating the policy or using labels."""
    features = {}
    for index, clause in enumerate(case["policy"]["requirements"]):
        prefix = f"{index}:"
        action = case["proposal"]
        value = ds.current_value(events, clause["field"], action)
        op = clause["op"]
        features[prefix + "operator"] = op
        available = value is not ds.UNRESOLVED
        features[prefix + "available"] = available
        if available:
            expected = ds.resolve(clause["expected"], action)
            features[prefix + "value"] = json.dumps(value, sort_keys=True) if isinstance(value, (str, bool)) else type(value).__name__
            features[prefix + "expected"] = json.dumps(expected, sort_keys=True) if isinstance(expected, (str, bool)) else type(expected).__name__
            if type(value) is type(expected) is int:
                delta = value - expected
                features[prefix + "numeric_delta"] = math.copysign(math.log1p(abs(delta)), delta)
    return features


def classifier(seed):
    return make_pipeline(DictVectorizer(), LogisticRegression(
        C=1, class_weight="balanced", solver="liblinear", max_iter=400, random_state=seed))


def fit_selector(cases, annotations, seed, use_policy=True):
    features, targets = [], []
    for case in cases:
        if len(case["events"]) == 8:
            relevant = set(annotations[case["case_id"]]["relevance_event_ids"])
            for event in case["events"]:
                features.append(event_features(case, event, use_policy))
                targets.append(int(event["event_id"] in relevant))
    return classifier(seed).fit(features, targets)


def load_split(root, split):
    cases = rows(root, "public/cases.jsonl")
    labels = indexed(root, "private/labels.jsonl", "case_id")
    ids = {c["case_id"] for c in cases}
    if len(ids) != len(cases) or ids != set(labels):
        raise ValueError("Invalid case/annotation correspondence")
    if any(c["split"] != split or labels[c["case_id"]]["split"] != split for c in cases):
        raise ValueError(f"Expected only {split} rows")
    return cases, labels


def verify_files(root):
    for name, checksum in json.loads((root / "checksums.json").read_text()).items():
        if digest(root / name) != checksum:
            raise ValueError(f"Dataset checksum mismatch: {name}")


def train(dev, out):
    verify_files(dev)
    cases, labels = load_split(dev / "train", "train")
    valid, _ = load_split(dev / "validation", "validation")
    families = {c["family_id"] for c in cases}
    if families & {c["family_id"] for c in valid}:
        raise ValueError("Development splits overlap")
    folds = json.loads((dev / "crossfit_folds.json").read_text())
    fold_map = {f: fold["fold"] for fold in folds for f in fold["out_of_fold_families"]}
    if set(fold_map) != families or sum(len(f["out_of_fold_families"]) for f in folds) != len(families):
        raise ValueError("Invalid cross-fit partition")
    models = {}
    for seed in SEEDS:
        fold_models = {}
        for fold in folds:
            excluded = set(fold["out_of_fold_families"])
            fold_models[fold["fold"]] = fit_selector(
                [case for case in cases if case["family_id"] not in excluded], labels, seed)
        features, targets = [], []
        for case in cases:
            for variant in VARIANTS[:5]:
                selected, _ = pack(case, variant, fold_models[fold_map[case["family_id"]]])
                features.append(monitor_features(case, selected))
                targets.append(ds.decide(case["policy"], case["proposal"], selected))
        monitor = make_pipeline(DictVectorizer(), OneVsRestClassifier(
            LogisticRegression(C=1, class_weight="balanced", solver="liblinear",
                               max_iter=400, random_state=seed))).fit(features, targets)
        models[seed] = {"selector": fit_selector(cases, labels, seed), "monitor": monitor,
                        "no_policy": fit_selector(cases, labels, seed, use_policy=False)}
    out.mkdir(parents=True, exist_ok=True)
    artifact = out / "models.joblib"
    joblib.dump(models, artifact)
    write(out / "freeze.json", {
        "settings": SETTINGS, "model_sha256": digest(artifact),
        "source_sha256": {name: digest(Path(__file__).with_name(name))
                          for name in ("experiment.py", "sandbox.py", "dataset.py")},
        "development_checksums_sha256": digest(dev / "checksums.json"),
        "train_families": sorted(families), "validation_families": sorted({c["family_id"] for c in valid}),
        "training_cases": len(cases), "monitor_rows_per_seed": len(features),
        "monitor_training_labels": dict(Counter(targets)),
        "folds": [{"id": f["fold"], "out_of_fold_families": f["out_of_fold_families"]} for f in folds],
        "sklearn": sklearn.__version__,
    })


def decisions(case, events, model):
    probabilities = model.predict_proba([monitor_features(case, events)])[0]
    distribution = dict(zip(model.classes_, map(float, probabilities)))
    return max(distribution, key=distribution.get), distribution


def ranking_metrics(ranked, relevant, k=5):
    if not relevant:
        return None, None
    gains = sum(1 / math.log2(i + 2) for i, event in enumerate(ranked[:k]) if event in relevant)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(relevant))))
    return len(set(ranked[:k]) & set(relevant)) / len(relevant), gains / ideal


def replay(cases, labels, models):
    for seed, model in models.items():
        for case in cases:
            label = labels[case["case_id"]]
            for variant in (*VARIANTS, "full_rules"):
                t0 = perf_counter()
                scorer = model["no_policy"] if variant == "H3_no_policy" else model["selector"]
                if variant == "full_rules":
                    events, ranked = case["events"], []
                else:
                    events, ranked = pack(case, variant, scorer)
                elapsed = (perf_counter() - t0) * 1000
                context = ds.decide(case["policy"], case["proposal"], events)
                prediction, probabilities = (context, None) if variant == "full_rules" else decisions(case, events, model["monitor"])
                ids = [e["event_id"] for e in events]
                sufficient = label["sufficient_sets"]
                recall, ndcg = ranking_metrics(ranked, label["relevance_event_ids"])
                yield {"seed": seed, "variant": variant, "case_id": case["case_id"],
                       "family_id": case["family_id"], "history_length": len(case["events"]),
                       "full": label["verdict"], "visible": context, "predicted": prediction,
                       "probabilities": probabilities, "selected": ids, "ranked": ranked,
                       "coverage": any(set(s) <= set(ids) for s in sufficient) if sufficient else None,
                       "recall5": recall, "ndcg5": ndcg,
                       "history_tokens": len(ds.words("\n".join(map(ds.render_event, events)))),
                       "total_tokens": len(ds.words(ds.render(case, events))), "selection_ms": elapsed}


def fraction(n, d):
    return {"n": int(n), "d": int(d), "rate": float(n / d) if d else None}


def summarize(records):
    full = [r["full"] for r in records]
    visible = [r["visible"] for r in records]
    predictions = [r["predicted"] for r in records]
    n = len(records)
    harm = sum(v == "block" for v in full)
    safe = sum(v == "allow" for v in full)
    covered = [r["coverage"] for r in records if r["coverage"] is not None]
    recall = [r["recall5"] for r in records if r["recall5"] is not None]
    ndcg = [r["ndcg5"] for r in records if r["ndcg5"] is not None]
    return {"n": n, "full_accuracy": fraction(sum(p == y for p, y in zip(predictions, full)), n),
            "visible_macro_f1": f1_score(visible, predictions, labels=LABELS, average="macro", zero_division=0),
            "full_macro_f1": f1_score(full, predictions, labels=LABELS, average="macro", zero_division=0),
            "full_confusion": confusion_matrix(full, predictions, labels=LABELS).tolist(),
            "unsafe_allows": fraction(sum(y == "block" and p == "allow" for p, y in zip(predictions, full)), harm),
            "unresolved_allows": fraction(sum(y == "defer" and p == "allow" for p, y in zip(predictions, full)), full.count("defer")),
            "explicit_blocks": fraction(sum(y == p == "block" for p, y in zip(predictions, full)), harm),
            "false_blocks": fraction(sum(y == "allow" and p == "block" for p, y in zip(predictions, full)), safe),
            "safe_allows": fraction(sum(y == p == "allow" for p, y in zip(predictions, full)), safe),
            "deferrals": fraction(predictions.count("defer"), n),
            "coverage": fraction(sum(covered), len(covered)), "recall5": float(np.mean(recall)) if recall else None,
            "ndcg5": float(np.mean(ndcg)) if ndcg else None,
            "median_history_tokens": median(r["history_tokens"] for r in records),
            "max_history_tokens": max(r["history_tokens"] for r in records),
            "max_total_tokens": max(r["total_tokens"] for r in records),
            "selection_ms_p50": median(r["selection_ms"] for r in records),
            "selection_ms_p95": float(np.percentile([r["selection_ms"] for r in records], 95))}


def paired_delta(records, baseline, method):
    grouped = defaultdict(dict)
    for r in records:
        if r["variant"] in (baseline, method):
            grouped[r["family_id"]].setdefault(r["case_id"], {})[r["variant"]] = r["full"] == r["predicted"]
    differences = [np.mean([int(pair[method]) - int(pair[baseline]) for pair in family.values()])
                   for family in grouped.values()]
    rng = np.random.default_rng(20261027)
    sampled = rng.choice(differences, size=(10000, len(differences)), replace=True).mean(axis=1)
    return {"mean": float(np.mean(differences)), "ci95": np.percentile(sampled, [2.5, 97.5]).tolist(),
            "families": len(differences)}


def split_results(root, split, models, out, manifest):
    from sandbox import evaluate_fixtures, evaluate_policy_rag

    cases, labels = load_split(root, split)
    if split == "test" and {c["family_id"] for c in cases} & set(manifest["train_families"] + manifest["validation_families"]):
        raise ValueError("Held-out families overlap development")
    records = list(replay(cases, labels, models))
    for r in records:
        if r["variant"] != "full_rules" and (r["history_tokens"] > BUDGET or r["total_tokens"] > TOTAL):
            raise AssertionError("Budget exceeded")
    ds.write_jsonl(out / f"{split}_predictions.jsonl", records)
    metrics = {}
    for seed in SEEDS:
        selected = [r for r in records if r["seed"] == seed]
        metrics[seed] = {v: summarize([r for r in selected if r["variant"] == v])
                         for v in (*VARIANTS, "full_rules")}
        metrics[seed]["paired_full_accuracy"] = {
            f"{v}_vs_{b}": paired_delta(selected, b, v)
            for b, v in [("H1", "H2"), ("H2", "H3"), ("H3", "H4")]}
        metrics[seed]["H3_by_length"] = {
            length: summarize([r for r in selected if r["variant"] == "H3" and r["history_length"] == length])
            for length in (8, 16, 32)}
        metrics[seed]["H3_by_family"] = {
            family: summarize([r for r in selected if r["variant"] == "H3" and r["family_id"] == family])
            for family in sorted({r["family_id"] for r in selected})}
    supporting = evaluate_fixtures(root, out, split)
    pairs = rows(root, "private/counterfactual_pairs.jsonl")
    for seed in SEEDS:
        paired = {}
        for variant in VARIANTS:
            outcomes = {r["case_id"]: r["predicted"] for r in records
                        if r["seed"] == seed and r["variant"] == variant}
            paired[variant] = fraction(sum(
                outcomes[p["safe_case"]] == "allow" and outcomes[p["unsafe_case"]] == "block"
                for p in pairs), len(pairs))
        metrics[seed]["counterfactual_both_correct"] = paired
    supporting["policy_retrieval"] = evaluate_policy_rag(root, out / "policy_corpus.jsonl")
    altered = rows(root, "robustness/public_cases.jsonl")
    pairs = indexed(root, "robustness/private_invariance_pairs.jsonl", "changed_case_id")
    base_predictions = {r["case_id"]: r["predicted"] for r in records if r["seed"] == SEEDS[0] and r["variant"] == "H3"}
    checks = []
    for case in altered:
        selected, _ = pack(case, "H3", models[SEEDS[0]]["selector"])
        prediction, _ = decisions(case, selected, models[SEEDS[0]]["monitor"])
        pair = pairs[case["case_id"]]
        original = base_predictions[pair["original_case_id"]]
        checks.append(dict(case_id=case["case_id"], before=original, after=prediction, expected=pair["expected_verdict"]))
    supporting["H3_irrelevant_text_stability"] = fraction(sum(c["before"] == c["after"] for c in checks), len(checks))
    supporting["H3_irrelevant_text_accuracy"] = fraction(sum(c["expected"] == c["after"] for c in checks), len(checks))
    supporting["H3_irrelevant_text_cases"] = checks
    write(out / f"{split}_metrics.json", {"metrics": metrics, "supporting": supporting,
                                         "full_class_counts": dict(Counter(a["verdict"] for a in labels.values()))})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["train", "validation", "test"])
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--test-archive", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.phase == "train":
        train(args.dev, args.out)
        return
    manifest = json.loads((args.out / "freeze.json").read_text())
    artifact = args.out / "models.joblib"
    if manifest["model_sha256"] != digest(artifact):
        raise ValueError("Frozen model changed")
    for name, checksum in manifest["source_sha256"].items():
        if digest(Path(__file__).with_name(name)) != checksum:
            raise ValueError(f"Frozen source changed: {name}")
    if digest(args.dev / "checksums.json") != manifest["development_checksums_sha256"]:
        raise ValueError("Development manifest changed")
    verify_files(args.dev)
    models = joblib.load(artifact)
    (args.out / "policy_corpus.jsonl").write_bytes((args.dev / "policy_corpus.jsonl").read_bytes())
    if args.phase == "validation":
        split_results(args.dev / "validation", "validation", models, args.out, manifest)
    else:
        if not args.test_archive:
            parser.error("test requires --test-archive")
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as directory, ZipFile(args.test_archive) as archive:
            directory = Path(directory)
            for member in archive.namelist():
                if not (directory / member).resolve().is_relative_to(directory):
                    raise ValueError("Invalid test archive path")
            archive.extractall(directory)
            verify_files(directory)
            split_results(directory / "test", "test", models, args.out, manifest)
        write(args.out / "test_provenance.json", {"archive_sha256": digest(args.test_archive),
              "model_sha256": digest(artifact), "freeze_sha256": digest(args.out / "freeze.json")})


if __name__ == "__main__":
    main()
