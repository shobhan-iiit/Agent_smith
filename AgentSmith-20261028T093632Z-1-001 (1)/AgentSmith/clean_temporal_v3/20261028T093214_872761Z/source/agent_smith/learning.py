"""Optional MiniLM/LoRA training and inference.

Install the `training` extra first. This module deliberately imports heavy
libraries only when needed so CPU simulator work has no torch dependency.
Training accepts explicit labeled JSONL; it never invents labels or opens test data.
"""
import argparse
import json
from pathlib import Path

MODEL = "microsoft/MiniLM-L12-H384-uncased"


def fit(train_path, validation_path, output, task="selector", seed=17, epochs=3):
    """Fine-tune adapters plus a head; choose best checkpoint on validation loss.

    Records: {"text": "...", "label": 0, "split": "train", "family_id": "..."}.
    Caller must build family-disjoint data and cross-fitted monitor contexts.
    """
    import random
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from peft import LoraConfig, TaskType, get_peft_model

    if not 1 <= epochs <= 3:
        raise ValueError("Pilot training is capped at three epochs")
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    rows = {}
    for split, path in [("train", train_path), ("validation", validation_path)]:
        rows[split] = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        if not rows[split] or any(r["split"] != split for r in rows[split]):
            raise ValueError(f"Expected nonempty {split} data only")
    if {r["family_id"] for r in rows["train"]} & {r["family_id"] for r in rows["validation"]}:
        raise ValueError("Family leakage between train and validation")
    classes = 2 if task == "selector" else 3
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    for record in rows["train"] + rows["validation"]:
        if not isinstance(record["label"], int) or not 0 <= record["label"] < classes:
            raise ValueError("Invalid label")
        if len(tokenizer.encode(record["text"])) > 512:
            raise ValueError("Oversize input; repair serialization rather than silently truncate")
    base = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=classes)
    config = LoraConfig(task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16,
                        target_modules=["query", "value"], lora_dropout=.05,
                        modules_to_save=["classifier"])
    model = get_peft_model(base, config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    class Records(Dataset):
        def __init__(self, records):
            self.records = records

        def __len__(self):
            return len(self.records)

        def __getitem__(self, index):
            return self.records[index]

    def collate(batch):
        tokens = tokenizer([r["text"] for r in batch], padding=True, truncation=False,
                           return_tensors="pt")
        tokens["labels"] = torch.tensor([r["label"] for r in batch])
        return {k: v.to(device) for k, v in tokens.items()}

    loaders = {split: DataLoader(Records(records), batch_size=16,
                                shuffle=split == "train", collate_fn=collate)
               for split, records in rows.items()}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    history = []
    for epoch in range(epochs):
        model.train()
        for batch in loaders["train"]:
            optimizer.zero_grad()
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        total, count = 0., 0
        with torch.no_grad():
            for batch in loaders["validation"]:
                n = len(batch["labels"])
                total += model(**batch).loss.item() * n
                count += n
        validation_loss = total / count
        history.append({"epoch": epoch + 1, "validation_loss": validation_loss})
        if validation_loss < best:
            best = validation_loss
            model.save_pretrained(destination)
            tokenizer.save_pretrained(destination)
        (destination / "training.json").write_text(json.dumps({
            "task": task, "seed": seed, "base_model": MODEL,
            "history": history, "device": str(device),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)
        }, indent=2))
        print(json.dumps(history[-1]), flush=True)


class Predictor:
    """Reload saved adapter AND classification head for batched inference."""
    def __init__(self, checkpoint):
        import torch
        from transformers import AutoTokenizer
        from peft import AutoPeftModelForSequenceClassification
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        metadata = json.loads((Path(checkpoint) / "training.json").read_text())
        classes = 3 if metadata["task"] == "monitor" else 2
        self.model = AutoPeftModelForSequenceClassification.from_pretrained(
            checkpoint, num_labels=classes
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()

    def probabilities(self, texts):
        result = []
        for start in range(0, len(texts), 16):
            batch = texts[start:start + 16]
            if any(len(self.tokenizer.encode(t)) > 512 for t in batch):
                raise ValueError("Oversize inference input")
            encoded = self.tokenizer(batch, padding=True, truncation=False, return_tensors="pt")
            with self.torch.no_grad():
                logits = self.model(**{k: v.to(self.device) for k, v in encoded.items()}).logits
            result.extend(logits.softmax(-1).cpu().tolist())
        return result

    def __call__(self, query, documents):
        # Identical separator must be used when constructing selector training text.
        return [p[1] for p in self.probabilities([query + "\nEVENT\n" + d for d in documents])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task", choices=["selector", "monitor"], default="selector")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    fit(args.train, args.validation, args.output, args.task, args.seed, args.epochs)


if __name__ == "__main__":
    main()
