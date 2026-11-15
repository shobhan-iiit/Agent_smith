"""MiniLM training with explicit label dimensions and independently reloadable adapters."""

from contextlib import nullcontext
from dataclasses import dataclass
import json
import os
from pathlib import Path
from packaging.version import Version
import random
import shutil

import numpy as np
from sklearn.metrics import f1_score, log_loss
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL = "microsoft/MiniLM-L12-H384-uncased"
LIMIT = 512


@dataclass(frozen=True)
class Settings:
    base: str
    seed: int
    epochs: int
    batch: int
    adaptation: str = "lora"


class Encoder:
    def __init__(self, source):
        self.tokenizer = AutoTokenizer.from_pretrained(source, use_fast=True)

    def encode(self, texts, specials=True):
        return self.tokenizer(texts, truncation=False, add_special_tokens=specials,
                              verbose=False)["input_ids"]

    def collate(self, ids):
        if max(map(len, ids)) > LIMIT:
            raise ValueError("Input exceeds MiniLM's 512-token limit")
        return self.tokenizer.pad({"input_ids": ids}, padding=True,
                                  pad_to_multiple_of=8, return_tensors="pt")


def base_model(source, classes):
    # Building the classification head before attaching PEFT fixes the 3-vs-2 reload error.
    if Version(torch.__version__.split("+", 1)[0]) < Version("2.6"):
        raise RuntimeError("MiniLM ships PyTorch .bin weights. Use a Kaggle image with PyTorch 2.6 or later.")
    return AutoModelForSequenceClassification.from_pretrained(source, num_labels=classes)


class Predictor:
    def __init__(self, path, base_override=None):
        from peft import PeftModel
        from safetensors.torch import load_file

        path = Path(path)
        self.meta = json.loads((path / "checkpoint.json").read_text())
        self.encoder = Encoder(path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base = base_model(base_override or os.environ.get("AGENT_SMITH_MINILM_PATH") or self.meta["base"], len(self.meta["labels"]))
        if self.meta["adaptation"] == "lora":
            self.model = PeftModel.from_pretrained(base, path)
        else:
            base.classifier.load_state_dict(load_file(path / "head.safetensors"))
            self.model = base
        self.model.to(self.device).eval()

    def probabilities(self, texts, batch=64):
        ids = self.encoder.encode(texts)
        return infer(self.model, self.encoder, ids, self.device, batch)


def precision(device):
    return torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()


@torch.inference_mode()
def infer(model, encoder, ids, device, batch):
    if not ids:
        raise ValueError("Cannot score an empty input list")
    model.eval()
    result = []
    for start in range(0, len(ids), batch):
        inputs = encoder.collate(ids[start:start + batch]).to(device)
        with precision(device):
            logits = model(**inputs).logits
        result.append(logits.float().softmax(-1).cpu().numpy())
    return np.concatenate(result)


def rows_digest(records):
    import hashlib
    value = hashlib.sha256()
    for row in records:
        value.update(json.dumps((row["family_id"], row["text"], row["label"]),
                               ensure_ascii=False, separators=(",", ":")).encode())
        value.update(b"\n")
    return value.hexdigest()


def fit(train, validation, output, labels, settings, encoder):
    """Resume at completed-stage boundaries and choose the epoch by validation loss."""
    from peft import LoraConfig, TaskType, get_peft_model
    from safetensors.torch import save_file

    output = Path(output)
    stage_signature = {"base_revision": Path(settings.base).name, "labels": labels, "adaptation": settings.adaptation,
                       "seed": settings.seed, "epochs": settings.epochs, "batch": settings.batch,
                       "train_rows": len(train), "validation_rows": len(validation),
                       "train_sha256": rows_digest(train), "validation_sha256": rows_digest(validation)}
    if (output / "complete.json").is_file():
        recorded = json.loads((output / "complete.json").read_text())
        if recorded != stage_signature:
            raise ValueError(f"Checkpoint configuration changed at {output}")
        return
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    if set(r["family_id"] for r in train) & set(r["family_id"] for r in validation):
        raise ValueError("Model fitting and checkpoint validation overlap by family")

    random.seed(settings.seed)
    np.random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = base_model(settings.base, len(labels))
    if settings.adaptation == "lora":
        model = get_peft_model(model, LoraConfig(task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16,
                               target_modules=["query", "value"], lora_dropout=0.05,
                               modules_to_save=["classifier"]))
    else:
        model.requires_grad_(False)
        model.classifier.requires_grad_(True)
    model.to(device)
    train_ids = encoder.encode([r["text"] for r in train])
    val_ids = encoder.encode([r["text"] for r in validation])
    if max(map(len, train_ids + val_ids)) > LIMIT:
        raise ValueError("A training or validation sequence exceeds 512 tokens")
    val_targets = [r["label"] for r in validation]
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=2e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    generator = torch.Generator().manual_seed(settings.seed)

    def collate(examples):
        ids, targets = zip(*examples)
        return encoder.collate(list(ids)), torch.tensor(targets, dtype=torch.long)

    loader = DataLoader(list(zip(train_ids, (r["label"] for r in train))),
                        batch_size=settings.batch, shuffle=True, generator=generator, collate_fn=collate)
    best = float("inf")
    history = []
    reference = None
    for epoch in range(1, settings.epochs + 1):
        model.train()
        losses = []
        for inputs, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            with precision(device):
                loss = model(**inputs.to(device), labels=targets.to(device)).loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        probabilities = infer(model, encoder, val_ids, device, 64)
        loss = log_loss(val_targets, probabilities, labels=list(range(len(labels))))
        predictions = probabilities.argmax(1)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_loss": loss,
                        "validation_accuracy": float(np.mean(predictions == val_targets)),
                        "validation_macro_f1": f1_score(val_targets, predictions,
                            labels=list(range(len(labels))), average="macro", zero_division=0)})
        print(output.name, json.dumps(history[-1]), flush=True)
        if loss < best:
            best = loss
            if settings.adaptation == "lora":
                model.save_pretrained(output, safe_serialization=True)
            else:
                save_file({k: v.detach().cpu().contiguous() for k, v in model.classifier.state_dict().items()},
                          output / "head.safetensors")
            encoder.tokenizer.save_pretrained(output)
            (output / "checkpoint.json").write_text(json.dumps({**stage_signature, "base": settings.base, "best_epoch": epoch}, indent=2))
            reference = probabilities[: min(64, len(probabilities))]
        (output / "history.json").write_text(json.dumps(history, indent=2))
    del model, optimizer, trainable
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if reference is None:
        raise RuntimeError("No finite validation checkpoint was written")
    reloaded = Predictor(output)
    sample = infer(reloaded.model, reloaded.encoder, val_ids[:len(reference)], device, 64)
    difference = float(np.max(np.abs(sample - reference)))
    if difference > 0.005:
        raise RuntimeError(f"Checkpoint reload changed probabilities by {difference:.6f}")
    (output / "reload.json").write_text(json.dumps({"max_probability_difference": difference}))
    (output / "complete.json").write_text(json.dumps(stage_signature, indent=2))
    print(output.name, f"reload verified (max probability difference {difference:.6g})", flush=True)
