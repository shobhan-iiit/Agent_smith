"""Quick Kaggle check of real tokenizer, model loading and three-class PEFT round-trip."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from huggingface_hub import snapshot_download
import torch

from kaggle_run import BASE_FILES, MODEL, PreparedCases, safe_extract
from experiment import load_split
from neural import Encoder, Settings, fit

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--archive", type=Path, default=Path("Agent_Smith_Dataset_v1.zip"))
args = parser.parse_args()
if not torch.cuda.is_available():
    parser.error("Enable a Kaggle GPU accelerator first.")
base = snapshot_download(MODEL, allow_patterns=BASE_FILES)
encoder = Encoder(base)
with TemporaryDirectory() as temp:
    root = Path(temp)
    safe_extract(args.archive, root)
    dev = root / "Agent_Smith_Dataset_v1" / "development"
    train, train_labels = load_split(dev / "train", "train")
    val, val_labels = load_split(dev / "validation", "validation")
    train_rows = PreparedCases(train[:18], encoder).contexts(train_labels, None, ("H0", "H1", "H2"))
    val_rows = PreparedCases(val[:18], encoder).contexts(val_labels, None, ("H0", "H1", "H2"))
    fit(train_rows, val_rows, root / "smoke_monitor", ["allow", "block", "defer"],
        Settings(base, 17, 1, 8), encoder)
print("GPU smoke passed. The three-class adapter reloaded with matching probabilities.")
