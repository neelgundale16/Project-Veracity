"""
train.py — NLI Sentinel Fine-Tuning Script
============================================
Fine-tunes cross-encoder/nli-MiniLM2-L6-H768 on the merged
HaluEval + FEVER + TruthfulQA dataset prepared by prepare_data.py.

Auto-detects GPU. Saves the best checkpoint to:
  ./models/sentinel-finetuned/

After training, copy that folder into your main backend:
  backend/models/sentinel-finetuned/

Run:
  python train.py
"""

import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
import evaluate

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_MODEL   = "cross-encoder/nli-MiniLM2-L6-H768"
OUTPUT_DIR   = "./models/sentinel-finetuned"
DATA_DIR     = Path("./data")
MAX_LENGTH   = 256   # Keep short to stay fast at inference
NUM_EPOCHS   = 3
BATCH_SIZE   = 16
LR           = 2e-5
WARMUP_RATIO = 0.1
NUM_LABELS   = 3     # contradiction(0), entailment(1), neutral(2)

LABEL_MAP = {0: "contradiction", 1: "entailment", 2: "neutral"}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
class NLIDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_length: int = MAX_LENGTH):
        self.samples = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if row["label"] not in (0, 1, 2):
                    continue
                self.samples.append(row)

        self.tokenizer  = tokenizer
        self.max_length = max_length
        print(f"  Loaded {len(self.samples):,} samples from {path.name}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples[idx]
        enc = self.tokenizer(
            row["premise"],
            row["hypothesis"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(row["label"], dtype=torch.long),
        }


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────
accuracy_metric = evaluate.load("accuracy")
f1_metric       = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions    = np.argmax(logits, axis=-1)
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    f1       = f1_metric.compute(predictions=predictions, references=labels, average="macro")
    return {"accuracy": accuracy["accuracy"], "f1": f1["f1"]}


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  Project Veracity — Sentinel NLI Fine-Tuning")
    print("=" * 55)

    # Check data exists
    for split in ["train", "val", "test"]:
        path = DATA_DIR / f"{split}.jsonl"
        if not path.exists():
            print(f"\nERROR: {path} not found.")
            print("Run prepare_data.py first!\n")
            return

    # Load tokenizer + model
    print(f"\nLoading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model     = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=NUM_LABELS,
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    # Build datasets
    print("\nBuilding datasets...")
    train_ds = NLIDataset(DATA_DIR / "train.jsonl", tokenizer)
    val_ds   = NLIDataset(DATA_DIR / "val.jsonl",   tokenizer)

    # Training args
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir                  = OUTPUT_DIR,
        num_train_epochs            = NUM_EPOCHS,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size  = BATCH_SIZE * 2,
        learning_rate               = LR,
        warmup_ratio                = WARMUP_RATIO,
        weight_decay                = 0.01,
        evaluation_strategy         = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        metric_for_best_model       = "f1",
        greater_is_better           = True,
        logging_steps               = 100,
        report_to                   = "none",    # No wandb/tensorboard needed
        fp16                        = (device == "cuda"),   # Mixed precision on GPU
        dataloader_num_workers      = 0,         # Avoids Windows multiprocessing issues
    )

    trainer = Trainer(
        model           = model,
        args            = args,
        train_dataset   = train_ds,
        eval_dataset    = val_ds,
        compute_metrics = compute_metrics,
        callbacks       = [EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # Train!
    print(f"\nTraining for up to {NUM_EPOCHS} epochs...")
    print(f"  Train samples : {len(train_ds):,}")
    print(f"  Val samples   : {len(val_ds):,}")
    print(f"  Batch size    : {BATCH_SIZE}")
    print(f"  Learning rate : {LR}")
    print()

    trainer.train()

    # Save the best model
    print(f"\nSaving fine-tuned model → {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\n✅ Fine-tuning complete!")
    print(f"   Model saved to: {OUTPUT_DIR}")
    print("\nNext steps:")
    print("  1. Run evaluate.py to check accuracy")
    print(f"  2. Copy '{OUTPUT_DIR}' into your backend/models/ folder")
    print("  3. The backend will auto-load it on next startup\n")


if __name__ == "__main__":
    main()
