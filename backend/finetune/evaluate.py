"""
evaluate.py — Post Fine-Tuning Evaluation Script
==================================================
Compares the fine-tuned model vs the original pre-trained baseline
on the held-out test split.

Run AFTER train.py:
  python evaluate.py

Reports:
  - Accuracy, F1, Precision, Recall per class
  - Confusion matrix
  - Side-by-side comparison vs baseline
"""

import json
import torch
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from torch.nn.functional import softmax

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASELINE_MODEL   = "cross-encoder/nli-MiniLM2-L6-H768"
FINETUNED_MODEL  = "./models/sentinel-finetuned"
TEST_FILE        = Path("./data/test.jsonl")
BATCH_SIZE       = 32
MAX_LENGTH       = 256
LABEL_NAMES      = {0: "contradiction", 1: "entailment", 2: "neutral"}
device           = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────
# EVALUATION CORE
# ─────────────────────────────────────────────
def evaluate_model(model_name: str, test_rows: list[dict]) -> dict:
    print(f"\n  Loading: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()

    all_preds, all_labels = [], []

    for i in range(0, len(test_rows), BATCH_SIZE):
        batch = test_rows[i : i + BATCH_SIZE]
        premises    = [r["premise"]    for r in batch]
        hypotheses  = [r["hypothesis"] for r in batch]
        labels      = [r["label"]      for r in batch]

        enc = tokenizer(
            premises, hypotheses,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            logits = model(**enc).logits
            probs  = softmax(logits, dim=-1)
            preds  = torch.argmax(probs, dim=-1).cpu().tolist()

        all_preds.extend(preds)
        all_labels.extend(labels)

    return compute_report(all_preds, all_labels)


def compute_report(preds: list, labels: list) -> dict:
    n = len(labels)
    correct = sum(p == l for p, l in zip(preds, labels))
    accuracy = correct / n

    # Per-class TP, FP, FN
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for pred, label in zip(preds, labels):
        if pred == label:
            tp[label] += 1
        else:
            fp[pred]  += 1
            fn[label] += 1

    # Confusion matrix
    conf = [[0]*3 for _ in range(3)]
    for pred, label in zip(preds, labels):
        conf[label][pred] += 1

    # Per-class metrics
    per_class = {}
    for lbl in range(3):
        p = tp[lbl] / (tp[lbl] + fp[lbl] + 1e-9)
        r = tp[lbl] / (tp[lbl] + fn[lbl] + 1e-9)
        f = 2 * p * r / (p + r + 1e-9)
        per_class[LABEL_NAMES[lbl]] = {"precision": p, "recall": r, "f1": f}

    macro_f1 = np.mean([v["f1"] for v in per_class.values()])

    return {
        "accuracy":    accuracy,
        "macro_f1":    macro_f1,
        "per_class":   per_class,
        "confusion":   conf,
        "n":           n,
    }


def print_report(name: str, report: dict):
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Accuracy : {report['accuracy']*100:.2f}%")
    print(f"  Macro F1 : {report['macro_f1']:.4f}")
    print(f"\n  Per-class metrics:")
    print(f"  {'Class':<15} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'-'*47}")
    for cls, m in report["per_class"].items():
        print(f"  {cls:<15} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f}")

    print(f"\n  Confusion Matrix (rows=actual, cols=predicted):")
    print(f"  {'':15}", end="")
    for j in range(3):
        print(f" {LABEL_NAMES[j][:12]:>12}", end="")
    print()
    for i, row in enumerate(report["confusion"]):
        print(f"  {LABEL_NAMES[i]:<15}", end="")
        for val in row:
            print(f" {val:>12,}", end="")
        print()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  Project Veracity — Sentinel Model Evaluation")
    print("=" * 55)

    if not TEST_FILE.exists():
        print(f"\nERROR: {TEST_FILE} not found. Run prepare_data.py first!")
        return

    if not Path(FINETUNED_MODEL).exists():
        print(f"\nERROR: {FINETUNED_MODEL} not found. Run train.py first!")
        return

    print(f"\nLoading test set from {TEST_FILE}...")
    test_rows = [json.loads(l) for l in open(TEST_FILE, encoding="utf-8")]
    print(f"  {len(test_rows):,} test samples")

    print("\nEvaluating models...")
    baseline_report  = evaluate_model(BASELINE_MODEL,  test_rows)
    finetuned_report = evaluate_model(FINETUNED_MODEL, test_rows)

    print_report("Baseline (pre-trained)",  baseline_report)
    print_report("Fine-tuned (your model)", finetuned_report)

    # Delta summary
    delta_acc = (finetuned_report["accuracy"] - baseline_report["accuracy"]) * 100
    delta_f1  =  finetuned_report["macro_f1"] - baseline_report["macro_f1"]

    print(f"\n{'='*50}")
    print(f"  Improvement Summary")
    print(f"{'='*50}")
    print(f"  Accuracy : {delta_acc:+.2f}%")
    print(f"  Macro F1 : {delta_f1:+.4f}")

    if delta_f1 > 0:
        print("\n✅ Fine-tuned model is BETTER — safe to deploy!")
        print(f"   Copy './models/sentinel-finetuned' to 'backend/models/sentinel-finetuned'")
    else:
        print("\n⚠️  Fine-tuned model did not improve. Consider:")
        print("   - More training epochs (increase NUM_EPOCHS in train.py)")
        print("   - Lower learning rate (try 1e-5)")
        print("   - Adding more training data")


if __name__ == "__main__":
    main()
