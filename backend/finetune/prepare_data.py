"""
prepare_data.py — Fine-Tuning Data Pipeline
============================================
Downloads HaluEval + FEVER + TruthfulQA from HuggingFace
and formats them into a unified NLI training dataset.

Label mapping (matches cross-encoder/nli-MiniLM2-L6-H768):
  0 = contradiction  (LLM said something WRONG)
  1 = entailment     (LLM said something CORRECT)
  2 = neutral        (Cannot verify)

Run from terminal:
  python prepare_data.py

Output files (in ./data/):
  train.jsonl, val.jsonl, test.jsonl
"""

import json
import random
import os
from pathlib import Path
from datasets import load_dataset

random.seed(42)
DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────
# LOADER 1 — HaluEval
# ─────────────────────────────────────────────
def load_halueval() -> list[dict]:
    print("\n[1/3] Loading HaluEval...")
    rows = []

    for subset in ["qa_samples", "dialogue_samples", "summarization_samples"]:
        try:
            ds = load_dataset("pminervini/HaluEval", subset, split="data")
            for ex in ds:
                # Each example has a correct answer and a hallucinated answer
                # We use the knowledge/context as the premise
                premise = ex.get("knowledge") or ex.get("document") or ex.get("dialogue_history") or ""
                if not premise:
                    continue

                correct     = ex.get("right_answer") or ex.get("hallucinated_response_correct") or ""
                hallucinated= ex.get("hallucinated_answer") or ex.get("hallucinated_response") or ""

                if correct:
                    rows.append({"premise": premise, "hypothesis": correct,      "label": 1})  # entailment
                if hallucinated:
                    rows.append({"premise": premise, "hypothesis": hallucinated,  "label": 0})  # contradiction
        except Exception as e:
            print(f"  Warning: Could not load HaluEval/{subset}: {e}")

    print(f"  HaluEval: {len(rows)} samples loaded")
    return rows


# ─────────────────────────────────────────────
# LOADER 2 — FEVER
# ─────────────────────────────────────────────
def load_fever() -> list[dict]:
    print("\n[2/3] Loading FEVER...")
    rows = []

    label_map = {
        "SUPPORTS":           1,  # entailment
        "REFUTES":            0,  # contradiction
        "NOT ENOUGH INFO":    2,  # neutral
    }

    try:
        # FEVER v1.0 via datasets
        ds = load_dataset("fever", "v1.0", split="train")
        for ex in ds:
            evidence = " ".join([e[2] for e in (ex.get("evidence") or []) if e[2]])
            claim    = ex.get("claim", "")
            label_str= ex.get("label", "")
            label    = label_map.get(label_str)

            if not evidence or not claim or label is None:
                continue

            rows.append({"premise": evidence, "hypothesis": claim, "label": label})

            # Balance: don't use more than 60K FEVER samples
            if len(rows) >= 60000:
                break

    except Exception as e:
        print(f"  Warning: Could not load FEVER: {e}")
        print("  Trying alternate FEVER source...")
        try:
            ds = load_dataset("copenlu/fever_gold_evidence", split="train")
            for ex in ds:
                evidence = ex.get("evidence_text", "")
                claim    = ex.get("claim", "")
                label_str= ex.get("label", "")
                label    = label_map.get(label_str)
                if not evidence or not claim or label is None:
                    continue
                rows.append({"premise": evidence, "hypothesis": claim, "label": label})
                if len(rows) >= 60000:
                    break
        except Exception as e2:
            print(f"  Warning: Alternate FEVER also failed: {e2}")

    print(f"  FEVER: {len(rows)} samples loaded")
    return rows


# ─────────────────────────────────────────────
# LOADER 3 — TruthfulQA
# ─────────────────────────────────────────────
def load_truthfulqa() -> list[dict]:
    print("\n[3/3] Loading TruthfulQA...")
    rows = []

    try:
        ds = load_dataset("truthful_qa", "generation", split="validation")
        for ex in ds:
            question    = ex.get("question", "")
            correct     = ex.get("best_answer", "")
            incorrect   = ex.get("incorrect_answers", [])

            # Use the question itself as the minimal "premise"
            # (TruthfulQA doesn't have separate evidence, so use question as context)
            if correct:
                rows.append({"premise": question, "hypothesis": correct, "label": 1})
            for wrong in (incorrect or [])[:2]:  # Take max 2 wrong answers per question
                rows.append({"premise": question, "hypothesis": wrong,   "label": 0})

    except Exception as e:
        print(f"  Warning: Could not load TruthfulQA: {e}")

    print(f"  TruthfulQA: {len(rows)} samples loaded")
    return rows


# ─────────────────────────────────────────────
# MERGE + SPLIT
# ─────────────────────────────────────────────
def save_splits(all_rows: list[dict]):
    random.shuffle(all_rows)
    n      = len(all_rows)
    n_train= int(n * 0.80)
    n_val  = int(n * 0.10)

    splits = {
        "train": all_rows[:n_train],
        "val":   all_rows[n_train:n_train + n_val],
        "test":  all_rows[n_train + n_val:],
    }

    for name, rows in splits.items():
        path = DATA_DIR / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        print(f"  Saved {len(rows):,} rows → {path}")

    # Label distribution
    from collections import Counter
    label_names = {0: "contradiction", 1: "entailment", 2: "neutral"}
    counts = Counter(r["label"] for r in all_rows)
    print("\n  Label distribution:")
    for lbl, cnt in sorted(counts.items()):
        pct = cnt / n * 100
        print(f"    {label_names[lbl]:15s}: {cnt:6,} ({pct:.1f}%)")


if __name__ == "__main__":
    print("=" * 55)
    print("  Project Veracity — NLI Fine-Tuning Data Pipeline")
    print("=" * 55)

    halueval    = load_halueval()
    fever       = load_fever()
    truthfulqa  = load_truthfulqa()

    all_rows = halueval + fever + truthfulqa
    print(f"\nTotal samples: {len(all_rows):,}")

    save_splits(all_rows)
    print("\nDone! Run train.py next.")
