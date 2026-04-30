import re
import time
import logging
import os
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from models import NLIResult

logger = logging.getLogger(__name__)

# ── Model selection ────────────────────────────────────────────────────────
# If a locally fine-tuned model exists, load it automatically.
# Otherwise fall back to the original HuggingFace pre-trained model.
# To deploy your fine-tuned model, just drop it into backend/models/sentinel-finetuned/
LOCAL_MODEL_PATH   = Path(__file__).parent / "models" / "sentinel-finetuned"
HF_MODEL_NAME      = "cross-encoder/nli-MiniLM2-L6-H768"
NLI_MODEL_NAME     = str(LOCAL_MODEL_PATH) if LOCAL_MODEL_PATH.exists() else HF_MODEL_NAME
CONTRADICTION_THRESHOLD = 0.50

# Patterns that indicate a fact-seeking sentence
FACTUAL_PATTERNS = re.compile(
    r'\$[\d,.]+\s*(?:billion|million|trillion|thousand)?'
    r'|\d+(?:\.\d+)?%'
    r'|\d+(?:\.\d+)?\s*(?:billion|million|trillion)'
    r'|\bQ[1-4]\s*\d{4}\b'
    r'|\bFY\s*\d{4}\b'
    r'|\bfiscal\s+year\s+\d{4}\b'
    r'|\b(?:revenue|income|earnings|profit|loss|growth|rate|GDP|CPI)\b'
    r'|\b(?:reported|generated|reached|grew|declined|increased|decreased)\b',
    re.IGNORECASE
)


class HaluGateSentinel:
    def __init__(self):
        self.model     = None
        self.tokenizer = None
        self._initialized = False

    def initialize(self):
        """Load NLI model. Called once at startup."""
        is_local = LOCAL_MODEL_PATH.exists()
        source   = "LOCAL fine-tuned" if is_local else "HuggingFace pre-trained"
        logger.info(f"Loading NLI model: {NLI_MODEL_NAME} [{source}]")
        start = time.perf_counter()

        self.tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
        self.model     = AutoModelForSequenceClassification.from_pretrained(
            NLI_MODEL_NAME
        )
        self.model.eval()
        self._initialized = True

        elapsed = (time.perf_counter() - start) * 1000
        logger.info(f"Sentinel ready | model={source} | load time: {elapsed:.0f}ms")
        return self

    def is_fact_seeking(self, sentence: str) -> bool:
        """
        Fast regex check — does this sentence contain a verifiable claim?
        Returns True  → proceed to NLI verification
        Returns False → skip (this is the 72.2% efficiency gain)
        Target: <5ms
        """
        return bool(FACTUAL_PATTERNS.search(sentence))

    def classify(self, claim_text: str, context_text: str) -> NLIResult:
        """
        NLI classification.
        premise   = context_text (ground truth from vault)
        hypothesis= claim_text   (what the LLM said)

        Model label order for cross-encoder/nli-MiniLM2-L6-H768:
          index 0 → contradiction
          index 1 → entailment
          index 2 → neutral
        Target: <12ms
        """
        if not self._initialized:
            raise RuntimeError("Call initialize() first.")

        inputs = self.tokenizer(
            context_text,
            claim_text,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        with torch.no_grad():
            logits = self.model(**inputs).logits

        probs = torch.softmax(logits, dim=-1)[0]

        contradiction_score = float(probs[0])
        entailment_score    = float(probs[1])
        neutral_score       = float(probs[2])

        scores = {
            "contradiction": contradiction_score,
            "entailment":    entailment_score,
            "neutral":       neutral_score,
        }
        label      = max(scores, key=scores.get)
        confidence = scores[label]

        is_hallucination = (
            contradiction_score > CONTRADICTION_THRESHOLD
        )

        return NLIResult(
            label              = label,
            confidence         = round(confidence, 4),
            entailment_score   = round(entailment_score, 4),
            neutral_score      = round(neutral_score, 4),
            contradiction_score= round(contradiction_score, 4),
            is_hallucination   = is_hallucination
        )


# Singleton
sentinel = HaluGateSentinel()