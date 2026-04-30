# models.py — All Pydantic Data Models for Project Veracity
# THIS FILE IS IMPORTED BY EVERY OTHER FILE — build it first
# Team: Leo | Hackathon: XEN-O-THON 2026

from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime


# ═══════════════════════════════════════════════════════
# MODEL 1 — ChatRequest
# What the user sends to the API
# ═══════════════════════════════════════════════════════
class ChatRequest(BaseModel):
    query: str                          # e.g. "What was Apple's revenue in 2022?"
    session_id: Optional[str] = None    # optional session tracking


# ═══════════════════════════════════════════════════════
# MODEL 2 — Claim
# A factual claim detected inside the LLM stream
# e.g. "$523 billion" or "Apple Inc" or "fiscal year 2022"
# ═══════════════════════════════════════════════════════
class Claim(BaseModel):
    text: str                           # the detected claim text
    type: Literal[                      # what kind of claim is it?
        "number",
        "name",
        "date",
        "policy",
        "statistic",
        "general",
        "entropy"                       # ← entropy-flagged uncertain sentence
    ]
    position: int                       # character position in the stream
    sentence: str                       # full sentence containing this claim


# ═══════════════════════════════════════════════════════
# MODEL 3 — VaultResult
# What ChromaDB returns after a semantic search
# ═══════════════════════════════════════════════════════
class VaultResult(BaseModel):
    matched_text: str                   # best matching fact from vault
    source_document: str                # e.g. "apple_annual_report.pdf"
    similarity_score: float             # 0 to 1 → higher = more similar
    distance: float                     # raw ChromaDB distance value


# ═══════════════════════════════════════════════════════
# MODEL 4 — NLIResult
# What the HaluGate Sentinel returns after NLI classification
# Entailment = fact is supported | Contradiction = hallucination!
# ═══════════════════════════════════════════════════════
class NLIResult(BaseModel):
    label: Literal[
        "entailment",
        "neutral",
        "contradiction"
    ]
    confidence: float                   # how confident is the model?
    entailment_score: float             # probability it's supported
    neutral_score: float                # probability it's neutral
    contradiction_score: float          # probability it's a hallucination
    is_hallucination: bool              # True if contradiction detected


# ═══════════════════════════════════════════════════════
# MODEL 5 — VerificationResult
# The full result of checking one claim through the pipeline
# claim → vault search → NLI check → correction (if needed)
# ═══════════════════════════════════════════════════════
class VerificationResult(BaseModel):
    claim: Claim                                    # the original claim
    vault_result: Optional[VaultResult] = None      # what vault found
    nli_result: Optional[NLIResult] = None          # what sentinel decided
    corrected_sentence: Optional[str] = None        # fixed sentence
    was_corrected: bool = False                     # did we fix anything?
    latency_ms: float = 0.0                         # how long it took (ms)


# ═══════════════════════════════════════════════════════
# MODEL 6 — StreamToken
# Each token that gets sent to the frontend via SSE
# status tells frontend: verified=green, corrected=highlight
# ═══════════════════════════════════════════════════════
class StreamToken(BaseModel):
    id: str                             # unique token ID e.g. "tok_42"
    text: str                           # the actual text content
    status: Literal[
        "streaming",                    # still coming in
        "verified",                     # checked, no hallucination
        "corrected",                    # was wrong, now fixed ✓
        "skipped",                      # non-factual, skipped check
        "high-entropy"                  # ← uncertain token flagged by entropy
    ] = "streaming"
    correction: Optional[str] = None    # the corrected text (if any)
    source: Optional[str] = None        # source PDF that proved the fix
    timestamp: float = Field(           # auto-set to current time
        default_factory=lambda: datetime.now().timestamp()
    )


# ═══════════════════════════════════════════════════════
# MODEL 7 — SessionStats
# Running statistics shown in the UI stats panel
# ═══════════════════════════════════════════════════════
class SessionStats(BaseModel):
    total_claims_detected: int = 0
    claims_verified: int = 0
    claims_skipped: int = 0
    hallucinations_found: int = 0
    corrections_made: int = 0
    avg_verification_latency_ms: float = 0.0
    total_pipeline_latency_ms: float = 0.0


# ═══════════════════════════════════════════════════════
# MODEL 8 — SSEEvent
# The wrapper for every Server-Sent Event to the frontend
# event_type tells frontend how to handle the data
# ═══════════════════════════════════════════════════════
class SSEEvent(BaseModel):
    event_type: Literal[
        "token",        # a new word/chunk arrived
        "correction",   # a hallucination was fixed
        "stats",        # updated statistics
        "done",         # stream finished
        "error"         # something went wrong
    ]
    data: dict          # the actual payload (flexible)
# ```

# ---

# ### 🔍 Key Things to Understand

# **Why `Literal[...]`?**
# It means only those exact values are allowed. If sentinel.py accidentally sends `label="wrong"` instead of `"entailment"`, Pydantic throws an error immediately. Catches bugs early.

# **Why `Optional[X] = None`?**
# Some fields only exist sometimes. `vault_result` is `None` if the vault found nothing. `corrected_sentence` is `None` if no correction was needed.

# **Why `Field(default_factory=...)`?**
# For `timestamp` — you can't write `default=datetime.now().timestamp()` because that runs once at import time. `default_factory` runs it fresh every time a new token is created.

# ---

# ### ✅ Self-Check Before Task 2
# ```
# □ File saved at backend/models.py
# □ 8 classes exist in the file
# □ No syntax errors (run: python models.py — should print nothing)