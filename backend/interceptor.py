"""
interceptor.py — Entropy-Based Streaming Interceptor

Replaces the old regex claim detector with real Shannon entropy computed
directly from Ollama's per-token logprobs.

Architecture:
  1. _stream_ollama_with_entropy() → yields (token_text, entropy_score) tuples
     using Ollama's /v1/chat/completions with logprobs=True, top_logprobs=5

  2. compute_shannon_entropy() → converts top-K logprobs into a single entropy
     value in bits (H = -Σ p_i log₂ p_i).

  3. stream_and_detect() → accumulates tokens into sentences. Any sentence
     containing ≥1 high-entropy token (H > ENTROPY_THRESHOLD) is treated as
     a claim and forwarded to NLI verification. Low-entropy sentences skip NLI.

Calibration:
  ENTROPY_THRESHOLD = 2.0 bits — derived from HaluEval dataset calibration
  (see research notebook f53b3131 in conversation history)
"""

import re
import json
import math
import httpx
import asyncio
import logging
import os
from typing import AsyncGenerator
from models import Claim

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# ENTROPY THRESHOLD
# Tokens above this are "uncertain" → flag the enclosing sentence as a claim
# 2.0 bits was calibrated on HaluEval; higher = stricter (fewer false positives)
# ─────────────────────────────────────────────────────────────────
ENTROPY_THRESHOLD: float = float(os.getenv("ENTROPY_THRESHOLD", "2.0"))

# Split accumulated text on sentence boundaries
SENTENCE_END = re.compile(r'(?<=[.!?])\s+')


# ─────────────────────────────────────────────────────────────────
# 1. SHANNON ENTROPY COMPUTATION
# ─────────────────────────────────────────────────────────────────
def compute_shannon_entropy(top_logprobs: list[dict]) -> float:
    """
    Convert Ollama's top-K log-probabilities into a single Shannon entropy
    value measured in bits.

    H = -Σ p_i · log₂(p_i)

    Since the top-K probs don't sum to 1 (the tail is truncated), we
    re-normalise them before computing entropy. This gives a meaningful
    "spread" measure: H = 0 if the model is 100% certain, H increases
    as the model becomes more uncertain.

    Args:
        top_logprobs: list of dicts with 'logprob' key, from Ollama response

    Returns:
        Entropy in bits (float). Returns 0.0 if list is empty.
    """
    if not top_logprobs:
        return 0.0

    # Convert log-probs → raw probabilities
    probs = [math.exp(d["logprob"]) for d in top_logprobs]
    total = sum(probs)
    if total == 0:
        return 0.0

    # Re-normalise to sum = 1 over the observed top-K
    normalised = [p / total for p in probs]

    # Shannon entropy in bits
    entropy = -sum(p * math.log2(p) for p in normalised if p > 0)
    return round(entropy, 4)


# ─────────────────────────────────────────────────────────────────
# 2. OLLAMA STREAMING WITH LOGPROBS
# Yields (token_text, entropy) tuples — one per token from the model
# ─────────────────────────────────────────────────────────────────
async def _stream_ollama_with_entropy(
    query: str,
) -> AsyncGenerator[tuple[str, float], None]:
    """
    Stream tokens from Ollama's /v1/chat/completions endpoint.
    Requests logprobs=True + top_logprobs=5 so we get entropy per token.
    Yields (token_text, entropy_bits) tuples.
    """
    API_URL = "http://localhost:11434/v1/chat/completions"

    payload = {
        "model": "llama3",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a financial analysis assistant. "
                    "Answer clearly and concisely based on grounded context."
                ),
            },
            {"role": "user", "content": query},
        ],
        "stream":       True,
        "temperature":  0.3,
        "max_tokens":   300,
        "logprobs":     True,
        "top_logprobs": 5,          # Enough for a good entropy estimate
        "options": {
            "num_ctx": 1024,        # Prevents OOM crash on llama3
        },
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream("POST", API_URL, json=payload) as response:

            if response.status_code != 200:
                body = await response.aread()
                raise httpx.HTTPStatusError(
                    f"Ollama returned {response.status_code}: "
                    f"{body.decode(errors='ignore')[:300]}",
                    request=response.request,
                    response=response,
                )

            # Ollama /v1/ streams SSE: "data: {json}\n\n"
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:].strip()
                if data == "[DONE]":
                    break

                try:
                    chunk  = json.loads(data)
                    choice = chunk["choices"][0]

                    token_text = choice.get("delta", {}).get("content", "")
                    if not token_text:
                        continue

                    # Extract per-token entropy from logprobs
                    logprobs_obj  = choice.get("logprobs") or {}
                    content_lp    = logprobs_obj.get("content") or []
                    top_lp        = content_lp[0].get("top_logprobs", []) if content_lp else []
                    entropy       = compute_shannon_entropy(top_lp)

                    yield (token_text, entropy)

                except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                    continue


# ─────────────────────────────────────────────────────────────────
# 3. MOCK STREAM (for USE_MOCK=true env var — dev/testing only)
# ─────────────────────────────────────────────────────────────────
async def _stream_mock(query: str) -> AsyncGenerator[tuple[str, float], None]:
    """Fake stream for offline testing. Entropy is randomly set to test pipeline."""
    import random
    mock_text = (
        "Apple reported revenue of $523 billion in fiscal year 2022. "
        "Microsoft Azure grew 45% in Q4 2023. "
        "Tesla delivered 2.1 million vehicles in 2023."
    )
    tokens = re.findall(r'\S+|\s+', mock_text)
    for tok in tokens:
        # Randomly assign high entropy to numeric tokens to test flagging
        is_number = bool(re.search(r'\d', tok))
        entropy   = random.uniform(1.8, 3.0) if is_number else random.uniform(0.0, 1.0)
        yield (tok, entropy)
        await asyncio.sleep(0.03)


# ─────────────────────────────────────────────────────────────────
# 4. CLAIM BUILDER — Entropy-based (replaces old regex detect_claims)
# ─────────────────────────────────────────────────────────────────
def build_entropy_claim(sentence: str, max_entropy: float) -> list[Claim]:
    """
    If the sentence contains any high-entropy token (max_entropy > threshold),
    treat the ENTIRE sentence as a single claim for NLI verification.

    This replaces the old regex-pattern approach:
      - Regex only caught numbers/dates/names → missed paraphrased facts
      - Entropy catches ANY uncertain token → model-uncertainty-driven detection
    """
    if max_entropy > ENTROPY_THRESHOLD:
        return [
            Claim(
                text=sentence,
                type="entropy",
                position=0,
                sentence=sentence,
            )
        ]
    return []


# ─────────────────────────────────────────────────────────────────
# 5. MAIN INTERFACE — stream_and_detect()
# Called by main.py's firewall_generator
# Yields tuples: ("TOKEN:<text>:<entropy>", []) for tokens
#          or    (sentence, [Claim]) for completed sentences
# ─────────────────────────────────────────────────────────────────
async def stream_and_detect(query: str) -> AsyncGenerator[tuple, None]:
    """
    Main streaming interface for the firewall.

    Token events:  ("TOKEN:<text>:<entropy>", [])
      - entropy encoded in the string so main.py can colour high-entropy tokens

    Sentence events: (sentence_text, [Claim(...)])
      - Claim list is non-empty only when max token entropy > ENTROPY_THRESHOLD
      - Empty claim list means sentence was confident → skip NLI, mark verified
    """
    use_mock    = os.getenv("USE_MOCK", "false").lower() == "true"
    stream_fn   = _stream_mock if use_mock else _stream_ollama_with_entropy

    text_buffer   = ""         # rolling sentence buffer
    entropy_buffer: list[float] = []  # per-token entropies for current sentence
    token_count   = 0

    try:
        async for (token, entropy) in stream_fn(query):
            token_count  += 1
            text_buffer  += token
            entropy_buffer.append(entropy)

            # ── Yield token event — entropy is embedded in the tag ──────
            # main.py unpacks: tag="TOKEN:<text>:<entropy>"
            yield (f"TOKEN:{token}:{entropy}", [])

            # ── Check for sentence boundary ──────────────────────────────
            parts = SENTENCE_END.split(text_buffer)
            if len(parts) > 1:
                for sentence in parts[:-1]:
                    sentence = sentence.strip()
                    if not sentence:
                        continue

                    # Compute max entropy seen in this sentence
                    # (we don't track exact per-word alignment, so we use
                    #  the rolling buffer and drain it proportionally)
                    n_tokens = max(1, len(re.findall(r'\S+', sentence)))
                    window   = entropy_buffer[:n_tokens]
                    max_h    = max(window) if window else 0.0

                    # Build claim only if uncertain enough
                    claims = build_entropy_claim(sentence, max_h)
                    logger.debug(
                        f"Sentence entropy max={max_h:.3f}b | "
                        f"claim={'YES' if claims else 'NO'} | '{sentence[:60]}'"
                    )
                    yield (sentence, claims)

                    # Drain processed tokens from buffer
                    entropy_buffer = entropy_buffer[n_tokens:]

                text_buffer = parts[-1]

        # ── Flush remaining buffer at end of stream ─────────────────────
        if text_buffer.strip():
            max_h  = max(entropy_buffer) if entropy_buffer else 0.0
            claims = build_entropy_claim(text_buffer.strip(), max_h)
            yield (text_buffer.strip(), claims)

    except httpx.HTTPStatusError as e:
        logger.error(f"Groq API error: {e}")
        yield (f"ERROR:{str(e)}", [])

    except httpx.ConnectError:
        logger.error("Cannot reach Ollama at localhost:11434")
        yield ("ERROR:Cannot connect to Ollama. Is it running?", [])

    except Exception as e:
        logger.error(f"Interceptor error: {e}", exc_info=True)
        yield (f"ERROR:{str(e)}", [])

    logger.info(f"Stream done | tokens: {token_count} | threshold: {ENTROPY_THRESHOLD}b")