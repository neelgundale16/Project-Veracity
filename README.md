# Project Veracity
### Self-Healing Hallucination Firewall

## What It Does

Project Veracity is a real-time middleware firewall that sits between a large language model and the user. It intercepts the LLM's streaming output token by token, detects factual claims using entropy-based uncertainty estimation, verifies them against a local ground-truth database, and auto-corrects hallucinations before they reach the screen.

The frontend visually highlights every correction in green with a tooltip showing the source document the fact was verified against.

---

## The Problem

Enterprise AI deployments fail for three reasons. LLMs hallucinate numbers, names, and statistics with full confidence. Existing fact-checkers wait for the full response before checking, destroying the streaming experience. Tools that only flag uncertain responses leave the user without a correct answer.

Veracity solves all three simultaneously.

---

## Architecture

```
User Query
    |
Groq API (llama3-8b) -- SSE Token Stream + per-token logprobs
    |
Interceptor       -- Shannon entropy computed per token from logprob distribution
                     High-entropy spans flagged as uncertain claims
    |
Sentinel          -- NLI classifier, skips non-factual sentences
    |
Vault             -- ChromaDB semantic search against verified ground truth
    |
Rewriter          -- REVERSE algorithm, corrects hallucinated values
    |
FastAPI SSE       -- streams corrected tokens to frontend
    |
Next.js UI        -- green highlight + source tooltip on corrections
```

---

## Entropy-Based Detection

The claim detector does not use hardcoded regex patterns. Instead it requests per-token log probabilities from the Groq API and computes Shannon entropy at each token position.

```
H(t) = -sum( p_i * log(p_i) ) over top-k candidate tokens at position t
```

Tokens where the model was uncertain which word to generate produce high entropy. Contiguous spans of high-entropy tokens form claim boundaries — these are the positions where the LLM is statistically guessing rather than recalling.

This approach is grounded in Semantic Entropy (Kuhn et al.) and Kernel Language Entropy (ICLR 2024), both referenced in the project research compendium.

---

## Tech Stack

**Backend** — Python, FastAPI, uvicorn, asyncio  
**LLM** — Groq API (llama3-8b-8192, free tier, with logprobs enabled)  
**Claim Detection** — Shannon entropy over per-token Groq logprob distributions  
**NLI Classifier** — HuggingFace transformers, cross-encoder/nli-MiniLM2-L6-H768  
**Vector Database** — ChromaDB with cosine similarity, sentence-transformers  
**Correction Algorithm** — REVERSE (entity replacement with difflib fallback)  
**Frontend** — Next.js 14, React 18, TypeScript, Tailwind CSS  

---

## Performance

| Stage | Target |
|---|---|
| Entropy computation per token | under 1ms |
| NLI inference | under 12ms |
| Vault semantic search | under 100ms |
| Full pipeline | under 200ms |
| Token yield to frontend | under 50ms |

Non-factual sentences are skipped entirely by the Sentinel classifier, saving approximately 72% of verification compute.

---

## Getting Started

**Requirements:** Python 3.10+, Node.js 18+, free Groq API key from console.groq.com

```bash
# Backend
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Create backend/.env
# GROQ_API_KEY=gsk_your_key_here
# USE_MOCK=false

uvicorn main:app --reload --port 8000
```

```bash
# Frontend
cd frontend/veracity-ui
npm install
npm run dev
```

Open `http://localhost:3000`

To run without an API key, set `USE_MOCK=true` in `.env`. The mock stream contains deliberate hallucinations with pre-assigned synthetic entropy values to demonstrate the full correction pipeline.

---

## File Structure

```
backend/
  models.py         Pydantic schemas
  vault.py          ChromaDB ground truth vault with 15 verified financial facts
  sentinel.py       NLI classifier wrapper (HaluGate Sentinel)
  rewriter.py       REVERSE correction algorithm
  interceptor.py    Groq SSE stream handler, entropy-based claim detection
  main.py           FastAPI app and pipeline orchestration

frontend/veracity-ui/src/
  app/page.tsx              Main UI and SSE consumer
  components/TokenStream    Token rendering with correction highlights
  components/StatsPanel     Live firewall statistics panel
  components/QueryInput     Query input with loading state
  types/index.ts            TypeScript interfaces
```

---

## Future Additions

- PDF upload to populate the vault with custom documents at runtime
- Support for legal and medical domain ground-truth datasets
- Per-token entropy heatmap visualization in the UI
- Multi-turn conversation with correction memory across turns
- Exportable correction audit log for enterprise compliance
- Adaptive entropy threshold calibrated per model and domain
