import json
import time
import uuid
import logging
import os
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from typing import AsyncGenerator, Optional

from models import ChatRequest, StreamToken, SessionStats
from vault import vault, load_demo_financial_data
from sentinel import sentinel
from rewriter import rewriter
from interceptor import stream_and_detect

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🛡️  Project Veracity — starting up...")

    vault.initialize()
    if vault.get_count() == 0:
        logger.info("Vault empty — loading demo financial data...")
        load_demo_financial_data(vault)
    logger.info(f"Vault ready | {vault.get_count()} documents loaded")

    sentinel.initialize()
    logger.info("Sentinel ready")

    logger.info("✅ All systems go. Hallucination firewall is active.")
    yield
    logger.info("Project Veracity shutting down.")


app = FastAPI(title="Project Veracity", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


def sse(event_type: str, data: dict) -> str:
    return f"data: {json.dumps({'event_type': event_type, 'data': data})}\n\n"


def build_augmented_query(query: str) -> str:
    """
    Pull relevant context from vault and inject it into the prompt
    before sending anything to the LLM.
    Context is truncated to avoid Ollama OOM crashes on large documents.
    """
    vault_result = vault.search(query)

    if not vault_result:
        return (
            "You are Project Veracity, a grounded financial and document-analysis assistant.\n"
            "Answer the user's question clearly.\n"
            "If you do not have enough grounded context, say so plainly.\n\n"
            f"User query: {query}"
        )

    # Truncate context to prevent Ollama runner OOM on large documents
    MAX_CONTEXT_CHARS = 800
    context_text = vault_result.matched_text
    if len(context_text) > MAX_CONTEXT_CHARS:
        context_text = context_text[:MAX_CONTEXT_CHARS] + "…"

    return (
        "You are Project Veracity, a grounded document-analysis assistant.\n"
        "Use the provided ground-truth vault context as the primary source of truth.\n"
        "Do not claim that no document was uploaded if context is provided below.\n"
        "Answer only from the grounded context when possible.\n"
        "If the answer is not in the context, say that the uploaded/grounded document "
        "does not contain enough information.\n\n"
        f"User query: {query}\n\n"
        "GROUND-TRUTH CONTEXT:\n"
        f"{context_text}\n\n"
        f"SOURCE DOCUMENT: {vault_result.source_document}"
    )


async def firewall_generator(query: str) -> AsyncGenerator[str, None]:
    stats = SessionStats()
    pipeline_start = time.perf_counter()

    try:
        augmented_query = build_augmented_query(query)
        logger.info(f"Augmented query built for: '{query[:80]}'")

        async for raw, claims in stream_and_detect(augmented_query):

            if raw.startswith("TOKEN:"):
                # Format: TOKEN:<text>:<entropy_bits>
                parts = raw[6:].rsplit(":", 1)
                token_text  = parts[0]
                try:
                    entropy = float(parts[1])
                except (IndexError, ValueError):
                    entropy = 0.0

                from interceptor import ENTROPY_THRESHOLD
                status = "high-entropy" if entropy > ENTROPY_THRESHOLD else "streaming"
                token_event = StreamToken(
                    id=str(uuid.uuid4())[:8],
                    text=token_text,
                    status=status
                )
                yield sse("token", token_event.model_dump())
                continue

            if raw.startswith("ERROR:"):
                yield sse("error", {"message": raw[6:].strip()})
                return

            if not claims:
                # No high-entropy tokens → sentence is confident → skip NLI
                stats.claims_skipped += 1
                continue

            # Entropy flagged this sentence → send to NLI verification
            stats.total_claims_detected += len(claims)

            for claim in claims:
                t0 = time.perf_counter()
                # Entropy already pre-screened; no regex filter needed

                vault_result = vault.search(claim.sentence)
                if vault_result is None:
                    stats.claims_verified += 1
                    continue

                nli_result = sentinel.classify(
                    claim_text=claim.sentence,
                    context_text=vault_result.matched_text
                )

                latency = (time.perf_counter() - t0) * 1000
                logger.info(
                    f"'{claim.text}' | {nli_result.label} "
                    f"({nli_result.confidence:.2f}) | {latency:.1f}ms"
                )

                n = stats.claims_verified + 1
                stats.avg_verification_latency_ms = (
                    (stats.avg_verification_latency_ms * (n - 1) + latency) / n
                )
                stats.claims_verified += 1

                if nli_result.is_hallucination:
                    stats.hallucinations_found += 1
                    corrected = rewriter.rewrite(claim.sentence, vault_result, nli_result)
                    payload = rewriter.build_correction_payload(
                        claim.sentence, corrected, vault_result.source_document
                    )
                    stats.corrections_made += 1
                    logger.info(f"CORRECTED | '{claim.text}' → {vault_result.source_document}")

                    yield sse("correction", {
                        "original_claim": claim.text,
                        "original_sentence": claim.sentence,
                        "corrected_sentence": corrected,
                        "source": vault_result.source_document,
                        "similarity_score": vault_result.similarity_score,
                        "nli_label": nli_result.label,
                        "nli_confidence": nli_result.confidence,
                        "diff_ratio": payload.get("diff_ratio", 0),
                    })

                yield sse("stats", stats.model_dump())

        stats.total_pipeline_latency_ms = (time.perf_counter() - pipeline_start) * 1000
        logger.info(
            f"Done | claims:{stats.total_claims_detected} "
            f"corrections:{stats.corrections_made} "
            f"total:{stats.total_pipeline_latency_ms:.0f}ms"
        )
        yield sse("done", stats.model_dump())

    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        yield sse("error", {"message": str(e)})


@app.post("/chat")
async def chat(request: ChatRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    logger.info(f"Query: '{request.query[:80]}'")
    return StreamingResponse(
        firewall_generator(request.query),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
        }
    )


@app.post("/vault/upload")
async def vault_upload(
    file: UploadFile = File(...),
    source_name: Optional[str] = Form(None)
):
    filename = source_name or file.filename or "uploaded_document"
    content_type = file.content_type or ""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file was received. Please attach a file before uploading.")

    raw_bytes = await file.read()

    if not raw_bytes:
        raise HTTPException(status_code=400, detail=f"File '{filename}' appears to be empty. Please upload a file with content.")

    chunks: list = []

    if "pdf" in content_type or filename.lower().endswith(".pdf"):
        try:
            import pypdf, io as _io
        except ImportError:
            raise HTTPException(
                status_code=500,
                detail="pypdf is not installed on the server. Run: pip install pypdf"
            )
        try:
            reader = pypdf.PdfReader(_io.BytesIO(raw_bytes))
            if len(reader.pages) == 0:
                raise HTTPException(status_code=400, detail=f"'{filename}' has no pages.")
            for page in reader.pages:
                text = page.extract_text()
                if text and text.strip():
                    words, chunk, chars = text.split(), [], 0
                    for w in words:
                        chunk.append(w)
                        chars += len(w) + 1
                        if chars >= 500:
                            chunks.append(" ".join(chunk))
                            chunk, chars = [], 0
                    if chunk:
                        chunks.append(" ".join(chunk))
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Could not read '{filename}'. It may be encrypted, corrupted, or a scanned image-only PDF. Error: {e}"
            )
    else:
        try:
            text = raw_bytes.decode("utf-8")
            words, chunk, chars = text.split(), [], 0
            for w in words:
                chunk.append(w)
                chars += len(w) + 1
                if chars >= 500:
                    chunks.append(" ".join(chunk))
                    chunk, chars = [], 0
            if chunk:
                chunks.append(" ".join(chunk))
        except UnicodeDecodeError:
            raise HTTPException(
                status_code=400,
                detail=f"'{filename}' could not be decoded. File must be a PDF or a plain UTF-8 text file."
            )

    if not chunks:
        raise HTTPException(
            status_code=400,
            detail=f"No text could be extracted from '{filename}'. It may be a scanned image PDF with no selectable text."
        )

    vault.add_documents_bulk(chunks, filename)
    logger.info(f"Uploaded '{filename}' | {len(chunks)} chunks → vault")

    return JSONResponse({
        "status":       "success",
        "filename":     filename,
        "chunks_added": len(chunks),
        "vault_total":  vault.get_count(),
        "message":      f"Successfully loaded {len(chunks)} facts from '{filename}' into the vault."
    })


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "llm_provider": "Groq",
        "llm_model": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        "vault_documents": vault.get_count(),
        "sentinel_loaded": sentinel._initialized,
        "mock_mode": os.getenv("USE_MOCK", "false"),
        "version": "1.0.0"
    }


@app.get("/vault/count")
async def vault_count():
    return {"count": vault.get_count()}


@app.post("/vault/add")
async def vault_add(payload: dict):
    text = payload.get("text", "").strip()
    source = payload.get("source", "manual_entry")
    if not text:
        raise HTTPException(status_code=400, detail="text field required")
    vault.add_document(text=text, source_name=source)
    return {"status": "added", "vault_total": vault.get_count()}


@app.delete("/vault/clear")
async def vault_clear():
    vault.clear_vault()
    load_demo_financial_data(vault)
    return {"status": "reset", "vault_total": vault.get_count()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)