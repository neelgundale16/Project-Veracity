# vault.py — Ground Truth Vault (ChromaDB Vector Database)
# Team: Leo | Hackathon: XEN-O-THON 2026
# YOUR FILE — Data Engineer Role

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from models import VaultResult
from typing import Optional
import os
import io
from pypdf import PdfReader
import logging

# ─────────────────────────────────────────────
# CONSTANTS — never hardcode these anywhere else
# ─────────────────────────────────────────────
VAULT_PATH           = "./vault_db"
COLLECTION_NAME      = "ground_truth_vault"
EMBEDDING_MODEL      = "all-MiniLM-L6-v2"
N_RESULTS            = 3
# Tweak this in your .env file to control how strict vault matches must be
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.75"))
UPLOAD_DIR           = "./uploaded_docs"  # ← NEW: PDFs saved here

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════
# CLASS: GroundTruthVault
# The entire vector database wrapped in one clean class
# ═══════════════════════════════════════════════════════
class GroundTruthVault:

    def __init__(self):
        self.embedder     = None
        self.client       = None
        self.collection   = None
        self._initialized = False

        # NEW — create upload folder automatically on startup
        os.makedirs(UPLOAD_DIR, exist_ok=True)


    # ───────────────────────────────────────────────────
    # METHOD 1 — initialize()
    # Call this once at app startup
    # Loads the embedding model + connects to ChromaDB
    # ───────────────────────────────────────────────────
    def initialize(self):
        logger.info("[Vault] Starting initialization...")

        # Load the sentence embedding model
        # This converts text → 384 numbers (embeddings)
        logger.info("[Vault] Loading embedding model...")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("[Vault] Embedding model loaded ✓")

        # Connect to ChromaDB (creates vault_db/ folder automatically)
        self.client = chromadb.PersistentClient(path=VAULT_PATH)

        # Get existing collection OR create a fresh one
        # cosine = best distance metric for text similarity
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

        self._initialized = True

        count = self.collection.count()
        logger.info(f"[Vault] Initialized ✓ — {count} facts in vault")
        return count


    # ───────────────────────────────────────────────────
    # METHOD 2 — add_document()
    # Adds ONE verified fact into the vault
    # ───────────────────────────────────────────────────
    def add_document(
        self,
        text: str,
        source_name: str,
        chunk_id: Optional[str] = None
    ):
        # Auto-generate a unique ID if none provided
        doc_id = chunk_id or f"{source_name}_{hash(text) % 100000}"

        # Convert text to embedding (list of 384 numbers)
        embedding = self.embedder.encode(text).tolist()

        # Store in ChromaDB
        self.collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[{"source": source_name}]
        )

        logger.info(f"[Vault] Added → '{doc_id}' from {source_name}")
        return doc_id


    # ───────────────────────────────────────────────────
    # METHOD 3 — add_documents_bulk()
    # Adds MANY facts at once (faster than one by one)
    # ───────────────────────────────────────────────────
    def add_documents_bulk(self, texts: list[str], source_name: str):
        # Generate IDs for each chunk
        ids = [f"{source_name}_chunk_{i}" for i in range(len(texts))]

        # Convert ALL texts to embeddings at once (faster)
        embeddings = self.embedder.encode(texts).tolist()

        # Same source for all
        metadatas = [{"source": source_name} for _ in texts]

        # Bulk insert into ChromaDB
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )

        logger.info(
            f"[Vault] Bulk added {len(texts)} facts "
            f"from {source_name} ✓"
        )


    # ───────────────────────────────────────────────────
    # METHOD 4 — search() ← MOST IMPORTANT METHOD
    # Called mid-stream for every detected claim
    # Returns the best matching fact OR None if not confident
    # Target: must complete in < 100ms
    # ───────────────────────────────────────────────────
    def search(self, claim_text: str) -> Optional[VaultResult]:
        # Safety check
        if not self._initialized:
            logger.error("[Vault] Not initialized! Call initialize() first.")
            return None

        # Can't search empty vault
        if self.collection.count() == 0:
            logger.warning("[Vault] Vault is empty — no facts to search")
            return None

        # Convert claim to embedding
        claim_embedding = self.embedder.encode(claim_text).tolist()

        # Search ChromaDB for closest matching facts
        results = self.collection.query(
            query_embeddings=[claim_embedding],
            n_results=min(N_RESULTS, self.collection.count())
        )

        # Pull out the best (first) result
        best_text     = results["documents"][0][0]
        best_source   = results["metadatas"][0][0]["source"]
        best_distance = results["distances"][0][0]

        # ChromaDB cosine distance: 0 = identical, 2 = opposite
        # Convert to similarity: 1.0 = perfect, 0.0 = no match
        similarity = 1.0 - (best_distance / 2.0)

        # If similarity too low → not confident → return None
        if similarity < SIMILARITY_THRESHOLD:
            logger.info(
                f"[Vault] No confident match found "
                f"(similarity={similarity:.3f} < {SIMILARITY_THRESHOLD})"
            )
            return None

        logger.info(
            f"[Vault] Match found! similarity={similarity:.3f} "
            f"source={best_source}"
        )

        return VaultResult(
            matched_text=best_text,
            source_document=best_source,
            similarity_score=round(similarity, 4),
            distance=round(best_distance, 4)
        )


    # ───────────────────────────────────────────────────
    # METHOD 5 — get_count()
    # Returns how many facts are stored in vault
    # ───────────────────────────────────────────────────
    def get_count(self) -> int:
        return self.collection.count()


    # ───────────────────────────────────────────────────
    # METHOD 6 — clear_vault()
    # Wipes all facts and starts fresh
    # ───────────────────────────────────────────────────
    def clear_vault(self):
        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("[Vault] Vault cleared ✓ — fresh start")


    # ───────────────────────────────────────────────────
    # NEW METHOD 7 — save_pdf_file()
    # Saves the uploaded PDF physically to disk
    # Returns the saved file path
    # ───────────────────────────────────────────────────
    def save_pdf_file(self, pdf_bytes: bytes, filename: str) -> str:
        save_path = os.path.join(UPLOAD_DIR, filename)

        with open(save_path, "wb") as f:
            f.write(pdf_bytes)

        logger.info(f"[Vault] PDF saved to disk → {save_path} ✓")
        return save_path


    # ───────────────────────────────────────────────────
    # NEW METHOD 8 — extract_chunks_from_pdf()
    # Reads PDF bytes → returns clean text chunks
    # Each chunk = one meaningful paragraph
    # ───────────────────────────────────────────────────
    def extract_chunks_from_pdf(
        self,
        pdf_bytes: bytes,
        filename: str
    ) -> list[str]:
        # Load PDF from memory (no saving to disk needed here)
        reader = PdfReader(io.BytesIO(pdf_bytes))
        chunks = []

        for page_num, page in enumerate(reader.pages):
            # Extract raw text from this page
            text = page.extract_text()

            if not text:
                continue  # skip empty pages

            # Split page into paragraphs
            paragraphs = text.split('\n\n')

            for para in paragraphs:
                # Clean up whitespace and newlines
                clean = para.strip().replace('\n', ' ')

                # Only keep meaningful chunks (not too short)
                if len(clean) > 50:
                    chunks.append(clean)

        logger.info(
            f"[Vault] Extracted {len(chunks)} chunks "
            f"from {filename} ({len(reader.pages)} pages) ✓"
        )
        return chunks


    # ───────────────────────────────────────────────────
    # NEW METHOD 9 — ingest_pdf() ← MAIN UPLOAD METHOD
    # Called by upload.py when frontend uploads a PDF
    # Does everything in one call:
    # 1. Saves PDF to disk
    # 2. Extracts text chunks
    # 3. Stores all chunks in ChromaDB
    # Returns stats dict for frontend to display
    # ───────────────────────────────────────────────────
    def ingest_pdf(self, pdf_bytes: bytes, filename: str) -> dict:
        logger.info(f"[Vault] Starting PDF ingestion: {filename}")

        # STEP 1 — Save PDF file physically to disk
        saved_path = self.save_pdf_file(pdf_bytes, filename)

        # STEP 2 — Extract text chunks from PDF
        chunks = self.extract_chunks_from_pdf(pdf_bytes, filename)

        # If no text found → return failure
        if not chunks:
            logger.warning(f"[Vault] No text extracted from {filename}")
            return {
                "success": False,
                "filename": filename,
                "chunks_added": 0,
                "total_vault_size": self.get_count(),
                "message": f"Could not extract text from {filename}"
            }

        # STEP 3 — Store all chunks in ChromaDB vault
        self.add_documents_bulk(
            texts=chunks,
            source_name=filename
        )

        logger.info(
            f"[Vault] Ingestion complete! "
            f"{len(chunks)} chunks stored from {filename} ✓"
        )

        # Return success stats to frontend
        return {
            "success": True,
            "filename": filename,
            "saved_path": saved_path,
            "chunks_added": len(chunks),
            "total_vault_size": self.get_count(),
            "message": f"Successfully loaded {len(chunks)} facts from {filename}"
        }


    # ───────────────────────────────────────────────────
    # NEW METHOD 10 — list_uploaded_pdfs()
    # Returns list of all PDFs saved in uploaded_docs/
    # Frontend uses this for "Loaded Documents" panel
    # ───────────────────────────────────────────────────
    def list_uploaded_pdfs(self) -> list[dict]:
        pdfs = []

        if not os.path.exists(UPLOAD_DIR):
            return pdfs

        for filename in os.listdir(UPLOAD_DIR):
            if filename.endswith(".pdf"):
                filepath = os.path.join(UPLOAD_DIR, filename)
                size_kb  = os.path.getsize(filepath) / 1024
                pdfs.append({
                    "filename": filename,
                    "size_kb": round(size_kb, 2),
                    "path": filepath
                })

        logger.info(f"[Vault] {len(pdfs)} PDFs found in upload folder")
        return pdfs


# ═══════════════════════════════════════════════════════
# FUNCTION: load_demo_financial_data()
# Seeds the vault with 10 real financial facts
# Called once at startup for the hackathon demo
# ═══════════════════════════════════════════════════════
def load_demo_financial_data(vault: GroundTruthVault):
    logger.info("[Vault] Loading demo financial facts...")

    facts = [
        (
            "Apple Inc reported total revenue of $394.3 billion "
            "in fiscal year 2022.",
            "apple_annual_report_2022.pdf"
        ),
        (
            "Microsoft Azure cloud revenue grew by 28% "
            "in Q4 fiscal year 2023.",
            "microsoft_q4_2023_earnings.pdf"
        ),
        (
            "Tesla delivered 1.81 million vehicles globally "
            "in the full year 2023.",
            "tesla_2023_delivery_report.pdf"
        ),
        (
            "The US Federal Reserve held interest rates at "
            "5.25% to 5.50% in December 2023.",
            "fed_reserve_december_2023_statement.pdf"
        ),
        (
            "Amazon Web Services generated $90.8 billion "
            "in revenue for full year 2023.",
            "amazon_aws_annual_report_2023.pdf"
        ),
        (
            "Nvidia reported total revenue of $60.9 billion "
            "for fiscal year 2024.",
            "nvidia_fy2024_annual_report.pdf"
        ),
        (
            "JPMorgan Chase reported net income of $49.6 billion "
            "for the full year 2023.",
            "jpmorgan_2023_annual_report.pdf"
        ),
        (
            "The US Consumer Price Index rose 3.4% "
            "year-over-year in December 2023.",
            "us_bureau_labor_statistics_dec_2023.pdf"
        ),
        (
            "The S&P 500 index closed at 4769.83 "
            "on December 29, 2023.",
            "sp500_market_close_dec2023.pdf"
        ),
        (
            "Goldman Sachs reported net earnings of $8.5 billion "
            "for the full year 2023.",
            "goldman_sachs_2023_annual_report.pdf"
        ),
    ]

    for text, source in facts:
        vault.add_document(text=text, source_name=source)

    logger.info(f"[Vault] {len(facts)} demo facts loaded ✓")


# ═══════════════════════════════════════════════════════
# SINGLETON — one vault instance shared across the app
# Every other file does: from vault import vault
# ═══════════════════════════════════════════════════════
vault = GroundTruthVault()