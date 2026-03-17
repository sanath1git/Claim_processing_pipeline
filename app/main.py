"""
FastAPI application — HTTP gateway for the claim processing pipeline.

Endpoints:
  POST /api/process — upload a PDF claim and get structured JSON back
  GET  /health      — basic health check
"""
from __future__ import annotations

import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import logging

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from app.config import settings
from app.pdf_processor import process_pdf
from app.graph import claim_pipeline

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Claim Processing Pipeline",
    description="Multi-agent AI pipeline for processing medical insurance claim PDFs. "
                "Classifies pages, extracts structured data, and returns consolidated JSON.",
    version="1.0.0",
)


@app.get("/health")
async def health():
    """Basic health check."""
    return {"status": "ok"}


@app.post("/api/process")
async def process_claim(
    claim_id: str = Form(..., description="Unique claim identifier, e.g. CLM-001"),
    file: UploadFile = File(..., description="PDF file to process"),
):
    """
    Upload a claim PDF and receive structured extraction results.

    The pipeline:
    1. Renders each page to a high-res image
    2. Classifies pages by document type (identity, discharge, bill, etc.)
    3. Routes pages to specialised extraction agents
    4. Returns merged JSON with all extracted data
    """

    # --- Validate API key ---------------------------------------------------
    if not settings.GOOGLE_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="GOOGLE_API_KEY is not configured. Set it in .env",
        )

    # --- Validate file -------------------------------------------------------
    if file.content_type not in ("application/pdf",):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file.content_type}'. Only PDF files are accepted.",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Validate PDF magic bytes to avoid relying only on content-type headers.
    if not file_bytes.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid PDF.")

    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > settings.MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({size_mb:.1f} MB). Maximum is {settings.MAX_FILE_SIZE_MB} MB.",
        )

    logger.info("Processing claim %s — file: %s (%.2f MB)", claim_id, file.filename, size_mb)

    # --- PDF → PageBundles ---------------------------------------------------
    try:
        pages = process_pdf(file_bytes)
    except Exception as e:
        logger.error("PDF processing failed: %s", e, exc_info=True)
        raise HTTPException(status_code=422, detail=f"Failed to process PDF: {e}")

    logger.info("Extracted %d pages", len(pages))

    # --- Run LangGraph pipeline ----------------------------------------------
    initial_state = {
        "claim_id": claim_id,
        "pages": [p.model_dump() for p in pages],
        "routing_map": {},
        "id_data": {},
        "discharge_data": {},
        "bill_data": {},
        "final_output": {},
        "confidence_flags": [],
    }

    try:
        result = claim_pipeline.invoke(initial_state)
    except Exception as e:
        logger.error("Pipeline failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")

    final = result.get("final_output", {})

    logger.info("Claim %s processed successfully", claim_id)

    return JSONResponse(content=final)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)