"""
PDF preprocessing — converts an uploaded PDF into a list of PageBundles.

Uses PyMuPDF (fitz) to render each page to a high-res PNG and pdfplumber
to extract any native text layer.
"""

from __future__ import annotations

import base64
import io
import tempfile
import os

import fitz  # PyMuPDF
import pdfplumber
from PIL import Image

from app.config import settings
from app.models import PageBundle


def _render_page_image(page: fitz.Page, dpi: int = 200) -> str:
    """Render a single fitz.Page to a base64-encoded PNG string."""
    zoom = dpi / 72  # fitz default is 72 DPI
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)

    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _extract_native_text(pdf_path: str, page_index: int) -> tuple[str, bool]:
    """
    Extract native text from a specific page using pdfplumber.
    Returns (text, has_native_text).
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if page_index < len(pdf.pages):
                text = pdf.pages[page_index].extract_text() or ""
                text = text.strip()
                return text, len(text) > 20  # threshold for 'meaningful' text
    except Exception:
        pass
    return "", False


def process_pdf(file_bytes: bytes) -> list[PageBundle]:
    """
    Convert raw PDF bytes into a list of PageBundle objects.

    Each PageBundle contains:
      - A high-res base64 PNG of the page
      - Native text (if available)
      - A flag indicating whether native text was found
    """
    # Write to temp file so pdfplumber can open it (needs seekable file)
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
    try:
        with os.fdopen(tmp_fd, "wb") as f:
            f.write(file_bytes)

        bundles: list[PageBundle] = []
        doc = fitz.open(tmp_path)

        for idx in range(len(doc)):
            page = doc[idx]

            image_b64 = _render_page_image(page, dpi=settings.PDF_DPI)
            text, has_text = _extract_native_text(tmp_path, idx)

            bundles.append(
                PageBundle(
                    page_number=idx + 1,  # 1-indexed
                    image_base64=image_b64,
                    text_content=text,
                    has_native_text=has_text,
                )
            )

        doc.close()
        return bundles
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
