"""
Segregator Agent — classifies each PDF page by document type.

Uses Gemini with vision. Pages are batched (3-4 per call) to reduce
API latency. Returns a routing_map: dict[str, list[int]].
"""

from __future__ import annotations

import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from app.config import settings
from app.models import PageBundle


# Document types the segregator can assign — expanded to match assignment
DOC_TYPES = [
    "claim_forms",
    "cheque_or_bank_details",
    "identity_document",
    "itemized_bill",
    "discharge_summary",
    "prescription",
    "investigation_report",
    "cash_receipt",
    "other",
]

SYSTEM_PROMPT = """You are a medical claim document classifier. You will be shown page images from a claim PDF.

For EACH page, classify it into exactly one of these document types:
- claim_forms           — insurance claim application forms
- cheque_or_bank_details— bank details, cheque images, or payment advice
- identity_document     — ID cards, insurance cards, Aadhaar, PAN, passport
- itemized_bill         — hospital bills, pharmacy bills, itemized invoices
- discharge_summary     — hospital discharge reports, clinical summaries
- prescription          — prescriptions, medication orders, prescription slips
- investigation_report  — lab reports, imaging reports, investigation results
- cash_receipt          — standalone cash receipts or payment vouchers
- other                 — anything that does not fit the above categories

Respond with ONLY valid JSON. The keys must be "page_<number>" and values must be one of the document types listed above.

Example response:
{"page_1": "identity_document", "page_2": "discharge_summary", "page_3": "itemized_bill"}
"""


def _build_batch_message(pages: list[PageBundle]) -> HumanMessage:
    """Build a single HumanMessage with multiple page images for one LLM call."""
    content: list[dict] = [
        {"type": "text", "text": f"Classify the following {len(pages)} page(s):"}
    ]
    for p in pages:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{p.image_base64}",
                },
            }
        )
        content.append(
            {"type": "text", "text": f"(Above image is page {p.page_number})"}
        )
    return HumanMessage(content=content)


def classify_pages(pages: list[PageBundle]) -> tuple[dict[str, list[int]], list[str]]:
    """
    Classify all pages and return a routing_map.

    Pages are batched for efficiency. Returns (routing_map, warnings), where
    routing_map maps each doc type to a list of 1-indexed page numbers.
    """
    llm = ChatGoogleGenerativeAI(
        model=settings.SEGREGATOR_MODEL,
        google_api_key=settings.GOOGLE_API_KEY,
        temperature=0,
        max_output_tokens=1024,
    )

    routing_map: dict[str, list[int]] = {dt: [] for dt in DOC_TYPES}
    warnings: list[str] = []
    batch_size = settings.PAGE_BATCH_SIZE

    for i in range(0, len(pages), batch_size):
        batch = pages[i : i + batch_size]
        message = _build_batch_message(batch)

        classifications = None
        raw = ""
        for _ in range(2):
            response = llm.invoke(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    message,
                ]
            )

            # Parse the JSON response
            raw = response.content.strip() if isinstance(response.content, str) else str(response.content)
            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
                if raw.endswith("```"):
                    raw = raw[:-3]
                raw = raw.strip()

            try:
                classifications = json.loads(raw)
                break
            except json.JSONDecodeError:
                classifications = None

        if classifications is None:
            page_nums = [p.page_number for p in batch]
            warnings.append(
                f"Segregator JSON parse failed for pages {page_nums}; routed to 'other' for manual review."
            )
            for p in batch:
                routing_map["other"].append(p.page_number)
            continue

        for p in batch:
            key = f"page_{p.page_number}"
            doc_type = classifications.get(key, "other")
            if doc_type not in DOC_TYPES:
                doc_type = "other"
            routing_map[doc_type].append(p.page_number)

    # Remove empty keys
    return {k: v for k, v in routing_map.items() if v}, warnings
