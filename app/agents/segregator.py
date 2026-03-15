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


# Document types the segregator can assign
DOC_TYPES = [
    "identity_document",
    "discharge_summary",
    "itemized_bill",
    "claim_forms",
    "other",
]

SYSTEM_PROMPT = """You are a medical claim document classifier. You will be shown page images from a claim PDF.

For EACH page, classify it into exactly one of these document types:
- identity_document  — ID cards, insurance cards, Aadhaar, PAN, passport
- discharge_summary  — hospital discharge reports, clinical summaries
- itemized_bill      — hospital bills, pharmacy bills, itemized invoices
- claim_forms        — insurance claim application forms
- other              — anything that does not fit the above categories

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


def classify_pages(pages: list[PageBundle]) -> dict[str, list[int]]:
    """
    Classify all pages and return a routing_map.

    Pages are batched for efficiency. The returned dict maps each doc type
    to a list of 1-indexed page numbers.
    """
    llm = ChatGoogleGenerativeAI(
        model=settings.SEGREGATOR_MODEL,
        google_api_key=settings.GOOGLE_API_KEY,
        temperature=0,
        max_output_tokens=1024,
    )

    routing_map: dict[str, list[int]] = {dt: [] for dt in DOC_TYPES}
    batch_size = settings.PAGE_BATCH_SIZE

    for i in range(0, len(pages), batch_size):
        batch = pages[i : i + batch_size]
        message = _build_batch_message(batch)

        response = llm.invoke(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                message,
            ]
        )

        # Parse the JSON response
        raw = response.content.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()

        try:
            classifications = json.loads(raw)
        except json.JSONDecodeError:
            # Fallback: mark all pages in this batch as "other"
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
    return {k: v for k, v in routing_map.items() if v}
