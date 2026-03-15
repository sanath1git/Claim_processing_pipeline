"""
Itemized Bill Extraction Agent.

Uses Gemini with structured output to extract billing line items
and totals from bill pages.
"""

from __future__ import annotations

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from app.config import settings
from app.models import PageBundle, BillExtraction


SYSTEM_PROMPT = """You are a medical document data extractor. You will be shown images of itemized hospital bill pages from a medical claim.

Extract the following:
- line_items: A list of line items, each with:
    - description: service or item name
    - quantity: number of units (null if not shown)
    - unit_price: price per unit (null if not shown)
    - amount: line total amount
- total_amount: The grand total / final amount on the bill
- currency: Currency code (default "INR" for Indian documents)

If a field is not visible, return null. For line_items, return an empty list if nothing can be extracted.
Return ONLY valid JSON matching the schema."""


def extract_bill(pages: list[PageBundle]) -> dict:
    """
    Extract billing data from the given page images.

    Multi-page bills are sent together in one call so the LLM can
    aggregate line items across pages.
    """
    llm = ChatGoogleGenerativeAI(
        model=settings.EXTRACTION_MODEL,
        google_api_key=settings.GOOGLE_API_KEY,
        temperature=0,
        max_output_tokens=4096,
    )

    structured_llm = llm.with_structured_output(BillExtraction)

    content: list[dict] = [
        {"type": "text", "text": "Extract ALL billing line items and totals from the following bill page(s). Include every line item visible:"}
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
        if p.has_native_text and p.text_content:
            content.append(
                {"type": "text", "text": f"[Native text from page {p.page_number}]: {p.text_content}"}
            )

    message = HumanMessage(content=content)

    result: BillExtraction = structured_llm.invoke(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            message,
        ]
    )

    return result.model_dump()
