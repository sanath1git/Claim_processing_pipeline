"""
Discharge Summary Extraction Agent.

Uses Gemini with structured output to extract clinical summary fields
from discharge summary pages.
"""

from __future__ import annotations

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from app.config import settings
from app.models import PageBundle, DischargeExtraction


SYSTEM_PROMPT = """You are a medical document data extractor. You will be shown images of hospital discharge summary pages from a medical claim.

Extract the following fields:
- diagnosis: A list of diagnosed conditions / ICD descriptions
- admit_date: Hospital admission date in YYYY-MM-DD format
- discharge_date: Hospital discharge date in YYYY-MM-DD format
- physician_name: Name of the attending or discharging physician
- hospital_name: Name of the hospital

If a field is not visible or cannot be determined, return null (or an empty list for diagnosis).
Return ONLY valid JSON matching the schema."""


def extract_discharge(pages: list[PageBundle]) -> dict:
    """
    Extract discharge summary fields from the given page images.

    Multi-page discharge summaries are sent together in one call.
    """
    llm = ChatGoogleGenerativeAI(
        model=settings.EXTRACTION_MODEL,
        google_api_key=settings.GOOGLE_API_KEY,
        temperature=0,
        max_output_tokens=2048,
    )

    structured_llm = llm.with_structured_output(DischargeExtraction)

    content: list[dict] = [
        {"type": "text", "text": "Extract discharge summary information from the following document page(s):"}
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

    result: DischargeExtraction = structured_llm.invoke(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            message,
        ]
    )

    return result.model_dump()
