"""
Identity Document Extraction Agent.

Uses Gemini with structured output to extract patient identity fields
from identity / insurance card pages.
"""

from __future__ import annotations

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from app.config import settings
from app.models import PageBundle, IDExtraction


SYSTEM_PROMPT = """You are a medical document data extractor. You will be shown images of identity documents (insurance cards, government IDs, etc.) from a medical claim.

Extract the following fields:
- patient_name: Full name of the patient / insured person
- date_of_birth: Date of birth in YYYY-MM-DD format
- id_number: Government or national ID number (Aadhaar, PAN, passport number, etc.)
- policy_number: Insurance policy / member ID number
- insurer: Name of the insurance company

If a field is not visible or cannot be determined, return null for that field.
Return ONLY valid JSON matching the schema."""


def extract_identity(pages: list[PageBundle]) -> dict:
    """
    Extract identity fields from the given page images.

    All pages are sent in a single LLM call (multi-page identity docs
    are processed together).
    """
    llm = ChatGoogleGenerativeAI(
        model=settings.EXTRACTION_MODEL,
        google_api_key=settings.GOOGLE_API_KEY,
        temperature=0,
        max_output_tokens=2048,
    )

    structured_llm = llm.with_structured_output(IDExtraction)

    content: list[dict] = [
        {"type": "text", "text": "Extract identity information from the following document page(s):"}
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
        # Include any native text as supplementary context
        if p.has_native_text and p.text_content:
            content.append(
                {"type": "text", "text": f"[Native text from page {p.page_number}]: {p.text_content}"}
            )

    message = HumanMessage(content=content)

    result: IDExtraction = structured_llm.invoke(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            message,
        ]
    )

    return result.model_dump()
