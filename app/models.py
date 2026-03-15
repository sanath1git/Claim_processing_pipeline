"""
Pydantic models for the claim processing pipeline.

Defines the LangGraph state, page bundles, and structured output schemas
used by extraction agents.
"""

from __future__ import annotations

from typing import Annotated, Any
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Page-level data carrier
# ---------------------------------------------------------------------------

class PageBundle(BaseModel):
    """Represents a single page extracted from the uploaded PDF."""

    page_number: int = Field(..., description="1-indexed page number")
    image_base64: str = Field(..., description="Base64-encoded PNG of the page at render DPI")
    text_content: str = Field(default="", description="Native text extracted via pdfplumber (may be empty for scanned pages)")
    has_native_text: bool = Field(default=False, description="True if meaningful native text was found")


# ---------------------------------------------------------------------------
# Extraction schemas (structured output from GPT-4o)
# ---------------------------------------------------------------------------

class IDExtraction(BaseModel):
    """Fields extracted from identity / insurance card pages."""

    patient_name: str | None = Field(default=None, description="Full name of the patient")
    date_of_birth: str | None = Field(default=None, description="Date of birth (YYYY-MM-DD if possible)")
    id_number: str | None = Field(default=None, description="Government or national ID number")
    policy_number: str | None = Field(default=None, description="Insurance policy number")
    insurer: str | None = Field(default=None, description="Name of the insurance company")


class DischargeExtraction(BaseModel):
    """Fields extracted from discharge summary pages."""

    diagnosis: list[str] = Field(default_factory=list, description="List of diagnosed conditions")
    admit_date: str | None = Field(default=None, description="Hospital admission date")
    discharge_date: str | None = Field(default=None, description="Hospital discharge date")
    physician_name: str | None = Field(default=None, description="Attending / discharging physician")
    hospital_name: str | None = Field(default=None, description="Name of the hospital")


class LineItem(BaseModel):
    """A single line item on an itemized bill."""

    description: str = Field(..., description="Description of the service / item")
    quantity: float | None = Field(default=None, description="Quantity")
    unit_price: float | None = Field(default=None, description="Price per unit")
    amount: float | None = Field(default=None, description="Line total")


class BillExtraction(BaseModel):
    """Fields extracted from itemized bill pages."""

    line_items: list[LineItem] = Field(default_factory=list, description="All bill line items")
    total_amount: float | None = Field(default=None, description="Grand total amount")
    currency: str = Field(default="INR", description="Currency code")


# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------

class ClaimState(BaseModel):
    """
    The state object passed between LangGraph nodes.

    Fields are progressively populated as each node runs.
    """

    claim_id: str = ""
    pages: list[PageBundle] = Field(default_factory=list)
    routing_map: dict[str, list[int]] = Field(default_factory=dict)  # doc_type -> [page_nums]
    id_data: dict[str, Any] = Field(default_factory=dict)
    discharge_data: dict[str, Any] = Field(default_factory=dict)
    bill_data: dict[str, Any] = Field(default_factory=dict)
    final_output: dict[str, Any] = Field(default_factory=dict)
    confidence_flags: list[str] = Field(default_factory=list)
