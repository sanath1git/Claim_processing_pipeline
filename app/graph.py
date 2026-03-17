"""
LangGraph workflow — orchestrates the claim processing pipeline.

State graph:
  preprocess → segregate → [extract_id, extract_discharge, extract_bill] → assemble

Uses conditional fan-out so extraction agents only run if the segregator
assigned pages to their document type.

KEY DESIGN: Uses TypedDict (not bare dict) as state schema so LangGraph
registers channels for every key upfront. Uses operator.add reducer on
confidence_flags so parallel branches can append without overwriting.
"""

from __future__ import annotations

import logging
import operator
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Annotated, TypedDict

from langgraph.graph import StateGraph, END

from app.models import PageBundle
from app.agents.segregator import classify_pages
from app.agents.id_agent import extract_identity
from app.agents.discharge_agent import extract_discharge
from app.agents.bill_agent import extract_bill

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Typed state schema — this is what makes state propagation work
# ---------------------------------------------------------------------------

class GraphState(TypedDict, total=False):
    """
    LangGraph state with explicit channels.

    Every key here gets a registered channel so it persists across
    node transitions. Without this (i.e. using bare `dict`), keys
    from the initial state silently vanish.

    confidence_flags uses operator.add as a reducer so parallel
    extraction branches can each append flags without overwriting
    each other.
    """
    claim_id: str
    pages: list[dict]                                     # serialised PageBundles
    routing_map: dict[str, list[int]]
    id_data: dict[str, Any]
    discharge_data: dict[str, Any]
    bill_data: dict[str, Any]
    final_output: dict[str, Any]
    confidence_flags: Annotated[list[str], operator.add]  # ← reducer for parallel merge


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

def preprocess_node(state: GraphState) -> dict:
    """Pass-through — pages are injected into state before the graph runs."""
    logger.info("Preprocess: %d pages loaded", len(state.get("pages", [])))
    return {}


def segregate_node(state: GraphState) -> dict:
    """Classify each page by document type."""
    pages_data = state.get("pages", [])
    pages = [
        PageBundle(**p) if isinstance(p, dict) else p
        for p in pages_data
    ]
    routing_map, warnings = classify_pages(pages)
    logger.info("Segregation result: %s", routing_map)
    updates: dict[str, Any] = {"routing_map": routing_map}
    if warnings:
        updates["confidence_flags"] = warnings
    return updates


def extract_id_node(state: GraphState) -> dict:
    """Extract identity fields from assigned pages."""
    routing_map = state.get("routing_map", {})
    page_nums = routing_map.get("identity_document", [])
    if not page_nums:
        return {"id_data": {}}

    pages_data = state.get("pages", [])
    pages = [
        PageBundle(**p) if isinstance(p, dict) else p
        for p in pages_data
    ]
    relevant_pages = [p for p in pages if p.page_number in page_nums]
    logger.info("ID Agent: processing pages %s", page_nums)

    try:
        id_data = extract_identity(relevant_pages)
    except Exception as e:
        logger.error("ID extraction failed: %s", e)
        return {"id_data": {}, "confidence_flags": [f"ID extraction failed: {e}"]}

    return {"id_data": id_data}


def extract_discharge_node(state: GraphState) -> dict:
    """Extract discharge summary fields from assigned pages."""
    routing_map = state.get("routing_map", {})
    page_nums = routing_map.get("discharge_summary", [])
    if not page_nums:
        return {"discharge_data": {}}

    pages_data = state.get("pages", [])
    pages = [
        PageBundle(**p) if isinstance(p, dict) else p
        for p in pages_data
    ]
    relevant_pages = [p for p in pages if p.page_number in page_nums]
    logger.info("Discharge Agent: processing pages %s", page_nums)

    try:
        discharge_data = extract_discharge(relevant_pages)
    except Exception as e:
        logger.error("Discharge extraction failed: %s", e)
        return {"discharge_data": {}, "confidence_flags": [f"Discharge extraction failed: {e}"]}

    return {"discharge_data": discharge_data}


def extract_bill_node(state: GraphState) -> dict:
    """Extract billing data from assigned pages."""
    routing_map = state.get("routing_map", {})
    page_nums = routing_map.get("itemized_bill", [])
    if not page_nums:
        return {"bill_data": {}}

    pages_data = state.get("pages", [])
    pages = [
        PageBundle(**p) if isinstance(p, dict) else p
        for p in pages_data
    ]
    relevant_pages = [p for p in pages if p.page_number in page_nums]
    logger.info("Bill Agent: processing pages %s", page_nums)

    try:
        bill_data = extract_bill(relevant_pages)
    except Exception as e:
        logger.error("Bill extraction failed: %s", e)
        return {"bill_data": {}, "confidence_flags": [f"Bill extraction failed: {e}"]}

    # Enforce deterministic total calculation from extracted line items.
    normalized_bill_data, bill_flags = _normalize_bill_totals(bill_data)
    if bill_flags:
        return {"bill_data": normalized_bill_data, "confidence_flags": bill_flags}

    return {"bill_data": normalized_bill_data}


def _to_decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None


def _normalize_bill_totals(bill_data: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Derive bill total from line items and reconcile model-provided totals."""
    flags: list[str] = []
    line_items = bill_data.get("line_items") or []
    computed_total = Decimal("0")
    has_any_amount = False

    for item in line_items:
        if not isinstance(item, dict):
            continue
        amount = _to_decimal(item.get("amount"))
        if amount is None:
            qty = _to_decimal(item.get("quantity"))
            unit_price = _to_decimal(item.get("unit_price"))
            if qty is not None and unit_price is not None:
                amount = qty * unit_price
        if amount is not None:
            computed_total += amount
            has_any_amount = True

    if not has_any_amount:
        return bill_data, flags

    rounded_total = float(computed_total.quantize(Decimal("0.01")))
    model_total = _to_decimal(bill_data.get("total_amount"))

    # Always provide a deterministic total; flag when we had to correct mismatch.
    normalized = dict(bill_data)
    normalized["total_amount"] = rounded_total
    if model_total is not None and abs(model_total - computed_total) > Decimal("0.5"):
        flags.append(
            f"Bill total mismatch detected. Using computed total {rounded_total:.2f} instead of model total {float(model_total):.2f}."
        )

    return normalized, flags


def assemble_node(state: GraphState) -> dict:
    """Merge all extraction results into the final JSON output."""
    pages_data = state.get("pages", [])

    # Build confidence flags for unclassified pages
    flags = list(state.get("confidence_flags", []))
    routing_map = state.get("routing_map", {})
    other_pages = routing_map.get("other", [])
    if other_pages:
        flags.append(f"Unclassified pages (require human review): {other_pages}")

    final_output = {
        "claim_id": state.get("claim_id", ""),
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "pages_processed": len(pages_data),
        "routing_map": routing_map,
        "identity": state.get("id_data", {}),
        "discharge_summary": state.get("discharge_data", {}),
        "itemized_bill": state.get("bill_data", {}),
        "confidence_flags": flags,
    }
    return {"final_output": final_output}


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph():
    """Build and compile the LangGraph claim processing workflow."""

    graph = StateGraph(GraphState)
    # Add nodes
    graph.add_node("preprocess", preprocess_node)
    graph.add_node("segregate", segregate_node)
    graph.add_node("extract_id", extract_id_node)
    graph.add_node("extract_discharge", extract_discharge_node)
    graph.add_node("extract_bill", extract_bill_node)
    graph.add_node("assemble", assemble_node)

    # Set entry point
    graph.set_entry_point("preprocess")

    # preprocess → segregate
    graph.add_edge("preprocess", "segregate")

    # Deterministic fan-out/fan-in: all extractor nodes run, but each node only
    # calls its LLM when routing_map has relevant pages.
    graph.add_edge("segregate", "extract_id")
    graph.add_edge("segregate", "extract_discharge")
    graph.add_edge("segregate", "extract_bill")

    # Wait for all extractors to complete before assembling final output.
    graph.add_edge(["extract_id", "extract_discharge", "extract_bill"], "assemble")

    # assemble → END
    graph.add_edge("assemble", END)

    return graph.compile()


# Pre-compiled graph instance
claim_pipeline = build_graph()
