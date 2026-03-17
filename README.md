# Claim Processing Pipeline

FastAPI service for processing medical claim PDFs using a LangGraph-based multi-agent workflow.

## Overview

This project ingests a claim PDF, classifies each page into document types, routes relevant pages to specialized extraction agents, and returns one consolidated JSON response.

Core stack:
- FastAPI for API layer
- LangGraph for orchestration
- Gemini (via LangChain) for classification and extraction
- PyMuPDF + pdfplumber for PDF preprocessing

## Assignment Requirement Mapping

- `POST /api/process` endpoint: implemented in `app/main.py`
- Input fields: `claim_id` (string) and `file` (PDF)
- Output: JSON with extracted data
- Segregator classifies pages into 9 required document types
- Only 3 extraction agents run on routed pages:
  - ID Agent
  - Discharge Summary Agent
  - Itemized Bill Agent
- Aggregator returns final combined JSON

## Development Architecture

### 1) API Layer
- File: `app/main.py`
- Responsibilities:
  - Request validation (`claim_id`, PDF upload, size limits)
  - PDF signature validation (`%PDF`)
  - Pipeline invocation and JSON response handling

### 2) PDF Preprocessing
- File: `app/pdf_processor.py`
- Responsibilities:
  - Convert each PDF page to base64 PNG
  - Extract native text where available
  - Emit `PageBundle` objects for downstream agents

### 3) Workflow Orchestration (LangGraph)
- File: `app/graph.py`
- Graph nodes:
  - `preprocess`
  - `segregate`
  - `extract_id`
  - `extract_discharge`
  - `extract_bill`
  - `assemble`

Execution shape:
- `preprocess` -> `segregate`
- Fan-out: `segregate` -> all three extraction nodes
- Each extraction node processes only relevant routed pages (or exits with empty data)
- Fan-in: all extraction nodes -> `assemble`
- `assemble` -> `END`

### 4) Segregator Agent
- File: `app/agents/segregator.py`
- Classifies pages into required document types:
  - `claim_forms`
  - `cheque_or_bank_details`
  - `identity_document`
  - `itemized_bill`
  - `discharge_summary`
  - `prescription`
  - `investigation_report`
  - `cash_receipt`
  - `other`
- Returns `routing_map` plus warnings when parsing fails

### 5) Extraction Agents
- `app/agents/id_agent.py`: patient identity fields
- `app/agents/discharge_agent.py`: clinical discharge fields
- `app/agents/bill_agent.py`: billing line items and totals

Routing rule:
- Extraction agents never receive the full PDF; each agent gets only routed page bundles from the segregator output.

### 6) Aggregation
- File: `app/graph.py` (`assemble_node`)
- Combines:
  - claim metadata
  - routing map
  - ID extraction output
  - discharge extraction output
  - bill extraction output
  - confidence flags

## API Contract

### Endpoint
- `POST /api/process`

### Request (multipart/form-data)
- `claim_id`: string
- `file`: PDF file

### Response
- JSON object containing consolidated extraction output.
- Current response includes fields such as:
  - `claim_id`
  - `processed_at`
  - `pages_processed`
  - `routing_map`
  - `identity`
  - `discharge_summary`
  - `itemized_bill`
  - `confidence_flags`

## Project Structure

- `app/main.py` - FastAPI entrypoint and endpoint logic
- `app/config.py` - environment and runtime settings
- `app/pdf_processor.py` - PDF-to-page preprocessing
- `app/models.py` - Pydantic schemas
- `app/graph.py` - LangGraph state + nodes + graph wiring
- `app/agents/segregator.py` - page classification agent
- `app/agents/id_agent.py` - identity extraction agent
- `app/agents/discharge_agent.py` - discharge extraction agent
- `app/agents/bill_agent.py` - bill extraction agent

## Local Setup (Recommended: uv)

Prerequisites:
- Python 3.12+
- `uv`
- Google Gemini API key

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies with uv:

```powershell
uv pip install -r requirements.txt
```

Create `.env` in repo root:

```text
GOOGLE_API_KEY=your_gemini_api_key_here
```

Run the API:

```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open docs:
- `http://localhost:8000/docs`

## Docker

Build image:

```bash
docker build -t claim-processing:latest .
```

Run container:

```bash
docker run --rm -p 8000:8000 -e GOOGLE_API_KEY=your_gemini_api_key_here claim-processing:latest
```

Health check:

```bash
curl http://localhost:8000/health
```

PowerShell alternative:

```powershell
Invoke-RestMethod -Method Get -Uri http://localhost:8000/health
```

## Example Request

```bash
curl -X POST "http://localhost:8000/api/process" \
  -F "claim_id=CLM-001" \
  -F "file=@sample_final.pdf"
```

## Known Limitations

- Extraction quality depends on document clarity and LLM output consistency.
- Low-confidence or unclassified pages are surfaced via `confidence_flags`.
- No automated test suite is included yet.

## Submission Notes

This repository includes:
- Source code
- Architecture documentation
- API usage instructions

Deployed API (Render):
- https://claim-processing-pipeline-fg71.onrender.com/docs
