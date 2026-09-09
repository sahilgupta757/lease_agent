import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from react import run_react
from src.llm import get_llm
from src.rag import search_market_context


class LeaseTerms(BaseModel):
    tenant: str | None = Field(None, description="Legal name of the tenant entity")
    monthly_rent: float | None = Field(None, description="Base monthly rent as a plain number, e.g. 12500.0")
    term_months: int | None = Field(None, description="Total lease term length in months")
    commencement: str | None = Field(None, description="Commencement date in YYYY-MM-DD form, if stated or determinable")
    escalation_pct: float | None = Field(None, description="Annual rent escalation percentage, e.g. 3.0 for 3%")


class ConfidenceAssessment(BaseModel):
    confidence: float = Field(..., ge=0, le=1, description="Overall confidence that extracted data is complete and correct")
    reasoning: str = Field(..., description="One or two sentences on what drove the score")
    risk_notes: list[str] = Field(
        default_factory=list,
        description="Short notes on ambiguous or unresolved language a human reviewer should know about, even for fields that have a value",
    )


class ReviewSummary(BaseModel):
    summary: str = Field(..., description="Plain-language summary of the lease for a human reviewer")
    questions_for_reviewer: list[str] = Field(..., description="Specific questions the reviewer needs to resolve before approval")


def retrieve_context(state: dict) -> dict:
    """No LLM call: embeds the raw lease text and pulls the top-k most similar
    comparable-deal / glossary snippets from the local knowledge base. This is
    naive/always-on RAG — retrieval happens unconditionally, before any
    reasoning about what's actually missing."""
    snippets = search_market_context(state["lease_text"], k=3)
    return {"retrieved_context": snippets}


def extract_terms(state: dict) -> dict:
    llm = get_llm().with_structured_output(LeaseTerms)
    context_block = "\n\n".join(state.get("retrieved_context", []))
    result: LeaseTerms = llm.invoke(
        "Extract the lease terms below from this commercial lease document. "
        "If a field is not stated or is ambiguous, leave it null rather than guessing — "
        "the reference market data below is background only, never a substitute for what "
        "the lease itself does or doesn't say.\n\n"
        f"Reference market data (context only):\n{context_block}\n\n"
        f"Lease text:\n{state['lease_text']}"
    )
    return {"extracted": result.model_dump(), "tool_trace": []}


def validate(state: dict) -> dict:
    extracted, trace, flags = run_react(dict(state["extracted"]), state["lease_text"])
    return {"extracted": extracted, "tool_trace": state["tool_trace"] + trace, "flags": flags}


def score(state: dict) -> dict:
    llm = get_llm().with_structured_output(ConfidenceAssessment)
    context_block = "\n\n".join(state.get("retrieved_context", []))
    result: ConfidenceAssessment = llm.invoke(
        "Assess how confident we should be in this lease data extraction, on a 0-1 scale.\n"
        "Lower the score for missing fields, contradictory statements (e.g. two different "
        "rent figures), or language that hedges on a value (e.g. 'roughly', 'TBD', 'market standard'). "
        "If the reference market data below suggests a hedged term (like 'market standard') sits "
        "well outside comparable norms, or confirms it's plausible, say so in a risk note.\n\n"
        f"Reference market data (context only): {context_block}\n\n"
        f"Extracted data: {state['extracted']}\n"
        f"Fields still flagged missing: {state['flags']}\n\n"
        f"Original lease text:\n{state['lease_text']}"
    )
    flags = list(state["flags"])
    for note in result.risk_notes:
        flags.append(f"risk: {note}")
    return {"confidence": result.confidence, "flags": flags}


def auto_commit(state: dict) -> dict:
    return {"decision": "auto_approved"}


def human_review(state: dict) -> dict:
    llm = get_llm().with_structured_output(ReviewSummary)
    context_block = "\n\n".join(state.get("retrieved_context", []))
    result: ReviewSummary = llm.invoke(
        "This lease extraction fell below the auto-approval confidence threshold and needs "
        "human review. Write a short summary and the specific questions a reviewer must "
        "resolve before this lease record can be approved. Where the reference market data "
        "below is relevant to a flagged question (e.g. a plausible number for a vague "
        "escalation clause), mention it as context for the reviewer, not as a resolved fact.\n\n"
        f"Reference market data (context only): {context_block}\n\n"
        f"Extracted data: {state['extracted']}\n"
        f"Flags: {state['flags']}\n\n"
        f"Original lease text:\n{state['lease_text']}"
    )
    trace_entry = {
        "tool": "human_review_summary",
        "input": {"flags": state["flags"]},
        "output": {"summary": result.summary, "questions": result.questions_for_reviewer},
    }
    return {"decision": "flagged_for_review", "tool_trace": state["tool_trace"] + [trace_entry]}


def write_audit(state: dict) -> dict:
    audit_id = str(uuid.uuid4())
    record = {
        "audit_id": audit_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "decision": state["decision"],
        "confidence": state["confidence"],
        "extracted": state["extracted"],
        "flags": state["flags"],
        "tool_trace": state["tool_trace"],
    }

    audit_dir = Path("audit")
    audit_dir.mkdir(exist_ok=True)
    (audit_dir / f"{audit_id}.json").write_text(json.dumps(record, indent=2))

    return {"audit_id": audit_id}
