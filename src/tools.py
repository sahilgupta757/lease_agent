import re

from dateutil import parser as dateparse
from langchain_core.tools import tool

from src.rag import search_market_context

_ONES = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}
_SCALES = {"hundred": 100, "thousand": 1_000, "million": 1_000_000}
_NUMBER_WORDS = set(_ONES) | set(_TENS) | set(_SCALES) | {"and"}


def _words_to_number(words: list[str]) -> int | None:
    total = 0
    current = 0
    found = False
    for raw in words:
        word = raw.lower()
        if word in _ONES:
            current += _ONES[word]
            found = True
        elif word in _TENS:
            current += _TENS[word]
            found = True
        elif word in _SCALES:
            scale = _SCALES[word]
            if scale == 100:
                current = (current or 1) * scale
            else:
                total += (current or 1) * scale
                current = 0
            found = True
        elif word == "and":
            continue
        else:
            break
    return (total + current) if found else None


@tool
def parse_money(text: str) -> float | None:
    """Extract a dollar amount from a short piece of lease text.

    Handles numeric forms like "$12,500 per month" as well as spelled-out
    amounts like "twelve thousand five hundred monthly". Pass a short,
    relevant excerpt (the sentence containing the amount) rather than an
    entire document, since this is pattern matching, not full-document
    reasoning. Returns the amount as a float, or None if no amount is found.
    """
    numeric = re.search(r"\$\s?(\d{1,3}(?:,\d{3})*(?:\.\d+)?)", text)
    if numeric:
        return float(numeric.group(1).replace(",", ""))

    word_match = re.search(
        r"((?:\b(?:" + "|".join(_NUMBER_WORDS) + r")\b\s*)+)", text, re.IGNORECASE
    )
    if word_match:
        value = _words_to_number(word_match.group(1).split())
        if value:
            return float(value)

    return None


@tool
def parse_date(text: str) -> str | None:
    """Extract a calendar date from a short piece of lease text and return
    it in YYYY-MM-DD form (e.g. "commencing on 1 January 2025" -> "2025-01-01").
    Returns None if no parseable date is present.
    """
    try:
        dt = dateparse.parse(text, fuzzy=True)
        return dt.strftime("%Y-%m-%d")
    except (ValueError, OverflowError):
        return None


@tool
def annualize_rent(monthly: float) -> float:
    """Convert a monthly rent figure into an annualized figure (monthly * 12)."""
    return monthly * 12


@tool
def lookup_market_context(query: str) -> list[str]:
    """Search a small knowledge base of comparable lease deals and glossary
    entries for context on ambiguous or market-referenced lease language
    (e.g. "market standard escalation", "base year stop", "six months rent").

    Use this to INFORM a confidence assessment or a reviewer note with what
    comparable deals or industry norms typically look like. Do NOT use its
    results to fill in or overwrite a field the lease itself never states a
    number for — a comparable deal is context, not a fact about this lease.
    """
    return search_market_context(query)


def flag_missing_fields(extracted: dict) -> list[str]:
    """Return the list of required lease fields that are missing or empty.

    Takes a dict like {"tenant": "...", "monthly_rent": 12500, "term_months": 60,
    "commencement": None, "escalation_pct": None} and returns the keys that are
    None or missing. This is plain orchestration logic, not an LLM tool.
    """
    required = ["tenant", "monthly_rent", "term_months", "commencement", "escalation_pct"]
    return [k for k in required if extracted.get(k) is None or extracted.get(k) == ""]
