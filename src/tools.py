import re
from dateutil import parser as dateparse


def parse_money(text: str) -> float | None:
    """
    extract a dollar amount from strings like "$12,500 per month" or "twelve thousand five hundred"
    """

    # extract the dollar amount from the text
    dollar_amount = re.search(r'\$(\d{1,3}(,\d{3})*)', text)
    if dollar_amount:
        return float(dollar_amount.group(1).replace(',', ''))
    else:
        return None
    
    #TODO: add support for converting english words to dollar amount


def parse_date(text: str) -> str | None:
    try:
        dt = dateparse.parse(text, fuzzy=True)
        return dt.strftime("%Y-%m-%d")
    except (ValueError, OverflowError):
        return None

def annualize_rent(monthly: float) -> float:
    """
    just monthly * 12
    """
    return monthly * 12

def flag_missing_fields(extracted: dict) -> list[str]:
    """
    takes a dict like {"tenant": "....", "monthly_rent": 12500, "term_months}: 60, "commencement": None, "escalation_pct": None}
    and returns a list of keys that are None or missing.
    """

    required = ["tenant", "monthly_rent", "term_months", "commencement", "escalation_pct"]
    return [k for k in required if extracted.get(k) is None or extracted.get(k) == ""]
