import re

from src.tools import parse_money, parse_date, flag_missing_fields


def parse_escalation(text: str) -> float | None:
        match = re.search(r'(\d+(?:\.\d+)?)\s*%', text)
        return float(match.group(1)) if match else None

def run_react(extracted: dict, lease_text: str) -> tuple[dict, list[dict, list[str]]]:
    """
    Returns: (updated_extracted, tool_trace, flags)

    Loop up to 5 iterations:
        1. Check flag_missing_fields(extracted)
        2. If no flag -> break
        3. For each flag, try to fix it:
            - "monthly_rent" -> call parse_money on lease_text
            - "commencement" -> call parse_date on lease_text
            - "escalation_pct" -> call parse_money on lease_text
        4. Append each tool call + result to tool_trace
        5. Re-check flags
    Return final extracted, the trace, and remaining flags.    
    """

    tool_trace = []

    for _ in range(5):
        flags = flag_missing_fields(extracted)
        if not flags:
            break

        for flag in flags:
            if flag == "monthly_rent":
                result = parse_money(lease_text)
                tool_trace.append({"tool": "parse_money", "input": lease_text[:80], "output": result})
                extracted["monthly_rent"] = result
            elif flag == "commencement":
                result = parse_date(lease_text)
                tool_trace.append({"tool": "parse_date", "input": lease_text[:80], "output": result})
                extracted["commencement"] = result
            elif flag == "escalation_pct":
                result = parse_escalation(lease_text)
                tool_trace.append({"tool": "parse_escalation", "input": lease_text[:80], "output": result})
                extracted["escalation_pct"] = result

    flags = flag_missing_fields(extracted)
    return extracted, tool_trace, flags
