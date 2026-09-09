import json
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.llm import get_llm
from src.tools import annualize_rent, flag_missing_fields, parse_date, parse_money

TOOLS = [parse_money, parse_date, annualize_rent]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}

SYSTEM_PROMPT = """You are a lease-data assistant helping fill in fields that a first-pass \
extraction could not find in a commercial lease document.

You have tools: parse_money(text), parse_date(text), annualize_rent(monthly).
These tools do simple pattern matching, not full-document reasoning, so when you call one \
pass a SHORT, relevant excerpt of the lease text (the sentence containing the value) rather \
than the entire document.

Work through the missing fields one at a time. If a tool cannot find a value and you are \
confident of the value yourself from reading the lease text, you may use your own reading \
instead of insisting on a tool result.

When you are done investigating, do not call any more tools. You will then be asked to \
report your final answers.
"""


def _extract_json(text: str) -> dict:
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {}


def run_react(
    extracted: dict, lease_text: str, max_iterations: int = 5
) -> tuple[dict, list[dict], list[str]]:
    """
    ReAct loop: the LLM decides which tool to call (Act), observes the result
    (Observe), and reasons about what to do next (Reason) until it stops
    requesting tools or we hit max_iterations. It then reports final values
    for the fields that were originally missing.

    Returns: (updated_extracted, tool_trace, remaining_flags)
    """
    flags = flag_missing_fields(extracted)
    if not flags:
        return extracted, [], []

    tool_trace: list[dict] = []
    llm_with_tools = get_llm().bind_tools(TOOLS)

    messages: list[AIMessage | HumanMessage | SystemMessage | ToolMessage] = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Missing fields: {flags}\n\n"
                f"Current extracted data: {json.dumps(extracted)}\n\n"
                f"Lease text:\n{lease_text}"
            )
        ),
    ]

    for _ in range(max_iterations):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            break

        for call in response.tool_calls:
            tool_fn = TOOLS_BY_NAME.get(call["name"])
            result = tool_fn.invoke(call["args"]) if tool_fn else None
            tool_trace.append({"tool": call["name"], "input": call["args"], "output": result})
            messages.append(ToolMessage(content=json.dumps(result), tool_call_id=call["id"]))

    messages.append(
        HumanMessage(
            content=(
                "Based on everything above, output ONLY a JSON object mapping each of these "
                f"fields {flags} to its resolved value (or null if still unknown). "
                "No prose, no markdown fences, just the JSON object."
            )
        )
    )
    final_response = get_llm().invoke(messages)  # fresh, tool-free call so it answers in plain JSON
    resolved = _extract_json(final_response.content)

    for field in flags:
        value = resolved.get(field)
        if value is not None:
            extracted[field] = value

    remaining_flags = flag_missing_fields(extracted)
    return extracted, tool_trace, remaining_flags
