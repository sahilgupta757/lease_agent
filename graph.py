from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from src.nodes import auto_commit, extract_terms, human_review, score, validate, write_audit


class LeaseState(TypedDict):
    lease_text: str
    extracted: dict
    tool_trace: list
    confidence: float
    flags: list
    decision: str                 # "auto_approved" / "flagged_for_review"
    audit_id: str


def route_decision(state: LeaseState) -> str:
    return "auto_commit" if state["confidence"] >= 0.8 else "human_review"


graph_builder = StateGraph(LeaseState)
graph_builder.add_node("extract_terms", extract_terms)
graph_builder.add_node("validate", validate)
graph_builder.add_node("score", score)
graph_builder.add_node("auto_commit", auto_commit)
graph_builder.add_node("human_review", human_review)
graph_builder.add_node("write_audit", write_audit)

graph_builder.add_edge(START, "extract_terms")
graph_builder.add_edge("extract_terms", "validate")
graph_builder.add_edge("validate", "score")

graph_builder.add_conditional_edges(
    "score",
    route_decision,
    {"auto_commit": "auto_commit", "human_review": "human_review"},
)

graph_builder.add_edge("auto_commit", "write_audit")
graph_builder.add_edge("human_review", "write_audit")
graph_builder.add_edge("write_audit", END)

graph = graph_builder.compile()
