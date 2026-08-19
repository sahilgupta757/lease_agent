from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class LeaseSchema(TypedDict):
    lease_text: ste
    extrated: dict
    tool_trace: list
    confidence: float
    flags: list
    decision: str                 # "auto-approved" / "human_approved" / "rejected"
    audit_id: str


graph_builder = StateGraph(LeaseSchema)
graph_builder.add_node("extract_text", extract_terms)
graph_builder.add_node("validate", validate)
graph_builder.add_node("score", score)
graph_builder.add_node("auto_commit", auto_commit)
graph_builder.add_node("human_review", human_review)
graph_builder.add_node("write_audit", write_audit)

graph_builder.add_edge(START, "extract_terms")
graph_builder.add_edge("extract_terms", "validate")
graph_builder.add_edge("validate", "score")

# Conditional: route based on confidence
def route_decision(state):
    return "auto_commit" if state["confidence"] > 0.8 else "human_review"


graph_builder.add_conditional_edges("score", route_decision)
graph_builder.add_edge("auto_commit", "write_audit")
graph_builder.add_edge("human_review", "write_audit")
graph_builder.add_edge("write_audit", END)

graph = graph_builder.compile()
