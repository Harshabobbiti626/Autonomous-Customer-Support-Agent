from typing import Any, Literal

from langchain_core.messages import HumanMessage

from src.config import settings
from src.models.state import AgentState


def router_node(state: AgentState) -> dict[str, Any]:
    """Classifies intent and sets high-level workflow routing."""
    if state.get("escalated"):
        return {"intent": "human_escalation"}

    # Check negative sentiment threshold
    if state.get("sentiment_score", 0.0) <= settings.NEGATIVE_SENTIMENT_THRESHOLD:
        return {
            "intent": "human_escalation",
            "escalated": True,
            "escalation_reason": "Customer sentiment dropped below threshold (-0.60)",
        }

    # Inspect last user message
    messages = state.get("messages", [])
    user_text = ""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            user_text = m.content.lower()
            break

    # Check explicit human handoff keywords
    escalation_keywords = [
        "human",
        "representative",
        "agent",
        "real person",
        "manager",
        "supervisor",
    ]
    if any(k in user_text for k in escalation_keywords):
        return {
            "intent": "human_escalation",
            "escalated": True,
            "escalation_reason": "Customer explicitly requested a human representative",
        }

    # Check refund / return intent
    if any(k in user_text for k in ["refund", "return", "broken", "defective", "money back"]):
        return {"intent": "refund_request"}

    # Check order tracking intent
    if any(k in user_text for k in ["track", "status", "where is", "order", "delivery", "ord-"]):
        return {"intent": "order_tracking"}

    return {"intent": "general_chat"}


def route_decision(state: AgentState) -> Literal["escalation", "support_worker"]:
    """Conditional edge decision function."""
    if state.get("escalated") or state.get("intent") == "human_escalation":
        return "escalation"
    return "support_worker"
