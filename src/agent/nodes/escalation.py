from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from src.models.state import AgentState, HandoffSummary
from src.services.crm_service import CRMService


async def escalation_node(state: AgentState) -> dict[str, Any]:
    """Generates structured handoff summary, dispatches ticket to CRM, and reassures customer."""
    customer = state["customer"]
    reason = state.get(
        "escalation_reason",
        "Customer requested human representative or sentiment threshold reached",
    )

    # Extract customer's latest question or issue
    primary_issue = "General assistance required"
    for m in reversed(state.get("messages", [])):
        if isinstance(m, HumanMessage):
            primary_issue = m.content[:200]
            break

    # Build 3-bullet structured summary for Zendesk / Salesforce API
    summary = HandoffSummary(
        customer_name=customer.full_name,
        customer_tier=customer.tier,
        primary_issue=primary_issue,
        attempted_actions=[
            "Intent analysis completed",
            "Customer authentication and order context resolved",
        ],
        escalation_reason=reason,
        recommended_action="Review customer account and assist with manual inquiry/exception handling",
    )

    # Outbound CRM Ticket Dispatch
    priority = "urgent" if state.get("sentiment_score", 0.0) < -0.5 else "high"
    crm_result = await CRMService.dispatch_ticket(
        customer=customer, summary=summary, priority=priority
    )
    ticket_ref = crm_result.get("ticket_id", f"SUP-{customer.user_id.upper()}-ESCALATED")

    reassurance_message = AIMessage(
        content=(
            f"I have connected you with our human support team. A specialist has been assigned "
            f"and provided with our full conversation history so you won't need to repeat yourself.\n\n"
            f"📋 **Ticket Reference**: `{ticket_ref}`\n"
            f"**Status**: In queue for priority assistance."
        )
    )

    return {"escalated": True, "handoff_summary": summary, "messages": [reassurance_message]}
