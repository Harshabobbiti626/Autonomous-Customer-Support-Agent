from typing import Any

from langchain_core.messages import AIMessage

from src.guardrails.output_checker import OutputChecker
from src.models.state import AgentState


def output_guard_node(state: AgentState) -> dict[str, Any]:
    """Validates final model outputs against policy constraints and disambiguates multi-order context."""
    messages = list(state.get("messages", []))
    if not messages:
        return {}

    last_msg = messages[-1]
    if not isinstance(last_msg, AIMessage):
        return {}

    content = last_msg.content

    # If disambiguation is needed because customer has multiple orders and gave no specific order ID
    if state.get("disambiguation_required") and not state.get("active_order"):
        candidate_orders = state.get("candidate_orders", [])
        if candidate_orders:
            lines = [
                "I see you have multiple recent orders on your account. Which one can I help you with?",
                "",
            ]
            for ord_info in candidate_orders[:4]:
                lines.append(
                    f"• **{ord_info['order_id']}** (Status: *{ord_info['status']}*, Total: ${ord_info['amount']:.2f})"
                )
            lines.append("\nPlease let me know the order number so I can retrieve live details.")
            last_msg.content = "\n".join(lines)
            return {"messages": [last_msg]}

    # Verify against policy hallucination
    is_valid, sanitized_content = OutputChecker.verify_output(content)
    if not is_valid:
        last_msg.content = sanitized_content

    return {"messages": [last_msg]}
