from typing import Any

from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig

from src.agent.llm_factory import get_llm
from src.agent.prompts import SYSTEM_PROMPT
from src.config import settings
from src.models.state import AgentState
from src.tools.tool_registry import ToolRegistry


def trim_conversation_history(messages: list, max_messages: int = 12) -> list:
    """Trim long conversation histories while preserving the SystemMessage and valid tool pairs."""
    if len(messages) <= max_messages:
        return messages

    system_msg = messages[0] if isinstance(messages[0], SystemMessage) else None
    working_messages = messages[1:] if system_msg else messages

    # Keep the latest messages
    trimmed_tail = working_messages[-(max_messages - 1) :]

    # Ensure we don't start with an orphaned ToolMessage without its preceding tool_call AIMessage
    while (
        trimmed_tail
        and not hasattr(trimmed_tail[0], "tool_calls")
        and getattr(trimmed_tail[0], "tool_call_id", None)
    ):
        trimmed_tail.pop(0)

    result = [system_msg] + trimmed_tail if system_msg else trimmed_tail
    return result


def build_dynamic_system_prompt(state: AgentState) -> str:
    """Build dynamic system prompt grounded in real-time SQL customer and order data."""
    customer = state.get("customer")
    active_order = state.get("active_order")
    candidate_orders = state.get("candidate_orders", [])
    refund_eligibility = state.get("refund_eligibility")

    lines = [SYSTEM_PROMPT, "\n=== AUTHENTICATED CUSTOMER CONTEXT (LIVE SQL GROUND TRUTH) ==="]
    if customer:
        lines.append(
            f"Customer Name: {customer.full_name} | ID: {customer.user_id} | Tier: {customer.tier} | Email: {customer.email}"
        )

    if active_order:
        items_summary = (
            ", ".join(
                [
                    f"{it.quantity}x {it.product_name} (${it.unit_price:.2f})"
                    for it in active_order.items
                ]
            )
            or "None"
        )
        lines.append(
            f"Active Order: {active_order.order_id}\n"
            f"  - Fulfillment Status: {active_order.fulfillment_status}\n"
            f"  - Total Amount: ${active_order.total_amount:.2f} {active_order.currency}\n"
            f"  - Carrier: {active_order.carrier_name or 'N/A'}\n"
            f"  - Tracking Number: {active_order.tracking_number or 'N/A'}\n"
            f"  - Latest Checkpoint: {active_order.last_checkpoint or 'N/A'}\n"
            f"  - Delivered At: {active_order.delivered_at or 'Not Delivered'}\n"
            f"  - Order Items: [{items_summary}]"
        )
    elif candidate_orders:
        orders_str = "; ".join(
            [f"{o['order_id']} ({o['status']}, ${o['amount']:.2f})" for o in candidate_orders]
        )
        lines.append(f"Recent Orders on File: {orders_str}")
    else:
        lines.append("Recent Orders on File: None")

    if refund_eligibility:
        lines.append(
            f"Refund Evaluation: Order {refund_eligibility.order_id} | Eligible: {refund_eligibility.eligible} | "
            f"Max Amount: ${refund_eligibility.max_refundable_amount:.2f} | Reason: {refund_eligibility.reason} | "
            f"Supervisor Approval Required: {refund_eligibility.requires_human_approval}"
        )

    if state.get("refund_executed"):
        lines.append(f"Refund Status: EXECUTED (Tx: {state.get('refund_transaction_id')})")

    lines.append("=================================================================")
    return "\n".join(lines)


async def support_worker_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """Core agent reasoning node: invokes tool-calling LLM with customer-scoped tools and trimmed window."""
    session = config.get("configurable", {}).get("db_session")
    customer = state["customer"]

    tools = ToolRegistry.get_tools_for_customer(session=session, customer_id=customer.user_id)
    llm = get_llm(tools=tools)

    messages = list(state.get("messages", []))

    # Prepend or refresh dynamic system prompt with real-time ground truth from state
    dynamic_sys_prompt = build_dynamic_system_prompt(state)
    if not messages or not isinstance(messages[0], SystemMessage):
        messages.insert(0, SystemMessage(content=dynamic_sys_prompt))
    else:
        messages[0] = SystemMessage(content=dynamic_sys_prompt)

    # Trim conversation window to prevent token bloat and runaway costs
    trimmed_messages = trim_conversation_history(
        messages, max_messages=settings.MAX_CONVERSATION_TURNS
    )

    response = await llm.ainvoke(trimmed_messages)
    return {"messages": [response]}
