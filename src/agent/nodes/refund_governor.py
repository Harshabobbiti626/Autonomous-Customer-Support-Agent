import json
from typing import Any

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

from src.config import settings
from src.db.services.order_service import OrderService
from src.models.state import AgentState, OrderItemDetail, OrderReference, RefundEligibility
from src.tools.tool_registry import ToolRegistry


async def tool_execution_and_governor_node(
    state: AgentState, config: RunnableConfig
) -> dict[str, Any]:
    """Executes requested tool calls and enforces Human-in-the-Loop breakpoints for high-value operations."""
    session = config.get("configurable", {}).get("db_session")
    customer = state["customer"]

    tools = ToolRegistry.get_tools_for_customer(session=session, customer_id=customer.user_id)
    tools_by_name = {t.name: t for t in tools}

    messages = state.get("messages", [])
    last_msg = messages[-1]

    tool_calls = getattr(last_msg, "tool_calls", [])
    if not tool_calls:
        return {}

    tool_messages: list[ToolMessage] = []
    consecutive_failures = state.get("consecutive_tool_failures", 0)
    state_updates: dict[str, Any] = {}

    for call in tool_calls:
        tool_name = call["name"]
        tool_args = call["args"]
        call_id = call["id"]

        target_tool = tools_by_name.get(tool_name)
        if not target_tool:
            tool_messages.append(
                ToolMessage(
                    content=json.dumps({"error": f"Tool '{tool_name}' is not recognized."}),
                    name=tool_name,
                    tool_call_id=call_id,
                )
            )
            consecutive_failures += 1
            continue

        # HUMAN-IN-THE-LOOP INTERRUPT CHECK:
        # If executing a refund for an order exceeding $50, trigger LangGraph interrupt()
        if tool_name == "execute_refund":
            order_id = tool_args.get("order_id", "").strip().upper()
            order = await OrderService.get_order_for_customer(session, order_id, customer.user_id)

            if order and order.total_amount > settings.MAX_AUTONOMOUS_REFUND_AMOUNT:
                # Halt execution and pause state in checkpointer
                supervisor_decision = interrupt(
                    {
                        "action": "APPROVE_HIGH_VALUE_REFUND",
                        "order_id": order.id,
                        "customer_id": customer.user_id,
                        "customer_name": customer.full_name,
                        "amount": order.total_amount,
                        "reason": f"Refund amount ${order.total_amount:.2f} exceeds ${settings.MAX_AUTONOMOUS_REFUND_AMOUNT:.2f} autonomous threshold.",
                    }
                )

                # Check supervisor response upon resume
                if not supervisor_decision or not supervisor_decision.get("approved"):
                    reviewer_note = (
                        supervisor_decision.get("note", "Denied by supervisor")
                        if supervisor_decision
                        else "Denied"
                    )
                    tool_messages.append(
                        ToolMessage(
                            content=json.dumps(
                                {
                                    "success": False,
                                    "status": "DENIED_BY_SUPERVISOR",
                                    "message": f"A supervisor reviewed your request for Order {order.id} and could not approve an immediate refund. Note: {reviewer_note}. A support ticket has been forwarded to tier-2 support.",
                                }
                            ),
                            name=tool_name,
                            tool_call_id=call_id,
                        )
                    )
                    state_updates["hitl_approved"] = False
                    state_updates["hitl_reviewer_note"] = reviewer_note
                    continue
                else:
                    state_updates["hitl_approved"] = True
                    state_updates["hitl_reviewer_note"] = supervisor_decision.get(
                        "note", "Approved by supervisor"
                    )

        try:
            # Execute tool asynchronously
            result = await target_tool.ainvoke(tool_args)
            content_str = json.dumps(result) if isinstance(result, (dict, list)) else str(result)
            tool_messages.append(
                ToolMessage(content=content_str, name=tool_name, tool_call_id=call_id)
            )
            consecutive_failures = 0

            # Dynamic Data Flow: Propagate tool result objects back into state
            if isinstance(result, dict):
                if tool_name == "check_order_status" and result.get("success"):
                    items = [
                        OrderItemDetail(
                            product_name=it.get("product_name", "Item"),
                            sku=it.get("sku", ""),
                            quantity=it.get("quantity", 1),
                            unit_price=it.get("unit_price", 0.0),
                            is_final_sale=it.get("is_final_sale", False),
                        )
                        for it in result.get("items", [])
                    ]
                    state_updates["active_order"] = OrderReference(
                        order_id=result.get("order_id"),
                        fulfillment_status=result.get("fulfillment_status", "PROCESSING"),
                        total_amount=result.get("total_amount", 0.0),
                        currency=result.get("currency", "USD"),
                        tracking_number=result.get("tracking_number"),
                        carrier_name=result.get("carrier_name"),
                        last_checkpoint=result.get("last_checkpoint"),
                        estimated_delivery=result.get("estimated_delivery"),
                        delivered_at=result.get("delivered_at"),
                        items=items,
                    )
                elif tool_name == "verify_refund_eligibility":
                    state_updates["refund_eligibility"] = RefundEligibility(
                        order_id=result.get("order_id", ""),
                        eligible=result.get("eligible", False),
                        reason=result.get("reason", ""),
                        max_refundable_amount=result.get("max_refundable_amount", 0.0),
                        requires_return_shipment=result.get("requires_return_shipment", False),
                        requires_human_approval=result.get("requires_human_approval", False),
                        approval_threshold=result.get("approval_threshold", 50.0),
                    )
                elif tool_name == "execute_refund" and result.get("success"):
                    state_updates["refund_executed"] = True
                    state_updates["refund_transaction_id"] = result.get("refund_id")
        except Exception as e:
            consecutive_failures += 1
            tool_messages.append(
                ToolMessage(
                    content=json.dumps({"error": f"Tool execution failed: {str(e)}"}),
                    name=tool_name,
                    tool_call_id=call_id,
                )
            )

    return {
        "messages": tool_messages,
        "consecutive_tool_failures": consecutive_failures,
        **state_updates,
    }
