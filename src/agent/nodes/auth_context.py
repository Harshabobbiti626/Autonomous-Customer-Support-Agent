from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from src.db.services.order_service import OrderService
from src.guardrails.output_checker import OutputChecker
from src.guardrails.pii_sanitizer import PIISanitizer
from src.guardrails.prompt_shield import PromptShield
from src.models.state import AgentState, OrderItemDetail, OrderReference


async def auth_context_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """Pre-flight node: Sanitizes input, verifies prompt safety, resolves customer orders."""
    session = config.get("configurable", {}).get("db_session")
    customer = state["customer"]

    messages = state.get("messages", [])
    last_msg = messages[-1] if messages else None

    # Calculate customer sentiment on latest user message
    user_text = last_msg.content if isinstance(last_msg, HumanMessage) else ""
    sentiment = OutputChecker.calculate_sentiment(user_text)

    # Sanitize PII (credit cards, SSNs, phone numbers)
    sanitized_text = PIISanitizer.sanitize(user_text)
    if isinstance(last_msg, HumanMessage) and sanitized_text != user_text:
        last_msg.content = sanitized_text

    # Prompt Shield / Injection detection
    is_safe, violation = PromptShield.evaluate(user_text)
    if not is_safe:
        return {
            "sentiment_score": -1.0,
            "escalated": True,
            "escalation_reason": f"Security violation detected: {violation}",
        }

    # Fetch customer orders from database for context resolution
    candidate_orders = []
    active_order = state.get("active_order")

    if session:
        orders = await OrderService.list_orders_for_customer(
            session=session, customer_id=customer.user_id, limit=5
        )
        for ord_obj in orders:
            candidate_orders.append(
                {
                    "order_id": ord_obj.id,
                    "status": ord_obj.fulfillment_status,
                    "amount": ord_obj.total_amount,
                    "created_at": str(ord_obj.created_at),
                }
            )

        # Dynamically check if the customer explicitly referenced any of their orders
        user_specified_order = None
        for ord_obj in orders:
            clean_id = ord_obj.id.lower()
            if clean_id in user_text.lower() or clean_id.replace("-", "") in user_text.lower():
                user_specified_order = ord_obj
                break

        target_obj = user_specified_order or (
            orders[0] if len(orders) == 1 and not active_order else None
        )

        if target_obj:
            items = [
                OrderItemDetail(
                    product_name=it.product_name,
                    sku=it.sku,
                    quantity=it.quantity,
                    unit_price=it.unit_price,
                    is_final_sale=it.is_final_sale,
                )
                for it in target_obj.items
            ]
            active_order = OrderReference(
                order_id=target_obj.id,
                fulfillment_status=target_obj.fulfillment_status,
                total_amount=target_obj.total_amount,
                currency=target_obj.currency,
                tracking_number=target_obj.tracking_number,
                carrier_name=target_obj.carrier_name,
                last_checkpoint=target_obj.last_checkpoint,
                estimated_delivery=str(target_obj.estimated_delivery)
                if target_obj.estimated_delivery
                else None,
                delivered_at=str(target_obj.delivered_at) if target_obj.delivered_at else None,
                items=items,
            )

    return {
        "sentiment_score": sentiment,
        "candidate_orders": candidate_orders,
        "active_order": active_order,
        "disambiguation_required": len(candidate_orders) > 1 and not active_order,
        "escalated": False,
    }
