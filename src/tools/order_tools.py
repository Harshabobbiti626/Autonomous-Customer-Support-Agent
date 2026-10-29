from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.db.services.audit_service import AuditService
from src.db.services.order_service import OrderService


class CheckOrderStatusSchema(BaseModel):
    order_id: str = Field(description="The order ID, e.g., 'ORD-101'. Must belong to the customer.")


def create_order_tools(session, customer_id: str):
    """Factory creating order-related tools strictly scoped to the authenticated customer."""

    @tool(args_schema=CheckOrderStatusSchema)
    async def check_order_status(order_id: str) -> dict:
        """Fetch tracking details and real-time shipment status for an order owned by the customer."""
        cleaned_order_id = order_id.strip().upper()
        order = await OrderService.get_order_for_customer(
            session=session, order_id=cleaned_order_id, customer_id=customer_id
        )

        if not order:
            await AuditService.log_event(
                session=session,
                customer_id=customer_id,
                action="CHECK_ORDER_STATUS_FAILED_OR_UNAUTHORIZED",
                tool_name="check_order_status",
                input_payload={"order_id": cleaned_order_id},
                output_payload={"error": "Not found or IDOR attempt blocked"},
            )
            return {
                "success": False,
                "message": f"Order '{cleaned_order_id}' was not found in your account records. Please verify the order number.",
            }

        items_summary = [
            {
                "product_name": item.product_name,
                "quantity": item.quantity,
                "unit_price": item.unit_price,
            }
            for item in order.items
        ]

        result = {
            "success": True,
            "order_id": order.id,
            "fulfillment_status": order.fulfillment_status,
            "total_amount": order.total_amount,
            "currency": order.currency,
            "carrier_name": order.carrier_name,
            "tracking_number": order.tracking_number,
            "last_checkpoint": order.last_checkpoint,
            "estimated_delivery": str(order.estimated_delivery)
            if order.estimated_delivery
            else None,
            "delivered_at": str(order.delivered_at) if order.delivered_at else None,
            "items": items_summary,
        }

        await AuditService.log_event(
            session=session,
            customer_id=customer_id,
            action="CHECK_ORDER_STATUS_SUCCESS",
            tool_name="check_order_status",
            input_payload={"order_id": cleaned_order_id},
            output_payload=result,
        )

        return result

    return [check_order_status]
