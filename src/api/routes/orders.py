from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.services.order_service import OrderService
from src.db.session import get_db
from src.models.api_schemas import OrderItemResponse, OrderSummaryResponse

router = APIRouter(prefix="/api/orders", tags=["Orders"])


@router.get("", response_model=list[OrderSummaryResponse])
async def list_orders(
    customer_id: str = Query(..., description="Customer ID to fetch orders for"),
    db: AsyncSession = Depends(get_db),
):
    """List recent orders owned by the customer."""
    orders = await OrderService.list_orders_for_customer(
        session=db, customer_id=customer_id, limit=10
    )

    response = []
    for ord_obj in orders:
        items = [
            OrderItemResponse(
                product_name=it.product_name,
                quantity=it.quantity,
                unit_price=it.unit_price,
                is_final_sale=it.is_final_sale,
            )
            for it in ord_obj.items
        ]
        response.append(
            OrderSummaryResponse(
                order_id=ord_obj.id,
                fulfillment_status=ord_obj.fulfillment_status,
                total_amount=ord_obj.total_amount,
                currency=ord_obj.currency,
                carrier_name=ord_obj.carrier_name,
                tracking_number=ord_obj.tracking_number,
                last_checkpoint=ord_obj.last_checkpoint,
                created_at=str(ord_obj.created_at),
                items=items,
            )
        )

    return response
