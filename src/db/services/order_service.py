from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.models.db_models import Order


class OrderService:
    """Hardened data access layer for customer orders with strict tenant & IDOR isolation."""

    @staticmethod
    async def get_order_for_customer(
        session: AsyncSession, order_id: str, customer_id: str
    ) -> Order | None:
        """Fetch order strictly verifying ownership by the authenticated customer.

        CRITICAL FOR IDOR PREVENTION:
        The WHERE clause binds customer_id programmatically from session context.
        An attacker querying an order ID belonging to another user receives None.
        """
        stmt = (
            select(Order)
            .where(Order.id == order_id, Order.customer_id == customer_id)
            .options(selectinload(Order.items))
        )
        result = await session.execute(stmt)
        return result.scalar_one_or_none()

    @staticmethod
    async def list_orders_for_customer(
        session: AsyncSession, customer_id: str, limit: int = 5
    ) -> list[Order]:
        """Fetch active or recent orders for the customer to assist in disambiguation."""
        stmt = (
            select(Order)
            .where(Order.customer_id == customer_id)
            .order_by(Order.created_at.desc())
            .limit(limit)
            .options(selectinload(Order.items))
        )
        result = await session.execute(stmt)
        return list(result.scalars().all())
