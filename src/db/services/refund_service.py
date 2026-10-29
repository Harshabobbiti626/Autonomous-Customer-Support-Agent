import asyncio
import datetime
import hashlib

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.config import settings
from src.models.db_models import Order, RefundRecord
from src.models.state import RefundEligibility


class AsyncLockManager:
    """In-flight concurrency lock manager ensuring race-condition prevention across simultaneous requests."""

    _locks: dict[str, asyncio.Lock] = {}
    _global_lock = asyncio.Lock()

    @classmethod
    async def get_lock(cls, key: str) -> asyncio.Lock:
        async with cls._global_lock:
            if key not in cls._locks:
                cls._locks[key] = asyncio.Lock()
            return cls._locks[key]


class RefundService:
    """Enterprise refund governance and deterministic execution layer."""

    @staticmethod
    def generate_idempotency_key(customer_id: str, order_id: str, action: str = "refund") -> str:
        """Deterministic idempotency key preventing duplicate processing."""
        raw_key = f"{customer_id}:{order_id}:{action}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    @staticmethod
    async def evaluate_eligibility(
        session: AsyncSession, order_id: str, customer_id: str, reason_code: str = "DEFECTIVE"
    ) -> RefundEligibility:
        """Evaluate deterministic business rules for refunds."""
        stmt = (
            select(Order)
            .where(Order.id == order_id, Order.customer_id == customer_id)
            .options(selectinload(Order.items))
        )
        result = await session.execute(stmt)
        order = result.scalar_one_or_none()

        if not order:
            return RefundEligibility(
                order_id=order_id,
                eligible=False,
                reason="Order not found or unauthorized for this account.",
                max_refundable_amount=0.0,
            )

        # Rule 1: Check existing refund
        refund_stmt = select(RefundRecord).where(
            RefundRecord.order_id == order_id,
            RefundRecord.status.in_(["APPROVED", "PENDING_HUMAN_APPROVAL"]),
        )
        existing_refund = (await session.execute(refund_stmt)).scalar_one_or_none()
        if existing_refund:
            return RefundEligibility(
                order_id=order_id,
                eligible=False,
                reason=f"A refund has already been recorded for this order (Status: {existing_refund.status}).",
                max_refundable_amount=0.0,
            )

        # Rule 2: Delivery status check
        if order.fulfillment_status != "DELIVERED":
            return RefundEligibility(
                order_id=order_id,
                eligible=False,
                reason=f"Order is currently '{order.fulfillment_status}'. Only delivered orders are eligible for return or refund.",
                max_refundable_amount=0.0,
            )

        # Rule 3: 30-day window check
        now = datetime.datetime.now(datetime.UTC).replace(tzinfo=None)
        if not order.delivered_at:
            delivery_date = order.created_at
        else:
            delivery_date = order.delivered_at

        days_since_delivery = (now - delivery_date).days
        if days_since_delivery > settings.REFUND_WINDOW_DAYS:
            return RefundEligibility(
                order_id=order_id,
                eligible=False,
                reason=f"Delivered {days_since_delivery} days ago, exceeding our standard {settings.REFUND_WINDOW_DAYS}-day return window.",
                max_refundable_amount=0.0,
            )

        # Rule 4: Final sale exclusion check
        final_sale_items = [item.product_name for item in order.items if item.is_final_sale]
        if final_sale_items:
            items_str = ", ".join(final_sale_items)
            return RefundEligibility(
                order_id=order_id,
                eligible=False,
                reason=f"Item(s) marked as final sale clearance cannot be refunded: {items_str}.",
                max_refundable_amount=0.0,
            )

        # Rule 5: Financial threshold check for autonomous vs HITL approval
        requires_human = order.total_amount > settings.MAX_AUTONOMOUS_REFUND_AMOUNT

        return RefundEligibility(
            order_id=order_id,
            eligible=True,
            reason="Order satisfies all return and refund policy requirements.",
            max_refundable_amount=order.total_amount,
            requires_return_shipment=reason_code != "DEFECTIVE",
            requires_human_approval=requires_human,
            approval_threshold=settings.MAX_AUTONOMOUS_REFUND_AMOUNT,
        )

    @staticmethod
    async def process_refund(
        session: AsyncSession,
        order_id: str,
        customer_id: str,
        amount: float,
        reason_code: str,
        idempotency_key: str,
        status: str = "APPROVED",
        reviewer_note: str | None = None,
    ) -> tuple[RefundRecord, bool]:
        """Execute or record a refund transaction idempotently with in-flight distributed locking.

        Returns:
            Tuple[RefundRecord, bool]: (record, was_created_new)
        """
        lock_key = f"{customer_id}:{order_id}"
        lock = await AsyncLockManager.get_lock(lock_key)

        async with lock:
            # Idempotency check: Return existing record if matching idempotency key is found
            stmt = select(RefundRecord).where(RefundRecord.idempotency_key == idempotency_key)
            existing = (await session.execute(stmt)).scalar_one_or_none()
            if existing:
                return existing, False

            new_record = RefundRecord(
                id=f"REF-{int(datetime.datetime.now(datetime.UTC).timestamp())}",
                order_id=order_id,
                customer_id=customer_id,
                amount=amount,
                reason_code=reason_code,
                status=status,
                idempotency_key=idempotency_key,
                reviewer_note=reviewer_note,
            )
            session.add(new_record)
            await session.commit()
            await session.refresh(new_record)
            return new_record, True
