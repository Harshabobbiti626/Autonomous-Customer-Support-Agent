import datetime

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.db.services.refund_service import RefundService
from src.models.db_models import Base, Customer, Order, OrderItem


@pytest_asyncio.fixture
async def refund_test_db():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    now = datetime.datetime.utcnow()

    async with session_factory() as session:
        cust = Customer(id="cust_test", email="test@example.com", full_name="Test User")

        # 1. In-transit order
        ord_transit = Order(
            id="ORD-TRANSIT",
            customer_id="cust_test",
            fulfillment_status="SHIPPED",
            total_amount=40.0,
        )

        # 2. Expired window (> 30 days)
        ord_expired = Order(
            id="ORD-EXPIRED",
            customer_id="cust_test",
            fulfillment_status="DELIVERED",
            total_amount=40.0,
            delivered_at=now - datetime.timedelta(days=40),
        )

        # 3. Final sale item
        ord_final_sale = Order(
            id="ORD-FINAL-SALE",
            customer_id="cust_test",
            fulfillment_status="DELIVERED",
            total_amount=30.0,
            delivered_at=now - datetime.timedelta(days=5),
        )
        item_final = OrderItem(
            order_id="ORD-FINAL-SALE",
            product_name="Clearance Hat",
            sku="HAT-01",
            unit_price=30.0,
            is_final_sale=True,
        )
        ord_final_sale.items.append(item_final)

        # 4. Eligible Autonomous (delivered 5 days ago, $45.00 <= $50.00)
        ord_eligible_auto = Order(
            id="ORD-ELIGIBLE-AUTO",
            customer_id="cust_test",
            fulfillment_status="DELIVERED",
            total_amount=45.0,
            delivered_at=now - datetime.timedelta(days=5),
        )
        item_normal = OrderItem(
            order_id="ORD-ELIGIBLE-AUTO",
            product_name="Desk Lamp",
            sku="LMP-01",
            unit_price=45.0,
            is_final_sale=False,
        )
        ord_eligible_auto.items.append(item_normal)

        # 5. Eligible High-Value (delivered 5 days ago, $120.00 > $50.00 -> HITL)
        ord_eligible_hitl = Order(
            id="ORD-ELIGIBLE-HITL",
            customer_id="cust_test",
            fulfillment_status="DELIVERED",
            total_amount=120.0,
            delivered_at=now - datetime.timedelta(days=5),
        )
        item_high = OrderItem(
            order_id="ORD-ELIGIBLE-HITL",
            product_name="Office Chair",
            sku="CHR-01",
            unit_price=120.0,
            is_final_sale=False,
        )
        ord_eligible_hitl.items.append(item_high)

        session.add_all(
            [cust, ord_transit, ord_expired, ord_final_sale, ord_eligible_auto, ord_eligible_hitl]
        )
        await session.commit()

    yield session_factory
    await engine.dispose()


@pytest.mark.asyncio
async def test_refund_rejected_if_not_delivered(refund_test_db):
    async with refund_test_db() as session:
        eligibility = await RefundService.evaluate_eligibility(session, "ORD-TRANSIT", "cust_test")
        assert eligibility.eligible is False
        assert "Only delivered orders are eligible" in eligibility.reason


@pytest.mark.asyncio
async def test_refund_rejected_if_window_exceeded(refund_test_db):
    async with refund_test_db() as session:
        eligibility = await RefundService.evaluate_eligibility(session, "ORD-EXPIRED", "cust_test")
        assert eligibility.eligible is False
        assert "exceeding our standard 30-day return window" in eligibility.reason


@pytest.mark.asyncio
async def test_refund_rejected_for_final_sale(refund_test_db):
    async with refund_test_db() as session:
        eligibility = await RefundService.evaluate_eligibility(
            session, "ORD-FINAL-SALE", "cust_test"
        )
        assert eligibility.eligible is False
        assert "final sale clearance" in eligibility.reason


@pytest.mark.asyncio
async def test_refund_eligible_autonomous(refund_test_db):
    async with refund_test_db() as session:
        eligibility = await RefundService.evaluate_eligibility(
            session, "ORD-ELIGIBLE-AUTO", "cust_test"
        )
        assert eligibility.eligible is True
        assert eligibility.requires_human_approval is False
        assert eligibility.max_refundable_amount == 45.0


@pytest.mark.asyncio
async def test_refund_eligible_requires_human_approval(refund_test_db):
    async with refund_test_db() as session:
        eligibility = await RefundService.evaluate_eligibility(
            session, "ORD-ELIGIBLE-HITL", "cust_test"
        )
        assert eligibility.eligible is True
        assert eligibility.requires_human_approval is True
        assert eligibility.max_refundable_amount == 120.0
