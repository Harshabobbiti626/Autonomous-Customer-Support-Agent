import pytest
import pytest_asyncio
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.db.services.refund_service import RefundService
from src.models.db_models import Base, Customer, Order, RefundRecord


@pytest_asyncio.fixture
async def idempotency_test_db():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

    async with session_factory() as session:
        cust = Customer(id="cust_idem", email="idem@example.com", full_name="Idem User")
        ord_obj = Order(
            id="ORD-IDEM",
            customer_id="cust_idem",
            fulfillment_status="DELIVERED",
            total_amount=25.0,
        )
        session.add_all([cust, ord_obj])
        await session.commit()

    yield session_factory
    await engine.dispose()


@pytest.mark.asyncio
async def test_refund_execution_idempotency(idempotency_test_db):
    """Verify that multiple refund calls with the same key return the existing record without duplicate DB inserts."""
    async with idempotency_test_db() as session:
        key = RefundService.generate_idempotency_key("cust_idem", "ORD-IDEM")

        # First call: creates new refund record
        record1, was_created1 = await RefundService.process_refund(
            session=session,
            order_id="ORD-IDEM",
            customer_id="cust_idem",
            amount=25.0,
            reason_code="DEFECTIVE",
            idempotency_key=key,
            status="APPROVED",
        )
        assert was_created1 is True
        assert record1.status == "APPROVED"

        # Second call (network retry or duplicate click): returns identical record
        record2, was_created2 = await RefundService.process_refund(
            session=session,
            order_id="ORD-IDEM",
            customer_id="cust_idem",
            amount=25.0,
            reason_code="DEFECTIVE",
            idempotency_key=key,
            status="APPROVED",
        )
        assert was_created2 is False
        assert record2.id == record1.id

        # Verify DB strictly has 1 record
        count_stmt = select(func.count(RefundRecord.id)).where(RefundRecord.order_id == "ORD-IDEM")
        count = (await session.execute(count_stmt)).scalar()
        assert count == 1
