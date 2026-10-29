import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.db.services.order_service import OrderService
from src.models.db_models import Base, Customer, Order
from src.tools.order_tools import create_order_tools


@pytest_asyncio.fixture
async def test_db():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

    async with session_factory() as session:
        # Seed two distinct customers
        cust_a = Customer(id="cust_A", email="a@example.com", full_name="User A")
        cust_b = Customer(id="cust_B", email="b@example.com", full_name="User B")

        # Order belonging to User B
        order_b = Order(
            id="ORD-USER-B",
            customer_id="cust_B",
            fulfillment_status="DELIVERED",
            total_amount=100.0,
            tracking_number="TRACK-B",
        )
        session.add_all([cust_a, cust_b, order_b])
        await session.commit()

    yield session_factory
    await engine.dispose()


@pytest.mark.asyncio
async def test_idor_prevention_at_service_layer(test_db):
    """Verify that User A cannot retrieve User B's order under any circumstance."""
    async with test_db() as session:
        # User B queries their own order -> Success
        order = await OrderService.get_order_for_customer(session, "ORD-USER-B", "cust_B")
        assert order is not None
        assert order.id == "ORD-USER-B"

        # User A attempts to query User B's order -> Strictly None (IDOR BLOCKED)
        unauthorized = await OrderService.get_order_for_customer(session, "ORD-USER-B", "cust_A")
        assert unauthorized is None


@pytest.mark.asyncio
async def test_idor_prevention_at_tool_layer(test_db):
    """Verify that tool execution for User A returns error message when targeting User B's order."""
    async with test_db() as session:
        tools_for_user_a = create_order_tools(session, customer_id="cust_A")
        check_status_tool = tools_for_user_a[0]

        result = await check_status_tool.ainvoke({"order_id": "ORD-USER-B"})
        assert result["success"] is False
        assert "not found in your account records" in result["message"]
