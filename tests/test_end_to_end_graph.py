import datetime
import uuid

import pytest
import pytest_asyncio
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.agent.graph import create_support_graph
from src.models.db_models import Base, Customer, Order, OrderItem
from src.models.state import CustomerContext


@pytest_asyncio.fixture
async def e2e_db():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    yield session_factory
    await engine.dispose()


@pytest.mark.asyncio
async def test_order_status_resolution(e2e_db):
    """Test dynamic autonomous resolution of order status inquiry with unique IDs and carrier metadata."""
    async with e2e_db() as session:
        now = datetime.datetime.utcnow()
        dyn_cust_id = f"cust_{uuid.uuid4().hex[:8]}"
        dyn_order_id = f"SHIP-{uuid.uuid4().hex[:6].upper()}"
        dyn_carrier = "DHL Express"
        dyn_tracking = f"DHL-{uuid.uuid4().hex[:10].upper()}"
        dyn_checkpoint = "Out for Delivery - Local Facility"

        cust = Customer(
            id=dyn_cust_id, email=f"{dyn_cust_id}@example.com", full_name="Dynamic Customer A"
        )
        ord1 = Order(
            id=dyn_order_id,
            customer_id=dyn_cust_id,
            fulfillment_status="SHIPPED",
            total_amount=68.50,
            carrier_name=dyn_carrier,
            tracking_number=dyn_tracking,
            last_checkpoint=dyn_checkpoint,
            created_at=now - datetime.timedelta(days=2),
        )
        item1 = OrderItem(
            order_id=dyn_order_id,
            product_name="Wireless Mechanical Keyboard",
            sku="TECH-KBD-01",
            unit_price=68.50,
        )
        session.add_all([cust, ord1, item1])
        await session.commit()

        graph = create_support_graph(checkpointer=MemorySaver())
        customer_ctx = CustomerContext(
            user_id=dyn_cust_id, email=f"{dyn_cust_id}@example.com", full_name="Dynamic Customer A"
        )
        config = {
            "configurable": {"thread_id": f"thread_{uuid.uuid4().hex[:8]}", "db_session": session}
        }
        initial_input = {
            "messages": [HumanMessage(content=f"Where is my package {dyn_order_id}?")],
            "customer": customer_ctx,
        }

        final_state = await graph.ainvoke(initial_input, config=config)
        messages = final_state["messages"]
        last_reply = messages[-1].content

        # Verify dynamic data flowed seamlessly through SQL -> Tools -> LLM -> Output
        assert dyn_order_id in last_reply
        assert dyn_carrier in last_reply or dyn_tracking in last_reply or "SHIPPED" in last_reply
        assert final_state.get("escalated") is False
        assert final_state.get("active_order") is not None
        assert final_state["active_order"].order_id == dyn_order_id
        assert final_state["active_order"].tracking_number == dyn_tracking


@pytest.mark.asyncio
async def test_dynamic_single_order_implicit_resolution(e2e_db):
    """Test that customer asking 'Where is my package?' without specifying an ID resolves their single order implicitly."""
    async with e2e_db() as session:
        now = datetime.datetime.utcnow()
        dyn_cust_id = f"cust_{uuid.uuid4().hex[:8]}"
        dyn_order_id = f"PKG-{uuid.uuid4().hex[:6].upper()}"
        dyn_tracking = f"UPS-{uuid.uuid4().hex[:8].upper()}"

        cust = Customer(
            id=dyn_cust_id, email=f"{dyn_cust_id}@example.com", full_name="Single Order Customer"
        )
        ord1 = Order(
            id=dyn_order_id,
            customer_id=dyn_cust_id,
            fulfillment_status="SHIPPED",
            total_amount=45.0,
            carrier_name="UPS",
            tracking_number=dyn_tracking,
            last_checkpoint="In Transit - Main Hub",
            created_at=now - datetime.timedelta(days=1),
        )
        session.add_all([cust, ord1])
        await session.commit()

        graph = create_support_graph(checkpointer=MemorySaver())
        customer_ctx = CustomerContext(
            user_id=dyn_cust_id,
            email=f"{dyn_cust_id}@example.com",
            full_name="Single Order Customer",
        )
        config = {
            "configurable": {
                "thread_id": f"thread_single_{uuid.uuid4().hex[:8]}",
                "db_session": session,
            }
        }
        # Customer does NOT type the order ID - system dynamically resolves it
        initial_input = {
            "messages": [HumanMessage(content="Where is my package?")],
            "customer": customer_ctx,
        }

        final_state = await graph.ainvoke(initial_input, config=config)
        last_reply = final_state["messages"][-1].content

        assert dyn_order_id in last_reply
        assert dyn_tracking in last_reply or "UPS" in last_reply
        assert final_state.get("active_order") is not None
        assert final_state["active_order"].order_id == dyn_order_id


@pytest.mark.asyncio
async def test_multi_order_dynamic_disambiguation(e2e_db):
    """Test that customer with multiple orders asking vaguely is presented with dynamic candidate list."""
    async with e2e_db() as session:
        now = datetime.datetime.utcnow()
        dyn_cust_id = f"cust_{uuid.uuid4().hex[:8]}"
        ord_id_a = f"ORD-A-{uuid.uuid4().hex[:4].upper()}"
        ord_id_b = f"ORD-B-{uuid.uuid4().hex[:4].upper()}"

        cust = Customer(id=dyn_cust_id, email="multi@example.com", full_name="Multi-Order User")
        ord_a = Order(
            id=ord_id_a, customer_id=dyn_cust_id, fulfillment_status="SHIPPED", total_amount=40.0
        )
        ord_b = Order(
            id=ord_id_b,
            customer_id=dyn_cust_id,
            fulfillment_status="DELIVERED",
            total_amount=80.0,
            delivered_at=now,
        )
        session.add_all([cust, ord_a, ord_b])
        await session.commit()

        graph = create_support_graph(checkpointer=MemorySaver())
        customer_ctx = CustomerContext(
            user_id=dyn_cust_id, email="multi@example.com", full_name="Multi-Order User"
        )
        config = {
            "configurable": {
                "thread_id": f"thread_multi_{uuid.uuid4().hex[:8]}",
                "db_session": session,
            }
        }
        initial_input = {
            "messages": [HumanMessage(content="I want to check my order status")],
            "customer": customer_ctx,
        }

        final_state = await graph.ainvoke(initial_input, config=config)
        last_reply = final_state["messages"][-1].content

        # Disambiguation list should contain both dynamic IDs from the database
        assert ord_id_a in last_reply
        assert ord_id_b in last_reply
        assert "multiple recent orders" in last_reply.lower()


@pytest.mark.asyncio
async def test_autonomous_refund_under_threshold(e2e_db):
    """Test that refund <= $50.00 executes autonomously and updates state with transaction ID."""
    async with e2e_db() as session:
        now = datetime.datetime.utcnow()
        dyn_cust_id = f"cust_{uuid.uuid4().hex[:8]}"
        dyn_order_id = f"RET-{uuid.uuid4().hex[:6].upper()}"

        cust = Customer(id=dyn_cust_id, email="auto@example.com", full_name="Auto Refund User")
        ord_delivered = Order(
            id=dyn_order_id,
            customer_id=dyn_cust_id,
            fulfillment_status="DELIVERED",
            total_amount=38.50,
            delivered_at=now - datetime.timedelta(days=4),
        )
        item = OrderItem(
            order_id=dyn_order_id,
            product_name="Ergonomic Mouse",
            sku="MSE-01",
            unit_price=38.50,
            is_final_sale=False,
        )
        session.add_all([cust, ord_delivered, item])
        await session.commit()

        graph = create_support_graph(checkpointer=MemorySaver())
        customer_ctx = CustomerContext(
            user_id=dyn_cust_id, email="auto@example.com", full_name="Auto Refund User"
        )
        config = {
            "configurable": {
                "thread_id": f"thread_auto_{uuid.uuid4().hex[:8]}",
                "db_session": session,
            }
        }
        initial_input = {
            "messages": [HumanMessage(content=f"Please process refund for order {dyn_order_id}")],
            "customer": customer_ctx,
        }

        final_state = await graph.ainvoke(initial_input, config=config)
        assert final_state.get("refund_executed") is True
        assert final_state.get("refund_transaction_id") is not None
        assert "successfully processed" in final_state["messages"][-1].content.lower()


@pytest.mark.asyncio
async def test_human_escalation_trigger(e2e_db):
    """Test that customer requesting human is escalated with dynamic handoff summary."""
    async with e2e_db() as session:
        dyn_cust_id = f"cust_{uuid.uuid4().hex[:8]}"
        cust = Customer(id=dyn_cust_id, email="frustrated@example.com", full_name="Frustrated User")
        session.add(cust)
        await session.commit()

        graph = create_support_graph(checkpointer=MemorySaver())
        customer_ctx = CustomerContext(
            user_id=dyn_cust_id, email="frustrated@example.com", full_name="Frustrated User"
        )
        config = {
            "configurable": {
                "thread_id": f"thread_esc_{uuid.uuid4().hex[:8]}",
                "db_session": session,
            }
        }
        initial_input = {
            "messages": [
                HumanMessage(content="Let me talk to a human agent right now! This is ridiculous.")
            ],
            "customer": customer_ctx,
        }

        final_state = await graph.ainvoke(initial_input, config=config)
        assert final_state.get("escalated") is True
        assert final_state.get("handoff_summary") is not None
        assert final_state["handoff_summary"].customer_name == "Frustrated User"
        assert "connected you with our human support team" in final_state["messages"][-1].content


@pytest.mark.asyncio
async def test_hitl_interrupt_and_resume_approval(e2e_db):
    """Test that high-value refund (> $50.00) triggers interrupt and resumes dynamically on supervisor approval."""
    async with e2e_db() as session:
        now = datetime.datetime.utcnow()
        dyn_cust_id = f"cust_{uuid.uuid4().hex[:8]}"
        dyn_order_id = f"HIGH-{uuid.uuid4().hex[:6].upper()}"
        refund_amount = 185.00

        cust = Customer(id=dyn_cust_id, email="hitl@example.com", full_name="HITL Customer")
        ord_high = Order(
            id=dyn_order_id,
            customer_id=dyn_cust_id,
            fulfillment_status="DELIVERED",
            total_amount=refund_amount,
            delivered_at=now - datetime.timedelta(days=2),
        )
        item_high = OrderItem(
            order_id=dyn_order_id,
            product_name="Studio Monitor Speaker",
            sku="SPK-01",
            unit_price=refund_amount,
            is_final_sale=False,
        )
        session.add_all([cust, ord_high, item_high])
        await session.commit()

        graph = create_support_graph(checkpointer=MemorySaver())
        customer_ctx = CustomerContext(
            user_id=dyn_cust_id, email="hitl@example.com", full_name="HITL Customer"
        )
        config = {
            "configurable": {
                "thread_id": f"thread_hitl_{uuid.uuid4().hex[:8]}",
                "db_session": session,
            }
        }
        initial_input = {
            "messages": [HumanMessage(content=f"Please process refund for order {dyn_order_id}")],
            "customer": customer_ctx,
        }

    # Step 1: LangGraph pauses on interrupt for high-value refunds
    state_after_call = await graph.ainvoke(initial_input, config=config)
    assert "__interrupt__" in state_after_call and len(state_after_call["__interrupt__"]) > 0

    interrupt_val = state_after_call["__interrupt__"][0].value
    assert interrupt_val["action"] == "APPROVE_HIGH_VALUE_REFUND"
    assert interrupt_val["order_id"] == dyn_order_id
    assert interrupt_val["amount"] == refund_amount
    assert interrupt_val["customer_name"] == "HITL Customer"

    # Step 2: Resume with supervisor approval
    resumed_state = await graph.ainvoke(
        Command(resume={"approved": True, "note": "Approved by supervisor John"}), config=config
    )

    assert resumed_state.get("hitl_approved") is True
    assert resumed_state.get("refund_executed") is True
    assert resumed_state.get("refund_transaction_id") is not None
    last_msg = resumed_state["messages"][-1].content
    assert (
        "successfully processed" in last_msg.lower()
        or "submitted and escalated" in last_msg.lower()
    )
