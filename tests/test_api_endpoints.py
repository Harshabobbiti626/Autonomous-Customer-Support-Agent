import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from src.api.app import app
from src.db.seed import seed_data
from src.db.session import init_db


@pytest_asyncio.fixture(autouse=True)
async def setup_api_test_db():
    await init_db()
    await seed_data()


@pytest.mark.asyncio
async def test_health_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "HEALTHY"
        assert "max_autonomous_refund_threshold" in data["governance"]


@pytest.mark.asyncio
async def test_api_chat_flow():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        payload = {
            "session_id": "api_test_session_1",
            "customer_id": "cust_001",
            "message": "Where is my order ORD-101?",
        }
        response = await client.post("/api/chat", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "RESOLVED"
        assert "ORD-101" in data["reply"]
        assert "SHIPPED" in data["reply"]


@pytest.mark.asyncio
async def test_api_orders_list():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/orders?customer_id=cust_001")
        assert response.status_code == 200
        orders = response.json()
        assert len(orders) >= 3
        order_ids = [o["order_id"] for o in orders]
        assert "ORD-101" in order_ids
