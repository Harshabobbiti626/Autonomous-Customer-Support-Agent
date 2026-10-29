import asyncio
import datetime

from sqlalchemy import select

from src.db.session import AsyncSessionLocal, init_db
from src.models.db_models import Customer, Order, OrderItem


async def seed_data():
    """Seed comprehensive customer and order data covering standard and edge cases."""
    await init_db()

    async with AsyncSessionLocal() as session:
        # Check if already seeded
        result = await session.execute(select(Customer).limit(1))
        if result.scalar_one_or_none():
            print("[INFO] Database already contains seed data.")
            return

        now = datetime.datetime.utcnow()

        # Customer 1: Alice Cooper (Standard customer with diverse orders)
        cust1 = Customer(
            id="cust_001", email="alice@example.com", full_name="Alice Cooper", tier="standard"
        )

        # Order 1: In Transit (Shipped)
        order1 = Order(
            id="ORD-101",
            customer_id="cust_001",
            fulfillment_status="SHIPPED",
            total_amount=64.99,
            currency="USD",
            carrier_name="FedEx",
            tracking_number="FX-98234710",
            last_checkpoint="In Transit - Memphis Distribution Center",
            estimated_delivery=now + datetime.timedelta(days=2),
            delivered_at=None,
            created_at=now - datetime.timedelta(days=3),
        )
        item1_1 = OrderItem(
            order_id="ORD-101",
            product_name="Wireless Noise-Canceling Earbuds",
            sku="TECH-EAR-01",
            quantity=1,
            unit_price=64.99,
            is_final_sale=False,
        )
        order1.items.append(item1_1)

        # Order 2: Delivered recently, eligible for autonomous refund ($42.50 <= $50 threshold)
        order2 = Order(
            id="ORD-102",
            customer_id="cust_001",
            fulfillment_status="DELIVERED",
            total_amount=42.50,
            currency="USD",
            carrier_name="UPS",
            tracking_number="1Z9999999999999999",
            last_checkpoint="Delivered - Front Porch",
            estimated_delivery=now - datetime.timedelta(days=5),
            delivered_at=now - datetime.timedelta(days=5),
            created_at=now - datetime.timedelta(days=8),
        )
        item2_1 = OrderItem(
            order_id="ORD-102",
            product_name="Ergonomic Mouse Pad",
            sku="ACC-MSE-09",
            quantity=1,
            unit_price=17.50,
            is_final_sale=False,
        )
        item2_2 = OrderItem(
            order_id="ORD-102",
            product_name="Braided USB-C Cable (2-pack)",
            sku="ACC-CBL-02",
            quantity=1,
            unit_price=25.00,
            is_final_sale=False,
        )
        order2.items.extend([item2_1, item2_2])

        # Order 3: Delivered > 30 days ago (Ineligible for refund due to policy window)
        order3 = Order(
            id="ORD-103",
            customer_id="cust_001",
            fulfillment_status="DELIVERED",
            total_amount=89.00,
            currency="USD",
            carrier_name="USPS",
            tracking_number="9400100000000000000000",
            last_checkpoint="Delivered - In/At Mailbox",
            estimated_delivery=now - datetime.timedelta(days=45),
            delivered_at=now - datetime.timedelta(days=45),
            created_at=now - datetime.timedelta(days=50),
        )
        item3_1 = OrderItem(
            order_id="ORD-103",
            product_name="Mechanical Gaming Keyboard",
            sku="TECH-KBD-88",
            quantity=1,
            unit_price=89.00,
            is_final_sale=False,
        )
        order3.items.append(item3_1)

        # Order 4: Delivered 10 days ago, total $149.99 (Eligible but > $50 -> Requires Human-In-The-Loop Approval)
        order4 = Order(
            id="ORD-104",
            customer_id="cust_001",
            fulfillment_status="DELIVERED",
            total_amount=149.99,
            currency="USD",
            carrier_name="FedEx",
            tracking_number="FX-77182944",
            last_checkpoint="Delivered - Parcel Locker",
            estimated_delivery=now - datetime.timedelta(days=10),
            delivered_at=now - datetime.timedelta(days=10),
            created_at=now - datetime.timedelta(days=14),
        )
        item4_1 = OrderItem(
            order_id="ORD-104",
            product_name="4K Ultra-Wide Monitor Stand",
            sku="ACC-MON-04",
            quantity=1,
            unit_price=149.99,
            is_final_sale=False,
        )
        order4.items.append(item4_1)

        # Customer 2: Bob Vance (Contains final sale exclusion item)
        cust2 = Customer(
            id="cust_002", email="bob@example.com", full_name="Bob Vance", tier="standard"
        )
        order5 = Order(
            id="ORD-201",
            customer_id="cust_002",
            fulfillment_status="DELIVERED",
            total_amount=35.00,
            currency="USD",
            carrier_name="DHL",
            tracking_number="DHL-44019283",
            last_checkpoint="Delivered - Reception",
            estimated_delivery=now - datetime.timedelta(days=4),
            delivered_at=now - datetime.timedelta(days=4),
            created_at=now - datetime.timedelta(days=7),
        )
        item5_1 = OrderItem(
            order_id="ORD-201",
            product_name="Clearance Warehouse T-Shirt",
            sku="APP-TSH-CLR",
            quantity=1,
            unit_price=35.00,
            is_final_sale=True,
        )
        order5.items.append(item5_1)

        session.add_all([cust1, order1, order2, order3, order4, cust2, order5])
        await session.commit()
        print("[SUCCESS] Seeded customers and orders successfully.")


if __name__ == "__main__":
    asyncio.run(seed_data())
