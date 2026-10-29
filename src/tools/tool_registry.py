from langchain_core.tools import BaseTool

from src.tools.order_tools import create_order_tools
from src.tools.refund_tools import create_refund_tools


class ToolRegistry:
    """Provides customer-scoped tool collections capturing security context."""

    @staticmethod
    def get_tools_for_customer(session, customer_id: str) -> list[BaseTool]:
        order_tools = create_order_tools(session=session, customer_id=customer_id)
        refund_tools = create_refund_tools(session=session, customer_id=customer_id)
        return order_tools + refund_tools
