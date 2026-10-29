from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(
        description="Unique session/thread ID for conversation state persistence"
    )
    customer_id: str = Field(description="Authenticated customer ID")
    message: str = Field(description="Customer message content")


class ChatResponse(BaseModel):
    session_id: str
    status: str  # "RESOLVED", "ESCALATED", "NEEDS_SUPERVISOR_APPROVAL"
    reply: str
    escalated: bool = False
    interrupt_details: dict[str, Any] | None = None
    handoff_summary: dict[str, Any] | None = None


class HITLDecisionRequest(BaseModel):
    session_id: str = Field(description="Target session ID currently paused on breakpoint")
    customer_id: str = Field(description="Customer ID associated with the session")
    approved: bool = Field(description="Whether the supervisor approved or denied the action")
    reviewer_note: str | None = Field(default=None, description="Supervisor review note")


class OrderItemResponse(BaseModel):
    product_name: str
    quantity: int
    unit_price: float
    is_final_sale: bool


class OrderSummaryResponse(BaseModel):
    order_id: str
    fulfillment_status: str
    total_amount: float
    currency: str
    carrier_name: str | None
    tracking_number: str | None
    last_checkpoint: str | None
    created_at: str
    items: list[OrderItemResponse]
