from typing import Annotated, Any, Literal

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class CustomerContext(BaseModel):
    """Authenticated customer profile injected securely from gateway token."""

    user_id: str
    tenant_id: str = "default_tenant"
    email: str
    full_name: str
    tier: Literal["standard", "vip", "flagged_fraud"] = "standard"


class OrderItemDetail(BaseModel):
    product_name: str
    sku: str
    quantity: int
    unit_price: float
    is_final_sale: bool = False


class OrderReference(BaseModel):
    """Active order entity under discussion in the conversation."""

    order_id: str
    fulfillment_status: Literal[
        "PROCESSING", "SHIPPED", "IN_TRANSIT", "DELIVERED", "CANCELLED", "RETURNED"
    ]
    total_amount: float
    currency: str = "USD"
    tracking_number: str | None = None
    carrier_name: str | None = None
    last_checkpoint: str | None = None
    estimated_delivery: str | None = None
    delivered_at: str | None = None
    items: list[OrderItemDetail] = Field(default_factory=list)


class RefundEligibility(BaseModel):
    """Evaluation result for refund eligibility."""

    order_id: str
    eligible: bool
    reason: str
    max_refundable_amount: float = 0.0
    requires_return_shipment: bool = False
    requires_human_approval: bool = False
    approval_threshold: float = 50.0


class HandoffSummary(BaseModel):
    """Structured summary created for the human live support representative."""

    customer_name: str
    customer_tier: str
    primary_issue: str
    attempted_actions: list[str]
    escalation_reason: str
    recommended_action: str


class AgentState(TypedDict):
    """Production LangGraph conversational state."""

    # Conversational messages with LangGraph's standard message reducer
    messages: Annotated[list[BaseMessage], add_messages]

    # Authenticated Customer Identity (Immutable by LLM)
    customer: CustomerContext

    # Active Entity Context
    active_order: OrderReference | None
    candidate_orders: list[dict[str, Any]]
    disambiguation_required: bool

    # Intent and Flow Control
    intent: (
        Literal[
            "order_tracking", "refund_request", "general_policy", "human_escalation", "general_chat"
        ]
        | None
    )

    # Refund State Tracking
    refund_eligibility: RefundEligibility | None
    refund_executed: bool
    refund_transaction_id: str | None
    idempotency_token: str | None

    # Safety, Risk & Health Metrics
    sentiment_score: float
    consecutive_tool_failures: int
    escalated: bool
    escalation_reason: str | None
    handoff_summary: HandoffSummary | None

    # Human In The Loop approval status (resumed after interrupt)
    hitl_approved: bool | None
    hitl_reviewer_note: str | None
