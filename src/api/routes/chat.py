from fastapi import APIRouter, Depends, HTTPException
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.errors import GraphInterrupt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.agent.graph import create_support_graph
from src.db.session import get_db
from src.models.api_schemas import ChatRequest, ChatResponse
from src.models.db_models import Customer
from src.models.state import CustomerContext

router = APIRouter(prefix="/api/chat", tags=["Chat"])

# Shared compiled graph instance with persistent in-memory checkpointer
support_graph = create_support_graph()


@router.post("", response_model=ChatResponse)
async def process_chat_message(payload: ChatRequest, db: AsyncSession = Depends(get_db)):
    """Process customer message through stateful LangGraph agent."""
    # 1. Authenticate & fetch customer record
    stmt = select(Customer).where(Customer.id == payload.customer_id)
    cust = (await db.execute(stmt)).scalar_one_or_none()
    if not cust:
        raise HTTPException(status_code=404, detail=f"Customer '{payload.customer_id}' not found.")

    customer_ctx = CustomerContext(
        user_id=cust.id, email=cust.email, full_name=cust.full_name, tier=cust.tier
    )

    config = {"configurable": {"thread_id": payload.session_id, "db_session": db}}

    initial_input = {"messages": [HumanMessage(content=payload.message)], "customer": customer_ctx}

    try:
        final_state = await support_graph.ainvoke(initial_input, config=config)

        # Check if graph paused on LangGraph interrupt()
        if "__interrupt__" in final_state and final_state["__interrupt__"]:
            interrupt_val = final_state["__interrupt__"][0].value
            return ChatResponse(
                session_id=payload.session_id,
                status="NEEDS_SUPERVISOR_APPROVAL",
                reply=(
                    "Your refund request exceeds our autonomous threshold ($50.00). "
                    "I have paused execution and notified a supervisor for authorization."
                ),
                escalated=False,
                interrupt_details=interrupt_val,
            )

        # Extract last assistant message
        messages = final_state.get("messages", [])
        last_reply = "I'm here to assist you."
        for m in reversed(messages):
            if isinstance(m, AIMessage) and m.content:
                last_reply = m.content
                break

        escalated = final_state.get("escalated", False)
        status = "ESCALATED" if escalated else "RESOLVED"
        handoff = final_state.get("handoff_summary")

        return ChatResponse(
            session_id=payload.session_id,
            status=status,
            reply=last_reply,
            escalated=escalated,
            handoff_summary=handoff.model_dump() if handoff else None,
        )

    except GraphInterrupt:
        current_state = support_graph.get_state(config)
        interrupt_val = None
        if current_state.tasks and current_state.tasks[0].interrupts:
            interrupt_val = current_state.tasks[0].interrupts[0].value

        return ChatResponse(
            session_id=payload.session_id,
            status="NEEDS_SUPERVISOR_APPROVAL",
            reply=(
                "Your refund request exceeds our autonomous threshold ($50.00). "
                "I have paused execution and notified a supervisor for authorization."
            ),
            escalated=False,
            interrupt_details=interrupt_val,
        )
