from fastapi import APIRouter, Depends, HTTPException
from langchain_core.messages import AIMessage
from langgraph.types import Command
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.routes.chat import support_graph
from src.db.session import get_db
from src.models.api_schemas import ChatResponse, HITLDecisionRequest

router = APIRouter(prefix="/api/hitl", tags=["Human-in-the-Loop"])


@router.post("/decision", response_model=ChatResponse)
async def submit_supervisor_decision(
    payload: HITLDecisionRequest, db: AsyncSession = Depends(get_db)
):
    """Resume an interrupted LangGraph state with supervisor approval or denial."""
    config = {"configurable": {"thread_id": payload.session_id, "db_session": db}}

    # Verify that the session is currently waiting on an interrupt
    state_snapshot = support_graph.get_state(config)
    if not state_snapshot.next:
        raise HTTPException(
            status_code=400,
            detail=f"Session '{payload.session_id}' has no pending interrupts or is already completed.",
        )

    resume_payload = {"approved": payload.approved, "note": payload.reviewer_note}

    # Resume graph execution via LangGraph Command
    final_state = await support_graph.ainvoke(Command(resume=resume_payload), config=config)

    messages = final_state.get("messages", [])
    last_reply = "Workflow completed."
    for m in reversed(messages):
        if isinstance(m, AIMessage) and m.content:
            last_reply = m.content
            break

    return ChatResponse(
        session_id=payload.session_id,
        status="RESOLVED",
        reply=last_reply,
        escalated=final_state.get("escalated", False),
    )
