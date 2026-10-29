from typing import Literal

from langgraph.graph import END, START, StateGraph

from src.agent.checkpointer_factory import get_checkpointer
from src.agent.nodes.auth_context import auth_context_node
from src.agent.nodes.escalation import escalation_node
from src.agent.nodes.output_guard import output_guard_node
from src.agent.nodes.refund_governor import tool_execution_and_governor_node
from src.agent.nodes.router import route_decision, router_node
from src.agent.nodes.support_worker import support_worker_node
from src.models.state import AgentState


def should_continue_tools(state: AgentState) -> Literal["tool_governor", "output_guard"]:
    """Determines whether LLM output requires tool execution or can proceed to output guard."""
    messages = state.get("messages", [])
    if not messages:
        return "output_guard"

    last_msg = messages[-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tool_governor"
    return "output_guard"


def create_support_graph(checkpointer=None):
    """Factory compiling the production Autonomous Customer Support LangGraph state machine."""
    workflow = StateGraph(AgentState)

    # 1. Register Graph Nodes
    workflow.add_node("auth_context", auth_context_node)
    workflow.add_node("router", router_node)
    workflow.add_node("support_worker", support_worker_node)
    workflow.add_node("tool_governor", tool_execution_and_governor_node)
    workflow.add_node("escalation", escalation_node)
    workflow.add_node("output_guard", output_guard_node)

    # 2. Define Control Flow & Edges
    workflow.add_edge(START, "auth_context")
    workflow.add_edge("auth_context", "router")

    # Conditional routing after intent classification & security check
    workflow.add_conditional_edges(
        "router", route_decision, {"escalation": "escalation", "support_worker": "support_worker"}
    )

    # Conditional edge after LLM generation: tools vs direct answer
    workflow.add_conditional_edges(
        "support_worker",
        should_continue_tools,
        {"tool_governor": "tool_governor", "output_guard": "output_guard"},
    )

    # Tool output flows back to worker node for synthesis
    workflow.add_edge("tool_governor", "support_worker")

    # Final nodes terminate to END
    workflow.add_edge("output_guard", END)
    workflow.add_edge("escalation", END)

    # Checkpointer for conversation state persistence and Human-in-the-Loop interrupts
    cp = checkpointer if checkpointer is not None else get_checkpointer()
    return workflow.compile(checkpointer=cp)
