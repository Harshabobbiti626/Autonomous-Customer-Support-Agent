import json
import re
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from src.config import settings


class MockSupportLLM:
    """Deterministic tool-calling mock LLM for testing, local CI/CD, and offline verification."""

    def __init__(self, tools: list[Any] | None = None):
        self.tools = tools or []
        self.tools_by_name = {t.name: t for t in self.tools}

    def bind_tools(self, tools: list[Any]):
        return MockSupportLLM(tools=tools)

    async def ainvoke(self, messages: list[BaseMessage]) -> AIMessage:
        """Async invocation simulating a frontier tool-calling LLM."""
        return self._generate_response(messages)

    def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        """Sync invocation simulating a frontier tool-calling LLM."""
        return self._generate_response(messages)

    def _generate_response(self, messages: list[BaseMessage]) -> AIMessage:
        # Check if the last message is a ToolMessage (tool execution completed)
        last_msg = messages[-1] if messages else None

        if isinstance(last_msg, ToolMessage):
            tool_content = last_msg.content
            try:
                data = json.loads(tool_content) if isinstance(tool_content, str) else tool_content
            except Exception:
                data = {"raw": tool_content}

            # Generate natural language response based on tool result
            if last_msg.name == "check_order_status":
                if data.get("success"):
                    status = data.get("fulfillment_status")
                    carrier = data.get("carrier_name", "Carrier")
                    tracking = data.get("tracking_number", "N/A")
                    checkpoint = data.get("last_checkpoint", "In Transit")
                    return AIMessage(
                        content=f"Your order **{data.get('order_id')}** is currently **{status}** via {carrier} (Tracking: `{tracking}`). Latest update: *{checkpoint}*."
                    )
                else:
                    return AIMessage(
                        content=data.get(
                            "message", "I was unable to locate that order under your account."
                        )
                    )

            elif last_msg.name == "verify_refund_eligibility":
                if data.get("eligible"):
                    amt = data.get("max_refundable_amount", 0.0)
                    requires_hitl = data.get("requires_human_approval", False)
                    if requires_hitl:
                        return AIMessage(
                            content=f"Good news! Order **{data.get('order_id')}** is eligible for a return/refund of **${amt:.2f}**. Because the amount exceeds our $50.00 threshold, would you like me to submit this for supervisor authorization?"
                        )
                    return AIMessage(
                        content=f"Good news! Order **{data.get('order_id')}** is eligible for an immediate refund of **${amt:.2f}**. Would you like me to process this refund now?"
                    )
                else:
                    return AIMessage(
                        content=f"I checked our return policy for order **{data.get('order_id')}**: unfortunately it is not eligible for a refund. Reason: {data.get('reason')}"
                    )

            elif last_msg.name == "execute_refund":
                return AIMessage(content=data.get("message", "Refund status updated."))

        # Extract last human message
        human_text = ""
        for m in reversed(messages):
            if isinstance(m, HumanMessage):
                human_text = m.content
                break

        # Dynamic entity extraction: find any order-like identifier
        order_patterns = [
            r"\b(?:order|ord|pkg|package|shipment|#)\s*[:#-]?\s*([A-Za-z0-9_-]{3,25})\b",
            r"\b([A-Za-z]{2,10}-[A-Za-z0-9]{2,15})\b",
            r"#([A-Za-z0-9_-]{3,25})\b",
        ]
        order_id = None
        for pattern in order_patterns:
            match = re.search(pattern, human_text, re.IGNORECASE)
            if match:
                candidate = match.group(1).upper()
                if candidate.lower() not in [
                    "order",
                    "package",
                    "refund",
                    "return",
                    "status",
                    "cancel",
                    "where",
                    "please",
                ]:
                    order_id = candidate
                    break

        # Multi-turn reference: check earlier messages in conversation for order ID
        if not order_id and len(messages) > 1:
            for m in reversed(messages[:-1]):
                content_str = m.content if isinstance(m.content, str) else str(m.content)
                for pattern in order_patterns:
                    match = re.search(pattern, content_str, re.IGNORECASE)
                    if match:
                        candidate = match.group(1).upper()
                        if candidate.lower() not in [
                            "order",
                            "package",
                            "refund",
                            "return",
                            "status",
                            "cancel",
                            "where",
                            "please",
                        ]:
                            order_id = candidate
                            break
                if order_id:
                    break

        # Dynamic context recovery: if order ID is not in user text, check Active Order in SystemMessage
        if not order_id:
            for m in messages:
                if isinstance(m, SystemMessage):
                    match = re.search(r"Active Order:\s*([A-Za-z0-9_-]{3,25})", m.content)
                    if match:
                        candidate = match.group(1).upper()
                        if candidate.lower() not in ["none", "null"]:
                            order_id = candidate
                            break

        text_lower = human_text.lower()

        # Tool Call: Refund execution request
        if (
            "process" in text_lower or "yes" in text_lower or "confirm" in text_lower
        ) and "refund" in text_lower:
            if order_id:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "execute_refund",
                            "args": {"order_id": order_id, "reason_code": "CHANGED_MIND"},
                            "id": f"call_exec_{order_id}",
                        }
                    ],
                )
            else:
                return AIMessage(
                    content="Which order would you like me to process a refund for? Please provide the order number."
                )

        # Tool Call: Refund eligibility check
        if any(w in text_lower for w in ["refund", "return", "money back", "broken", "defective"]):
            if order_id:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "verify_refund_eligibility",
                            "args": {"order_id": order_id, "reason_code": "CHANGED_MIND"},
                            "id": f"call_ref_{order_id}",
                        }
                    ],
                )
            else:
                return AIMessage(
                    content="I would be happy to check refund eligibility for you. Could you please specify the order number?"
                )

        # Tool Call: Order status check
        if any(
            w in text_lower for w in ["track", "status", "where is", "order", "package", "delivery"]
        ):
            if order_id:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "check_order_status",
                            "args": {"order_id": order_id},
                            "id": f"call_status_{order_id}",
                        }
                    ],
                )
            else:
                return AIMessage(
                    content="I can look up live tracking for you! Which order number would you like me to check?"
                )

        # General escalation
        if any(w in text_lower for w in ["human", "agent", "representative", "lawyer", "manager"]):
            return AIMessage(
                content="I understand your need to speak with a specialist. I am escalating your request directly to our live support team right now."
            )

        # Default conversational greeting / assistance
        return AIMessage(
            content="Hello! I am your Autonomous Customer Support Specialist. I can look up live order tracking, verify return eligibility, and process refunds under our 30-day policy. How can I assist you today?"
        )


def get_llm(tools: list[Any] | None = None):
    """Factory providing either live ChatOpenAI or deterministic MockSupportLLM."""
    if settings.OPENAI_API_KEY and not settings.MOCK_LLM_MODE:
        try:
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(
                model=settings.MODEL_NAME, temperature=0, api_key=settings.OPENAI_API_KEY
            )
            if tools:
                return llm.bind_tools(tools)
            return llm
        except ImportError:
            pass

    return MockSupportLLM(tools=tools)
