from typing import Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.db.services.audit_service import AuditService
from src.db.services.refund_service import RefundService


class VerifyRefundSchema(BaseModel):
    order_id: str = Field(
        description="The order ID for which return/refund is requested, e.g. 'ORD-102'."
    )
    reason_code: Literal["DEFECTIVE", "LATE_DELIVERY", "WRONG_ITEM", "CHANGED_MIND"] = Field(
        default="CHANGED_MIND",
        description="Reason for refund: 'DEFECTIVE', 'LATE_DELIVERY', 'WRONG_ITEM', or 'CHANGED_MIND'.",
    )


class ExecuteRefundSchema(BaseModel):
    order_id: str = Field(description="The verified order ID to be refunded.")
    reason_code: str = Field(description="The validated reason code.")


def create_refund_tools(session, customer_id: str):
    """Factory creating refund tools strictly scoped to the authenticated customer."""

    @tool(args_schema=VerifyRefundSchema)
    async def verify_refund_eligibility(order_id: str, reason_code: str = "CHANGED_MIND") -> dict:
        """Check store return policy to verify whether an order is eligible for a return or refund."""
        cleaned_order_id = order_id.strip().upper()
        eligibility = await RefundService.evaluate_eligibility(
            session=session,
            order_id=cleaned_order_id,
            customer_id=customer_id,
            reason_code=reason_code,
        )

        result = {
            "order_id": eligibility.order_id,
            "eligible": eligibility.eligible,
            "reason": eligibility.reason,
            "max_refundable_amount": eligibility.max_refundable_amount,
            "requires_return_shipment": eligibility.requires_return_shipment,
            "requires_human_approval": eligibility.requires_human_approval,
            "approval_threshold": eligibility.approval_threshold,
        }

        await AuditService.log_event(
            session=session,
            customer_id=customer_id,
            action="VERIFY_REFUND_ELIGIBILITY",
            tool_name="verify_refund_eligibility",
            input_payload={"order_id": cleaned_order_id, "reason_code": reason_code},
            output_payload=result,
        )

        return result

    @tool(args_schema=ExecuteRefundSchema)
    async def execute_refund(order_id: str, reason_code: str) -> dict:
        """Submit a refund request for processing. If the amount exceeds $50.00, it flags for supervisor review."""
        cleaned_order_id = order_id.strip().upper()

        # Re-verify eligibility to prevent race conditions
        eligibility = await RefundService.evaluate_eligibility(
            session=session,
            order_id=cleaned_order_id,
            customer_id=customer_id,
            reason_code=reason_code,
        )

        if not eligibility.eligible:
            return {"success": False, "message": f"Refund cannot be executed: {eligibility.reason}"}

        idempotency_key = RefundService.generate_idempotency_key(customer_id, cleaned_order_id)

        # Check if amount requires human approval
        if eligibility.requires_human_approval:
            record, was_created = await RefundService.process_refund(
                session=session,
                order_id=cleaned_order_id,
                customer_id=customer_id,
                amount=eligibility.max_refundable_amount,
                reason_code=reason_code,
                idempotency_key=idempotency_key,
                status="PENDING_HUMAN_APPROVAL",
                reviewer_note="Exceeds autonomous refund threshold ($50.00)",
            )
            return {
                "success": True,
                "status": "PENDING_HUMAN_APPROVAL",
                "refund_id": record.id,
                "amount": record.amount,
                "message": f"Your refund request of ${record.amount:.2f} has been submitted and escalated to our supervisor team for manual review due to the amount exceeding $50.00.",
            }

        # Autonomous refund approval (under $50)
        record, was_created = await RefundService.process_refund(
            session=session,
            order_id=cleaned_order_id,
            customer_id=customer_id,
            amount=eligibility.max_refundable_amount,
            reason_code=reason_code,
            idempotency_key=idempotency_key,
            status="APPROVED",
            reviewer_note="Autonomously approved by support agent under policy rules",
        )

        await AuditService.log_event(
            session=session,
            customer_id=customer_id,
            action="EXECUTE_REFUND_AUTONOMOUS",
            tool_name="execute_refund",
            input_payload={"order_id": cleaned_order_id, "amount": record.amount},
            output_payload={
                "refund_id": record.id,
                "status": "APPROVED",
                "was_created": was_created,
            },
        )

        return {
            "success": True,
            "status": "APPROVED",
            "refund_id": record.id,
            "amount": record.amount,
            "message": f"Refund of ${record.amount:.2f} has been successfully processed to your original payment method.",
        }

    return [verify_refund_eligibility, execute_refund]
