import json
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from src.models.db_models import AuditLog


class AuditService:
    """Audit logging service for SOC2 and compliance traceability."""

    @staticmethod
    async def log_event(
        session: AsyncSession,
        customer_id: str,
        action: str,
        tool_name: str | None = None,
        input_payload: Any | None = None,
        output_payload: Any | None = None,
    ) -> None:
        """Record an immutable audit log entry."""
        log = AuditLog(
            customer_id=customer_id,
            action=action,
            tool_name=tool_name,
            input_payload=json.dumps(input_payload, default=str) if input_payload else None,
            output_payload=json.dumps(output_payload, default=str) if output_payload else None,
        )
        session.add(log)
        await session.commit()
