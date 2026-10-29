import os
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Production configuration and operational thresholds."""

    APP_NAME: str = "Autonomous Support Agent"
    ENV: Literal["development", "staging", "production"] = "development"
    DEBUG: bool = True

    # Database Settings (defaults to local sqlite for zero-friction setup, supports Postgres)
    DATABASE_URL: str = "sqlite+aiosqlite:///./support_agent.db"

    # Checkpointer Engine (supports 'memory', 'sqlite', 'postgres' for horizontal scaling)
    CHECKPOINTER_TYPE: Literal["memory", "sqlite", "postgres"] = "memory"
    CHECKPOINTER_DB_PATH: str = "./checkpoints.db"
    POSTGRES_CHECKPOINTER_URL: str = ""

    # Financial & Business Governance
    MAX_AUTONOMOUS_REFUND_AMOUNT: float = 50.0  # Max USD value agent can approve without human
    REFUND_WINDOW_DAYS: int = 30  # Max days since delivery for eligible refund

    # Safety & Escalation Thresholds
    NEGATIVE_SENTIMENT_THRESHOLD: float = -0.60  # Triggers human handoff if below
    MAX_CONSECUTIVE_TOOL_FAILURES: int = 2
    MAX_CONVERSATION_TURNS: int = 12  # Maximum turns before message window trimming
    MAX_ALLOWED_TURNS: int = 15

    # CRM & Helpdesk Integration (Zendesk / Salesforce)
    CRM_PROVIDER: Literal["mock", "zendesk", "salesforce"] = "mock"
    CRM_API_URL: str = ""
    CRM_API_KEY: str = ""

    # Carrier API & Service Timeouts (seconds)
    TOOL_TIMEOUT_SECONDS: float = 2.0

    # Mock LLM Mode for Deterministic Testing without external API key
    MOCK_LLM_MODE: bool = False

    # LLM Provider Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    MODEL_NAME: str = "gpt-4o-mini"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
