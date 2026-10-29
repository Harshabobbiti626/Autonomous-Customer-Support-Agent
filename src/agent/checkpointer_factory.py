import logging

from langgraph.checkpoint.memory import MemorySaver

from src.config import settings

logger = logging.getLogger(__name__)


def get_checkpointer():
    """Factory providing persistent checkpointer (Postgres, SQLite, or Memory) based on configuration."""
    cp_type = settings.CHECKPOINTER_TYPE.lower()

    if cp_type == "postgres":
        if settings.POSTGRES_CHECKPOINTER_URL:
            try:
                from langgraph.checkpoint.postgres import PostgresSaver

                # Setup tables if needed and return saver
                saver = PostgresSaver.from_conn_string(settings.POSTGRES_CHECKPOINTER_URL)
                saver.setup()
                logger.info("Configured PostgresSaver persistent checkpointer.")
                return saver
            except Exception as e:
                logger.warning(
                    "Failed to initialize PostgresSaver (%s). Falling back to MemorySaver.", e
                )
        else:
            logger.warning("POSTGRES_CHECKPOINTER_URL not configured. Falling back to MemorySaver.")

    elif cp_type == "sqlite":
        try:
            import sqlite3

            from langgraph.checkpoint.sqlite import SqliteSaver

            conn = sqlite3.connect(settings.CHECKPOINTER_DB_PATH, check_same_thread=False)
            saver = SqliteSaver(conn)
            saver.setup()
            logger.info(
                "Configured SqliteSaver persistent checkpointer at %s.",
                settings.CHECKPOINTER_DB_PATH,
            )
            return saver
        except Exception as e:
            logger.warning("Failed to initialize SqliteSaver (%s). Falling back to MemorySaver.", e)

    return MemorySaver()
