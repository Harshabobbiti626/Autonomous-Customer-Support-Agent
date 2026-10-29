import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import chat, hitl, orders
from src.config import settings
from src.db.seed import seed_data
from src.db.session import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown hooks."""
    # Ensure database tables exist and seed demo records
    await init_db()
    await seed_data()
    yield


app = FastAPI(
    title=settings.APP_NAME,
    description="Production-grade Autonomous Customer Support Agent built on LangGraph.",
    version="1.0.0",
    lifespan=lifespan,
)


# Request latency and telemetry middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = time.perf_counter() - start_time
    response.headers["X-Response-Time-Seconds"] = f"{process_time:.4f}"
    return response


# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Sub-Routers
app.include_router(chat.router)
app.include_router(hitl.router)
app.include_router(orders.router)


@app.get("/health", tags=["Health"])
async def health_check():
    """Health status and autonomous governance thresholds."""
    return {
        "status": "HEALTHY",
        "service": settings.APP_NAME,
        "environment": settings.ENV,
        "checkpointer": {
            "type": settings.CHECKPOINTER_TYPE,
            "persistent": settings.CHECKPOINTER_TYPE in ("sqlite", "postgres"),
        },
        "crm": {"provider": settings.CRM_PROVIDER},
        "governance": {
            "max_autonomous_refund_threshold": f"${settings.MAX_AUTONOMOUS_REFUND_AMOUNT:.2f}",
            "refund_window_days": settings.REFUND_WINDOW_DAYS,
            "negative_sentiment_threshold": settings.NEGATIVE_SENTIMENT_THRESHOLD,
            "max_conversation_turns": settings.MAX_CONVERSATION_TURNS,
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=settings.DEBUG)
