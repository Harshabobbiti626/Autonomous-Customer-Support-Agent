# Autonomous Customer Support Agent

[![CI](https://github.com/Harshabobbiti626/Autonomous-Customer-Support-Agent/actions/workflows/ci.yml/badge.svg)](https://github.com/Harshabobbiti626/Autonomous-Customer-Support-Agent/actions/workflows/ci.yml)
[![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/orchestration-LangGraph-orange)](https://github.com/langchain-ai/langgraph)
[![FastAPI](https://img.shields.io/badge/framework-FastAPI-009688)](https://fastapi.tiangolo.com/)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

A stateful, production-grade AI customer support agent built with **LangGraph**, **parameterized tool execution**, **deterministic guardrails**, and **Human-in-the-Loop (HITL) supervisor breakpoints**.

Designed to autonomously resolve **60%+ of incoming support inquiries** (order tracking, shipment logistics, return and refund eligibility) without human intervention, while deterministically enforcing financial thresholds, preventing IDOR vulnerabilities, and defending against prompt injection.

---

## 🏛️ Core Architectural Guarantees

1. **Zero-Direct-SQL Principle (IDOR Prevention)**:
   - LLMs **never** execute raw SQL queries.
   - All tools (`check_order_status` and `verify_refund_eligibility`) interact strictly through parameterized repository/service layers.
   - `customer_id` is bound programmatically from the authenticated session context and cannot be manipulated via prompts.
2. **Deterministic Financial & Return Governance**:
   - Return window strictly capped at **30 days** post-delivery.
   - Undelivered or in-transit orders cannot be refunded.
   - Clearance / Final Sale items are strictly excluded.
   - Autonomous refunds are hard-capped at **$50.00**. Any refund exceeding this limit triggers a LangGraph breakpoint for supervisor authorization.
3. **Financial Idempotency & Distributed Locking**:
   - SHA-256 idempotency keys (`customer_id:order_id:action`) prevent duplicate credits or double-refunding across network retries.
   - In-flight transaction locks prevent concurrent race condition exploits.
4. **Human-in-the-Loop (HITL) Breakpoints**:
   - Implements LangGraph's native `interrupt()` and `Command(resume=...)` to pause execution state into persistent checkpointers and resume via webhooks or supervisor review.
5. **Multi-Layer Guardrails**:
   - **Input Shield**: PII sanitizer (credit cards, SSNs, phone numbers) + Prompt Shield (heuristics, base64 payload decoding, leetspeak normalization, and prompt leak defense).
   - **Output Guard**: Policy hallucination filter, sentiment scoring, and context-aware disambiguation.

---

## 📁 Repository Structure

```
├── .github/
│   └── workflows/
│       └── ci.yml                 # GitHub Actions automated lint & test pipeline
├── src/
│   ├── config.py                  # Pydantic BaseSettings, governance thresholds & checkpointer config
│   ├── models/
│   │   ├── state.py               # LangGraph TypedDict AgentState, CustomerContext
│   │   ├── db_models.py           # SQLAlchemy 2.0 ORM (Customers, Orders, Items, Refunds, Audit)
│   │   └── api_schemas.py         # Request/Response schemas for REST & Webhooks
│   ├── db/
│   │   ├── session.py             # Async engine & session factory
│   │   ├── seed.py                # Database seed data covering diverse edge cases
│   │   └── services/
│   │       ├── order_service.py   # Parameterized query layer with IDOR isolation
│   │       ├── refund_service.py  # Refund rule verification, in-flight locks & idempotent execution
│   │       └── audit_service.py   # Immutable audit trail
│   ├── services/
│   │   └── crm_service.py         # Outbound CRM ticket dispatcher (Zendesk, Salesforce, mock queue)
│   ├── guardrails/
│   │   ├── pii_sanitizer.py       # PII masking (Credit cards, SSNs, phone numbers)
│   │   ├── prompt_shield.py       # Advanced threat shield (base64 decode, leetspeak, prompt extraction)
│   │   └── output_checker.py      # Policy hallucination filter & sentiment scoring
│   ├── tools/
│   │   ├── order_tools.py         # check_order_status tool
│   │   ├── refund_tools.py        # verify_refund_eligibility & execute_refund
│   │   └── tool_registry.py       # Session-bound tool factory
│   ├── agent/
│   │   ├── prompts.py             # System prompts with grounding rules
│   │   ├── llm_factory.py         # ChatOpenAI + deterministic MockSupportLLM with multi-turn context
│   │   ├── checkpointer_factory.py# Enterprise checkpointer (PostgresSaver, SqliteSaver, MemorySaver)
│   │   ├── nodes/
│   │   │   ├── auth_context.py    # Session context resolver & PII sanitizer
│   │   │   ├── router.py          # Intent routing & escalation gate
│   │   │   ├── support_worker.py  # Core tool-calling agent with token window trimming
│   │   │   ├── refund_governor.py # Tool executor & LangGraph interrupt() governor
│   │   │   ├── escalation.py      # Live agent handoff summary & CRM dispatcher
│   │   │   └── output_guard.py    # Policy verification & order disambiguation
│   │   └── graph.py               # Complete LangGraph StateGraph assembly
│   └── api/
│       ├── app.py                 # FastAPI application with latency telemetry & CORS
│       └── routes/
│           ├── chat.py            # POST /api/chat with state persistence
│           ├── hitl.py            # POST /api/hitl/decision supervisor resume webhook
│           └── orders.py          # GET /api/orders
├── tests/
│   ├── test_security_idor.py      # IDOR attack prevention tests
│   ├── test_guardrails.py         # PII masking, injection defense, sentiment tests
│   ├── test_refund_rules.py       # 30-day window, status, final-sale tests
│   ├── test_idempotency.py        # Duplicate transaction prevention & concurrency tests
│   ├── test_end_to_end_graph.py   # Full graph execution & HITL breakpoint tests
│   ├── test_api_endpoints.py      # FastAPI HTTP integration tests
│   └── test_gaps_coverage.py      # Concurrency lock, base64 defense, trimming, CRM dispatch tests
├── demo_interactive.py            # Interactive Rich terminal CLI
├── pyproject.toml                 # Modern PEP 621 packaging & tool configs (Ruff, Pytest)
├── requirements.txt               # Pinned production dependencies
└── .env.example                   # Environment configuration template
```

---

## 🚀 Getting Started

### 1. Prerequisites & Virtual Environment

Requires **Python 3.11+**.

```bash
# Clone the repository
git clone https://github.com/Harshabobbiti626/Autonomous-Customer-Support-Agent.git
cd Autonomous-Customer-Support-Agent

# Create and activate virtual environment
python -m venv .venv

# On Windows (PowerShell)
.\.venv\Scripts\activate

# On Linux/macOS
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

Copy the `.env.example` file:
```bash
cp .env.example .env
```
*(By default, `MOCK_LLM_MODE=true` or blank `OPENAI_API_KEY` runs the built-in deterministic LLM emulator, requiring zero external API keys for testing and demonstrations).*

---

## 🧪 Testing & Code Quality

### Run Test Suite
```bash
pytest -v
```
*Executes all 29 unit, integration, and security tests covering guardrails, IDOR protection, idempotency, and HITL pauses.*

### Code Quality & Linting
```bash
ruff check .
ruff format --check .
```

---

## 💻 Interactive Terminal CLI

Experience live agent workflows, order tracking, HITL breakpoints, and defense against prompt injection directly in your terminal:

```bash
python demo_interactive.py
```

---

## 📡 REST API Deployment

Start the asynchronous FastAPI server:
```bash
python -m src.api.app
```
Interactive OpenAPI documentation will be available at:
`http://localhost:8000/docs`

### Key Endpoints
| Method | Route | Description |
|---|---|---|
| `GET` | `/health` | System health check, active governance thresholds & checkpointer status |
| `GET` | `/api/orders` | Retrieve authenticated customer's order history |
| `POST` | `/api/chat` | Stateful conversation interaction, returning answers or HITL interrupt notifications |
| `POST` | `/api/hitl/decision` | Webhook for supervisor to approve or reject paused high-value transactions |

---

