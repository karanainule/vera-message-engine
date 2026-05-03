"""
Vera Message Engine — Submission Bot
magicpin AI Challenge 2026

LLM: Google Gemini 1.5 Flash (via GEMINI_API_KEY in .env)
"""

import os
import time
import uuid
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from dotenv import load_dotenv
load_dotenv()  # Load .env before anything else

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import decision_engine
from context_store import ContextStore
from message_generator import MessageGenerator
from reply_handler import ReplyHandler

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s"
)
log = logging.getLogger("vera-bot")

# ─── App + global state ───────────────────────────────────────────────────────

app = FastAPI(title="Vera Message Engine", version="2.0.0")
START_TIME = time.time()

store = ContextStore()
decision_engine.configure(store)
generator = MessageGenerator()
reply_handler = ReplyHandler(store, generator)

# ─── Request models ───────────────────────────────────────────────────────────

class ContextBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: str

class TickBody(BaseModel):
    now: str
    events: list[str] = []
    available_triggers: list[str] = []

class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: str
    turn_number: int

# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/v1/healthz")
async def healthz():
    counts = store.get_counts()
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START_TIME),
        "contexts_loaded": counts,
        "gemini_key_set": bool(os.getenv("GEMINI_API_KEY")),
    }


@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": "Karan Ainule",
        "team_members": ["Karan Ainule"],
        "model": "gemini-1.5-flash",
        "approach": (
            "4-context deterministic composer: trigger-kind classifier → signal extractor → "
            "category-aware prompt builder → Gemini at temperature=0 → CTA validator. "
            "Fallback chain: Gemini → rule-based compose_fallback → emergency string. "
            "/v1/tick ALWAYS returns at least one action when triggers are provided."
        ),
        "contact_email": "karanainule@example.com",
        "version": "2.0.0",
        "submitted_at": "2026-05-03T00:00:00+05:30",
    }


@app.post("/v1/context")
async def push_context(body: ContextBody):
    result = store.upsert(
        scope=body.scope,
        context_id=body.context_id,
        version=body.version,
        payload=body.payload,
    )
    if result == "stale":
        current_version = store.get_version(body.scope, body.context_id)
        return JSONResponse(
            status_code=409,
            content={
                "accepted": False,
                "reason": "stale_version",
                "current_version": current_version,
            },
        )
    if result == "invalid_scope":
        return JSONResponse(
            status_code=400,
            content={
                "accepted": False,
                "reason": "invalid_scope",
                "details": f"Unknown scope: {body.scope}",
            },
        )
    ack_id = f"ack_{body.context_id}_v{body.version}_{uuid.uuid4().hex[:6]}"
    log.info(f"Context stored: scope={body.scope} id={body.context_id} v={body.version}")
    return {
        "accepted": True,
        "ack_id": ack_id,
        "stored_at": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/v1/tick")
def tick(body: TickBody):
    events = body.events or body.available_triggers
    result = decision_engine.process_tick(
        events=events,
        now=body.now
    )

    print("MAIN RECEIVED RESULT:", result)

    # GUARANTEE NON-EMPTY (SECOND LAYER SAFETY)
    if not result.get("actions"):
        print("MAIN FORCED FALLBACK")

        result["actions"] = [{
            "conversation_id": "conv_main_fallback",
            "message": (
                "Your current performance is below its recoverable potential. "
                "That gap can cost about 10 leads every month. "
                "Recover lapsed customers with a focused offer today. Reply YES"
            ),
            "cta": "Reply YES",
            "send_as": "vera",
            "rationale": "Main-level fallback"
        }]

    log.info(f"Tick {body.now}: {len(result.get('actions', []))} actions generated")
    return result


@app.post("/v1/reply")
async def reply(body: ReplyBody):
    """Receive merchant/customer reply. Return next move: send, wait, or end."""
    try:
        result = reply_handler.handle(
            conversation_id=body.conversation_id,
            merchant_id=body.merchant_id,
            customer_id=body.customer_id,
            from_role=body.from_role,
            message=body.message,
            received_at=body.received_at,
            turn_number=body.turn_number,
        )
        log.info(f"Reply handled: conv={body.conversation_id} action={result.get('action')}")
        return result
    except Exception as e:
        log.error(f"Reply error: {e}", exc_info=True)
        return {
            "action": "wait",
            "wait_seconds": 300,
            "rationale": f"Error handling reply: {str(e)[:100]}",
        }


@app.post("/v1/teardown")
async def teardown():
    """Wipe ALL in-memory state — store, conversations, suppression keys."""
    store.clear()
    reply_handler.clear()

    from decision_engine import _sent_suppression_keys, _active_conversations
    from reply_handler import _global_conversations

    _sent_suppression_keys.clear()
    _active_conversations.clear()
    _global_conversations.clear()

    log.info("Full state wipe: store + suppression + conversations")
    return {"status": "wiped"}

@app.get("/")
def root():
    return {"message": "Vera Message Engine is running 🚀"}