"""
ReplyHandler — manages multi-turn conversation state and reply composition.

Key capabilities:
1. Auto-reply detection (canned WA Business responses)
2. Intent transition detection (merchant says "let's do it")
3. Graceful exit (merchant says not interested / 3 unanswered nudges)
4. Hostile/off-topic routing
5. Language continuity

Stateful: maintains conversation history per conversation_id.
"""

import json
import logging
from typing import Optional

from context_store import ContextStore
from message_generator import MessageGenerator

log = logging.getLogger("vera-bot.reply")

# Known auto-reply patterns (verbatim fragments)
AUTO_REPLY_SIGNALS = [
    "aapki jaankari ke liye bahut-bahut shukriya",
    "main aapki yeh sabhi baatein",
    "main ek automated assistant hoon",
    "i am an automated assistant",
    "thank you for contacting",
    "thanks for contacting",
    "this is an automated response",
    "aapki madad ke liye shukriya",
    "team tak pahuncha deti hoon",
    "we will get back to you",
    "our team will contact you",
    "aapki jaankari ke liye shukriya",
]

# Intent signals → action mode
ACCEPT_INTENT_SIGNALS = [
    "yes", "haan", "ha", "yes please", "go ahead", "let's do it", "karo",
    "kar do", "sounds good", "ok sure", "ok go", "please go ahead",
    "yes do it", "proceed", "yes send", "send it", "draft kar do",
    "ok kar do", "bilkul", "zaroor", "chalega", "theek hai",
]

# Exit signals
# FIX: phrase-level signals (require word boundaries) vs substring signals
EXIT_PHRASE_SIGNALS = [
    "not interested", "no thanks", "nahi chahiye", "rehne do",
    "no need", "don't contact", "unsubscribe",
    "mat bhejo", "band karo", "go away",
    "stop the campaign", "stop sending", "stop messaging", "please stop",
]
# Single-word signals — only match when the ENTIRE message is this word
EXIT_EXACT_SIGNALS = [
    "stop", "nahin", "enough", "spam",
]

class ConversationState:
    def __init__(self, conversation_id: str, merchant_id: str, customer_id: Optional[str]):
        self.conversation_id = conversation_id
        self.merchant_id = merchant_id
        self.customer_id = customer_id
        self.turns: list[dict] = []
        self.auto_reply_count: int = 0
        self.last_bot_message: str = ""
        self.intent_accepted: bool = False
        self.ended: bool = False
        self.unanswered_count: int = 0
        self.seen_messages: list[str] = []


# Module-level conversations store (allows register_outbound from decision_engine)
_global_conversations: dict[str, "ConversationState"] = {}


def register_outbound(conv_id: str, merchant_id: str, customer_id: Optional[str], body: str) -> None:
    """Called by DecisionEngine after /tick to register the bot's first message.
    FIX: ensures last_bot_message is set BEFORE merchant replies, so accept flow
    can deliver the correct artifact instead of a generic 'On it' message."""
    state = _global_conversations.get(conv_id)
    if not state:
        state = ConversationState(conv_id, merchant_id, customer_id)
        _global_conversations[conv_id] = state
    state.last_bot_message = body
    # Record vera's outbound turn for correct turn tracking
    state.turns.append({"from": "vera", "msg": body, "tag": "outbound"})


class ReplyHandler:
    def __init__(self, store: ContextStore, generator: MessageGenerator):
        self.store = store
        self.generator = generator
        self._conversations = _global_conversations  # share module-level dict

    def handle(
        self,
        conversation_id: str,
        merchant_id: Optional[str],
        customer_id: Optional[str],
        from_role: str,
        message: str,
        received_at: str,
        turn_number: int,
    ) -> dict:
        # Get or create conversation state
        if conversation_id not in self._conversations:
            self._conversations[conversation_id] = ConversationState(
                conversation_id, merchant_id or "", customer_id
            )
        state = self._conversations[conversation_id]

        if state.ended:
            return {"action": "end", "rationale": "Conversation already ended"}

        msg_lower = message.lower().strip()

        # ── Auto-reply detection ────────────────────────────────────────────
        if _is_auto_reply(message, state):
            state.auto_reply_count += 1
            state.turns.append({"from": from_role, "msg": message, "tag": "auto_reply"})

            if state.auto_reply_count == 1:
                # Try once to get past the auto-reply
                return {
                    "action": "send",
                    "body": _compose_auto_reply_pierce(state, merchant_id),
                    "cta": "binary_yes_no",
                    "rationale": "Detected auto-reply; attempting one pass-through"
                }
            else:
                # Graceful exit after second auto-reply
                state.ended = True
                return {
                    "action": "end",
                    "rationale": "Multiple auto-replies detected; owner not present. Exiting gracefully."
                }

        # ── Exit intent detection ────────────────────────────────────────────
        if _is_exit_intent(msg_lower):
            state.ended = True
            state.turns.append({"from": from_role, "msg": message, "tag": "exit"})
            return {
                "action": "end",
                "rationale": "Merchant signaled not interested; exiting without further nudge"
            }

        # ── Hostile/off-topic detection ──────────────────────────────────────
        if _is_hostile(msg_lower):
            state.ended = True
            return {
                "action": "end",
                "rationale": "Hostile message received; disengaging respectfully"
            }

        # ── Accept intent detection ──────────────────────────────────────────
        if _is_accept_intent(msg_lower):
            state.intent_accepted = True
            state.turns.append({"from": from_role, "msg": message, "tag": "accept"})
            return self._handle_accept(state, merchant_id, customer_id)

        # ── Normal reply — compose contextual response ────────────────────────
        state.turns.append({"from": from_role, "msg": message, "tag": "reply"})
        state.unanswered_count = 0  # They replied, reset counter

        # FIX: count ALL turns (vera + merchant) — vera turns now tracked via register_outbound
        # End after 5 VERA outbound turns (not counting auto-reply piercing attempts)
        vera_turns = [t for t in state.turns if t["from"] == "vera" and t.get("tag") != "auto_reply"]
        if len(vera_turns) >= 5:
            state.ended = True
            return {
                "action": "end",
                "rationale": f"5-turn limit reached ({len(vera_turns)} vera turns); ending cleanly"
            }

        return self._compose_contextual_reply(state, merchant_id, customer_id, message)

    def clear(self):
        self._conversations.clear()

    def _handle_accept(
        self,
        state: ConversationState,
        merchant_id: Optional[str],
        customer_id: Optional[str],
    ) -> dict:
        """Merchant said YES — immediately deliver the promised artifact.
        FIX: state.last_bot_message is now set by register_outbound() from DecisionEngine,
        so this always has the original offer context to deliver against."""
        merchant = self.store.get("merchant", merchant_id or "") if merchant_id else None

        # FIX: last_bot_message is now always set via register_outbound from tick
        last_bot_body = state.last_bot_message

        if not merchant or not last_bot_body:
            # True fallback: no merchant context OR first message was lost
            # Use a helpful but non-generic response
            fallback_body = (
                "Got it! Give me 30 seconds to put this together for you. "
                "I'll send the draft in the next message — check back in a moment."
            )
            state.last_bot_message = fallback_body
            state.turns.append({"from": "vera", "msg": fallback_body, "tag": "reply"})
            return {
                "action": "send",
                "body": fallback_body,
                "cta": "none",
                "rationale": "Accept received; no merchant context — returning preparation acknowledgment"
            }

        category_slug = merchant.get("category_slug", "")
        category = self.store.get("category", category_slug) or {}
        owner = merchant.get("identity", {}).get("owner_first_name", "")
        active_offers = [o["title"] for o in merchant.get("offers", []) if o.get("status") == "active"]
        voice = category.get("voice", {})

        system = f"""You are Vera, magicpin's merchant-AI assistant.
The merchant {owner} just said YES to your offer.

Your PREVIOUS MESSAGE (what you offered): 
"{last_bot_body}"

Business: {merchant.get("identity", {}).get("name", "")} | Category: {category_slug.upper()}
Voice: {voice.get("tone", "peer")}
Active offers: {active_offers}
Taboo words: {voice.get("vocab_taboo", [])}

INSTRUCTIONS — DELIVER THE ARTIFACT NOW:
- Do NOT say "I'll prepare this" or "give me a moment" — you must deliver it immediately
- If you offered to draft a WhatsApp message: write the full draft now
- If you offered to draft a Google post: write the full post now (150-200 words)
- If you offered to create a campaign plan: write the specific steps now
- If you offered to pull/show data: show the actual numbers
- If you offered to share a checklist: write the full checklist now
- End with ONE follow-up question: "Want me to [next logical step]? Reply YES/NO"
- Match the category voice exactly
- Keep total response under 300 words

Return ONLY valid JSON:
{{"body": "...", "cta": "open_ended", "rationale": "1 sentence on what artifact was delivered"}}"""

        last_merchant_msg = state.turns[-2]["msg"] if len(state.turns) >= 2 else "yes"
        user = (
            f"Merchant said: \"{last_merchant_msg}\"\n\n"
            f"Deliver the artifact promised in your previous message now. Be specific and complete."
        )

        raw = _MessageGenerator().compose(system, user)
        result = _parse_reply_output(raw)
        result["action"] = "send"
        vera_body = result.get("body", "")
        state.last_bot_message = vera_body
        # Track vera's turn for correct turn counting
        state.turns.append({"from": "vera", "msg": vera_body, "tag": "reply"})
        return result

    def _compose_contextual_reply(
        self,
        state: ConversationState,
        merchant_id: Optional[str],
        customer_id: Optional[str],
        merchant_message: str,
    ) -> dict:
        """Compose a contextual reply based on what the merchant said."""
        merchant = self.store.get("merchant", merchant_id or "") if merchant_id else None
        if not merchant:
            return {
                "action": "send",
                "body": "Noted! Let me know if you'd like me to take any action on that.",
                "cta": "open_ended",
                "rationale": "No merchant context; generic acknowledgment"
            }

        category_slug = merchant.get("category_slug", "")
        category = self.store.get("category", category_slug) or {}
        owner = merchant.get("identity", {}).get("owner_first_name", "")

        # Build conversation history for context
        history = "\n".join([
            f"{'VERA' if t['from'] in ('vera','bot') else 'MERCHANT'}: {t['msg']}"
            for t in state.turns[-6:]  # Last 6 turns
        ])

        system = f"""You are Vera, magicpin's merchant-AI assistant.
You are in a WhatsApp conversation with {owner} at {merchant.get('identity', {}).get('name', '')}.
Category: {category_slug.upper()}
Voice: {category.get('voice', {}).get('tone', 'peer')}
Taboo words: {category.get('voice', {}).get('vocab_taboo', [])}

Rules:
- Continue naturally from the conversation history
- Don't re-introduce yourself
- Don't ask multiple questions
- One clear next step in your reply
- If they asked about something off-topic (e.g., GST, legal), politely redirect: "That's outside what I handle, but for your business on magicpin..."
- Match their language (Hindi-English mix if they used Hindi)

Return JSON: {{"body": "...", "cta": "open_ended|binary_yes_stop|none", "rationale": "..."}}"""

        user = f"""CONVERSATION SO FAR:
{history}

MERCHANT'S LATEST MESSAGE: "{merchant_message}"

Compose Vera's next reply. Be specific, grounded, actionable."""

        raw = MessageGenerator().compose(system, user)
        result = _parse_reply_output(raw)
        result["action"] = "send"
        vera_body = result.get("body", "")
        state.last_bot_message = vera_body
        # FIX: track vera's turn for correct turn counting
        state.turns.append({"from": "vera", "msg": vera_body, "tag": "reply"})
        return result


# ─── Detection Helpers ────────────────────────────────────────────────────────

def _is_auto_reply(message: str, state: ConversationState) -> bool:
    msg_lower = message.lower()
    # Check for known canned patterns
    for pattern in AUTO_REPLY_SIGNALS:
        if pattern in msg_lower:
            return True
    # Detect repetition (same message 2+ times)
    if message in state.seen_messages:
        return True
    state.seen_messages.append(message)
    return False


def _is_exit_intent(msg_lower: str) -> bool:
    # Phrase match: must be a substring but is long enough to be unambiguous
    if any(signal in msg_lower for signal in EXIT_PHRASE_SIGNALS):
        return True
    # Single-word match: only when the full message IS the word (prevents "stop me worrying")
    stripped = msg_lower.strip().rstrip("!?.")
    if stripped in EXIT_EXACT_SIGNALS:
        return True
    return False


def _is_accept_intent(msg_lower: str) -> bool:
    return any(signal == msg_lower.strip() or f" {signal} " in f" {msg_lower} " for signal in ACCEPT_INTENT_SIGNALS)


def _is_hostile(msg_lower: str) -> bool:
    hostile_markers = ["abuse", "idiot", "shut up", "stop messaging", "bother me", "harassing"]
    return any(m in msg_lower for m in hostile_markers)


def _compose_auto_reply_pierce(state: ConversationState, merchant_id: Optional[str]) -> str:
    """One attempt to reach the real owner past the auto-reply."""
    return (
        "Samajh gayi — lagta hai auto-reply chal rahi hai. "
        "Kya owner/manager available hain? Main 2 min mein kuch useful share kar sakti hoon "
        "unke business ke liye. Reply YES agar theek hai."
    )


def _parse_reply_output(raw: str) -> dict:
    """Parse LLM output for reply endpoint."""
    import re
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        inner = []
        in_block = False
        for line in lines:
            if line.startswith("```") and not in_block:
                in_block = True
                continue
            if line.startswith("```") and in_block:
                break
            if in_block:
                inner.append(line)
        text = "\n".join(inner).strip()

    try:
        return json.loads(text)
    except Exception:
        match = re.search(r'\{.*?"body".*?\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
    return {"body": raw[:400], "cta": "open_ended", "rationale": "Fallback"}
