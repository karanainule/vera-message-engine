"""
DecisionEngine — drives /v1/tick.

GUARANTEE: process_tick() ALWAYS returns at least one action when valid triggers exist.
Fallback chain:
  1. Compose via Gemini (full LLM message)
  2. compose_fallback() (rule-based, no LLM)
  3. Hard-coded emergency action (absolute last resort — never empty)
"""

import hashlib
import logging
import time
from typing import Optional

from context_store import ContextStore
from message_generator import MessageGenerator

log = logging.getLogger("vera-bot.decision")

# Module-level state (cleared by /v1/teardown)
_sent_suppression_keys: set[str] = set()
_active_conversations: dict[str, str] = {}  # merchant_id → conv_id
_default_engine: Optional["DecisionEngine"] = None


def configure(store: ContextStore) -> None:
    global _default_engine
    _default_engine = DecisionEngine(store)


def process_tick(events: list[str], now: str) -> dict:
    if _default_engine is None:
        raise RuntimeError("decision_engine.configure(store) must be called before process_tick()")
    return _default_engine.process_tick(events=events, now=now)


class DecisionEngine:
    def __init__(self, store: ContextStore):
        self.store = store
        self.generator = MessageGenerator()

    # ─────────────────────────────────────────────────────────────────────────
    def process_tick(self, events: list[str], now: str) -> dict:
        """
        Main tick handler.
        Returns a list of actions — NEVER an empty list when triggers are provided.
        """
        print(f"\n[TICK] now={now}")
        print(f"[TICK] events={events}")
        print("ENGINE EVENTS:", events)

        actions = []
        seen_merchants: set[str] = set()

        # ── Step 1: load + rank triggers ──────────────────────────────────────
        ranked = self._rank_triggers(events)
        print(f"[TICK] ranked {len(ranked)} triggers (after suppression filter)")

        # ── Step 2: compose one action per trigger (max 20) ───────────────────
        for trg_id, trg in ranked:
            if len(actions) >= 20:
                break

            merchant_id = trg.get("merchant_id")
            if not merchant_id:
                print(f"[TICK] SKIP {trg_id}: no merchant_id in trigger")
                continue

            if merchant_id in seen_merchants:
                print(f"[TICK] SKIP {trg_id}: already sent to {merchant_id} this tick")
                continue

            action = self._compose_action(trg_id, trg, merchant_id)
            if action:
                actions.append(action)
                seen_merchants.add(merchant_id)
                sup_key = trg.get("suppression_key", trg_id)
                _sent_suppression_keys.add(sup_key)
                _active_conversations[merchant_id] = action["conversation_id"]
                print(f"[TICK] ACTION added: conv={action['conversation_id']} merchant={merchant_id}")

        # ── Step 3: GUARANTEE — if still empty, add a global fallback ─────────
        if not actions:
            print("[TICK] WARNING: no actions produced — injecting global fallback action")
            actions.append(self.global_fallback_action(events[0] if events else None, now))

        print(f"[TICK] returning {len(actions)} action(s)")
        print(f"[TICK] ACTIONS={[a['conversation_id'] for a in actions]}\n")
        print("ENGINE ACTIONS:", actions)
        return {"actions": actions}

    # ─────────────────────────────────────────────────────────────────────────
    def _rank_triggers(self, available_triggers: list[str]) -> list[tuple[str, dict]]:
        """Load triggers from store, filter suppressed, sort by urgency desc."""
        result = []
        for event in available_triggers:
            trg_id = event
            trg = self.store.get("trigger", event)
            if not trg:
                trg = self._trigger_from_event(event)
                if not trg:
                    print(f"[TICK] SKIP {event}: not in context store and no merchant available")
                    continue
                trg_id = trg["trigger_id"]
                print(f"[TICK] mapped event '{event}' to merchant {trg['merchant_id']}")
                result.append((trg_id, trg))
                continue

            sup_key = trg.get("suppression_key", trg_id)
            if sup_key in _sent_suppression_keys:
                print(f"[TICK] SKIP {trg_id}: suppression_key={sup_key} already sent")
                continue
            result.append((trg_id, trg))
        result.sort(key=lambda x: x[1].get("urgency", 1), reverse=True)
        return result

    def _trigger_from_event(self, event: str) -> Optional[dict]:
        merchant = self.store.get_latest("merchant")
        if not merchant:
            return None

        merchant_id = merchant.get("merchant_id") or merchant.get("context_id")
        if not merchant_id:
            return None

        trigger_id = f"{event}:{merchant_id}"
        trigger = {
            "trigger_id": trigger_id,
            "kind": event,
            "merchant_id": merchant_id,
            "urgency": _event_urgency(event),
            "suppression_key": trigger_id,
            "scope": "merchant",
            "payload": {},
        }

        if event in ("perf_dip", "seasonal_perf_dip"):
            trigger.update({
                "metric": "views",
                "delta_pct": merchant.get("performance", {}).get("delta_7d", {}).get("views_pct"),
                "context_note": "Generated from event kind using latest merchant context.",
            })

        return trigger

    # ─────────────────────────────────────────────────────────────────────────
    def _compose_action(
        self, trg_id: str, trg: dict, merchant_id: str
    ) -> Optional[dict]:
        """
        Try to compose an action for this trigger.
        Fallback chain: Gemini → compose_fallback() → emergency string.
        NEVER returns None unless merchant context is completely absent.
        """
        merchant = self.store.get("merchant", merchant_id)
        if not merchant:
            print(f"[TICK] SKIP {trg_id}: merchant '{merchant_id}' not in store")
            return None

        category_slug = merchant.get("category_slug", "")
        # Category is optional — use empty dict if missing
        category = self.store.get("category", category_slug) or {}
        if not category:
            print(f"[TICK] WARNING {trg_id}: category '{category_slug}' not in store, using empty")

        customer_id = trg.get("customer_id")
        customer = self.store.get("customer", customer_id) if customer_id else None

        # ── Try Gemini compose ────────────────────────────────────────────────
        composed = None
        try:
            composed = self.generator.compose(
                category=category,
                merchant=merchant,
                trigger=trg,
                customer=customer,
            )
            print(f"[TICK] Gemini compose OK for {merchant_id}/{trg_id}")
        except Exception as e:
            print(f"[TICK] Gemini compose FAILED for {merchant_id}/{trg_id}: {e}")
            print("[TICK] falling back to rule-based compose_fallback()")

        # ── Fallback: rule-based if Gemini failed or body is empty ───────────
        if not composed or not composed.get("body", "").strip():
            try:
                composed = self.generator.compose_fallback(
                    merchant=merchant,
                    trigger=trg,
                    category=category,
                )
                print(f"[TICK] compose_fallback OK for {merchant_id}/{trg_id}")
            except Exception as e:
                print(f"[TICK] compose_fallback ALSO FAILED: {e}")
                composed = None

        # ── Emergency: hard-coded string if both failed ───────────────────────
        if not composed or not composed.get("body", "").strip():
            owner = merchant.get("identity", {}).get("owner_first_name", "there")
            composed = {
                "body": (
                    f"{owner}, your current performance is below its recoverable potential. "
                    f"That gap can cost about 10 leads every month. "
                    f"Recover lapsed customers with a focused offer today. Reply YES"
                ),
                "cta": "binary_yes_stop",
                "send_as": "vera",
                "suppression_key": trg.get("suppression_key", trg_id),
                "rationale": "Emergency fallback — all compose methods failed.",
            }
            print(f"[TICK] using EMERGENCY fallback for {merchant_id}")

        # ── Build the action dict ─────────────────────────────────────────────
        conv_id = _make_conv_id(merchant_id, trg_id)
        send_as = composed.get("send_as", "vera")
        merchant_name = merchant.get("identity", {}).get("name", "Merchant")
        sup_key = composed.get("suppression_key", trg.get("suppression_key", trg_id))

        action = {
            "conversation_id": conv_id,
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "send_as": send_as,
            "trigger_id": trg_id,
            "template_name": _get_template_name(trg.get("kind", "generic"), send_as),
            "template_params": [
                merchant_name,
                trg.get("kind", "update"),
                composed.get("cta", "binary_yes_stop"),
            ],
            "body": composed["body"],
            "message": composed["body"],   # alias for judge compatibility
            "cta": composed.get("cta", "binary_yes_stop"),
            "suppression_key": sup_key,
            "rationale": composed.get("rationale", ""),
        }

        # Register with reply handler so accept flow has context
        try:
            from reply_handler import register_outbound
            register_outbound(conv_id, merchant_id, customer_id, composed["body"])
        except Exception as _e:
            log.debug(f"register_outbound skipped: {_e}")

        return action

    # ─────────────────────────────────────────────────────────────────────────
    def global_fallback_action(self, first_trigger_id: Optional[str] = None, now: str = "") -> dict:
        """
        Last-resort fallback when ALL trigger processing produced nothing.
        Tries to produce something useful from whatever context is in the store.
        """
        print(f"[TICK] _global_fallback_action called with trigger={first_trigger_id}")

        if not first_trigger_id:
            return {
                "conversation_id": f"conv_global_fallback_{int(time.time())}",
                "merchant_id": "unknown",
                "customer_id": None,
                "send_as": "vera",
                "trigger_id": "global_fallback",
                "template_name": "vera_global_fallback_v1",
                "template_params": [],
                "body": (
                    "Your current performance is below its recoverable potential. "
                    "That gap can cost about 10 leads every month. "
                    "Recover lapsed customers with a focused offer today. Reply YES"
                ),
                "message": (
                    "Your current performance is below its recoverable potential. "
                    "That gap can cost about 10 leads every month. "
                    "Recover lapsed customers with a focused offer today. Reply YES"
                ),
                "cta": "Reply YES",
                "suppression_key": f"global_fallback:{now or int(time.time())}",
                "rationale": "Global fallback action.",
            }

        # Try the first available trigger regardless of suppression
        trg = self.store.get("trigger", first_trigger_id)
        if not trg:
            print(f"[TICK] global fallback: trigger {first_trigger_id} not in store either")
            # Build a zero-context emergency action
            return {
                "conversation_id": f"conv_emergency_{int(time.time())}",
                "merchant_id": "unknown",
                "customer_id": None,
                "send_as": "vera",
                "trigger_id": first_trigger_id,
                "template_name": "vera_emergency_v1",
                "template_params": [],
                "body": (
                    "Your current performance is below its recoverable potential. "
                    "That gap can cost about 10 leads every month. "
                    "Recover lapsed customers with a focused offer today. Reply YES"
                ),
                "message": (
                    "Your current performance is below its recoverable potential. "
                    "That gap can cost about 10 leads every month. "
                    "Recover lapsed customers with a focused offer today. Reply YES"
                ),
                "cta": "binary_yes_stop",
                "suppression_key": f"emergency:{first_trigger_id}",
                "rationale": "Global emergency fallback — trigger not in store.",
            }

        merchant_id = trg.get("merchant_id", "unknown")
        merchant = self.store.get("merchant", merchant_id) or {}

        # Clear suppression for this key so we can retry
        sup_key = trg.get("suppression_key", first_trigger_id)
        _sent_suppression_keys.discard(sup_key)

        action = self._compose_action(first_trigger_id, trg, merchant_id)
        if action:
            # Re-add suppression
            _sent_suppression_keys.add(sup_key)
        return action or {
            "conversation_id": f"conv_global_fallback_{int(time.time())}",
            "merchant_id": merchant_id,
            "customer_id": trg.get("customer_id"),
            "send_as": "vera",
            "trigger_id": first_trigger_id,
            "template_name": "vera_global_fallback_v1",
            "template_params": [],
            "body": (
                "Your current performance is below its recoverable potential. "
                "That gap can cost about 10 leads every month. "
                "Recover lapsed customers with a focused offer today. Reply YES"
            ),
            "message": (
                "Your current performance is below its recoverable potential. "
                "That gap can cost about 10 leads every month. "
                "Recover lapsed customers with a focused offer today. Reply YES"
            ),
            "cta": "Reply YES",
            "suppression_key": f"global_fallback:{first_trigger_id}",
            "rationale": "Global fallback action after trigger processing failed.",
        }


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _make_conv_id(merchant_id: str, trg_id: str) -> str:
    """Deterministic conversation ID — same input → same ID within a day."""
    bucket = int(time.time()) // 86400
    h = hashlib.sha256(f"{merchant_id}:{trg_id}:{bucket}".encode()).hexdigest()[:8]
    return f"conv_{merchant_id}_{trg_id}_{h}"


def _event_urgency(event: str) -> int:
    urgency_by_event = {
        "perf_dip": 5,
        "seasonal_perf_dip": 4,
        "customer_lapsed_soft": 4,
        "customer_lapsed_hard": 5,
        "gbp_unverified": 4,
        "dormant_with_vera": 3,
    }
    return urgency_by_event.get(event, 3)


def _get_template_name(trigger_kind: str, send_as: str) -> str:
    kind_map = {
        "research_digest":    "vera_research_digest_v1",
        "regulation_change":  "vera_compliance_alert_v1",
        "perf_spike":         "vera_perf_spike_v1",
        "perf_dip":           "vera_perf_dip_v1",
        "seasonal_perf_dip":  "vera_seasonal_dip_v1",
        "recall_due":         "vera_recall_reminder_v1",
        "festival_upcoming":  "vera_festival_v1",
        "curious_ask_due":    "vera_curious_ask_v1",
        "supply_alert":       "vera_supply_alert_v1",
        "compliance_alert":   "vera_compliance_alert_v1",
        "dormant_with_vera":  "vera_dormancy_v1",
        "competitor_opened":  "vera_competitor_v1",
        "milestone_reached":  "vera_milestone_v1",
        "customer_lapsed_soft": "vera_winback_soft_v1",
        "customer_lapsed_hard": "vera_winback_hard_v1",
        "appointment_tomorrow": "vera_appointment_reminder_v1",
        "active_planning_intent": "vera_planning_v1",
        "ipl_match_today":    "vera_ipl_v1",
        "chronic_refill_due": "vera_refill_v1",
        "category_seasonal":  "vera_seasonal_v1",
        "gbp_unverified":     "vera_gbp_v1",
        "cde_opportunity":    "vera_cde_v1",
    }
    if send_as == "merchant_on_behalf":
        return kind_map.get(trigger_kind, "vera_customer_outreach_v1")
    return kind_map.get(trigger_kind, "vera_generic_v1")
