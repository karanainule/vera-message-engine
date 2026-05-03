"""
MessageGenerator — the heart of Vera.

LLM: Google Gemini 1.5 Flash (temperature=0 for determinism)
Fallback: rule-based compose_fallback() — no LLM, always returns a valid message.
"""

import os
import json
import logging
import re
from typing import Optional

from google import genai
from google.genai import types
from dotenv import load_dotenv

from signal_extractor import (
    extract_merchant_signals,
    extract_trigger_anchor,
    extract_customer_signals,
)

load_dotenv()
log = logging.getLogger("vera-bot.generator")

# ── Gemini client setup (google-genai SDK) ───────────────────────────────────
_GEMINI_KEY = os.getenv("GEMINI_API_KEY")
_client = genai.Client(api_key=_GEMINI_KEY) if _GEMINI_KEY else None
_MODEL = "gemini-2.0-flash"


def _call_llm(system_prompt: str, user_prompt: str) -> str:
    """Call Gemini at temperature=0 for determinism."""
    if _client is None:
        raise RuntimeError("GEMINI_API_KEY not found; using rule-based fallback")

    full_prompt = f"{system_prompt}\n\n---\n\n{user_prompt}"
    response = _client.models.generate_content(
        model=_MODEL,
        contents=full_prompt,
        config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=1000,
        ),
    )
    return response.text.strip()


# ─── Composer ────────────────────────────────────────────────────────────────

class MessageGenerator:

    def compose(
        self,
        category: dict,
        merchant: dict,
        trigger: dict,
        customer: Optional[dict] = None,
    ) -> dict:
        """
        Main entry point. Returns {body, cta, send_as, suppression_key, rationale}.
        Raises on LLM failure — caller should use compose_fallback() instead.
        """
        m = extract_merchant_signals(merchant, category)
        t = extract_trigger_anchor(trigger, category)
        c = extract_customer_signals(customer) if customer else {}

        trigger_kind = trigger.get("kind", "generic")
        is_customer_facing = customer is not None or trigger.get("scope") == "customer"
        send_as = "merchant_on_behalf" if is_customer_facing else "vera"

        system_prompt = _build_system_prompt(category, m, send_as)
        user_prompt = _build_user_prompt(trigger_kind, m, t, c, category, merchant, customer)

        raw = _call_llm(system_prompt, user_prompt)
        composed = _parse_output(raw)

        # These fields are always set by us, not the LLM
        composed["send_as"] = send_as
        if not composed.get("suppression_key"):
            composed["suppression_key"] = (
                trigger.get("suppression_key")
                or f"{trigger_kind}:{merchant.get('merchant_id', 'unknown')}"
            )
        composed["cta"] = _validate_cta(composed.get("cta"), trigger_kind, is_customer_facing)

        return composed

    def compose_fallback(self, merchant: dict, trigger: dict, category: Optional[dict] = None) -> dict:
        """
        Pure rule-based fallback — NO LLM call.
        ALWAYS returns a valid, non-empty message.
        Used when Gemini fails or context is missing.
        """
        identity = merchant.get("identity", {})
        perf = merchant.get("performance", {})
        category = category or {}
        offers = [o["title"] for o in merchant.get("offers", []) if o.get("status") == "active"]

        owner = (
            identity.get("owner_first_name")
            or identity.get("name", "there").split()[0]
        )
        views = perf.get("views", 0)
        calls = perf.get("calls", 0)
        kind = trigger.get("kind", "update")

        if kind == "perf_dip":
            customer_aggregate = merchant.get("customer_aggregate", {})
            ctr = perf.get("ctr")
            peer_ctr = (
                perf.get("peer_ctr")
                or category.get("peer_stats", {}).get("avg_ctr")
                or (0.05 if ctr else None)
            )
            lapsed = (
                customer_aggregate.get("lapsed_customers")
                or customer_aggregate.get("lapsed_180d_plus")
                or customer_aggregate.get("lapsed_90d_plus")
                or 0
            )
            offer = offers[0] if offers else "a timely recovery offer"

            if views and ctr and peer_ctr and peer_ctr > ctr:
                lost_leads = max(1, int(views * (peer_ctr - ctr)))
                ctr_line = (
                    f"Your CTR is {ctr * 100:.1f}% vs peer ~{peer_ctr * 100:.1f}%."
                )
                loss_line = f"That gap costs roughly {lost_leads} leads every month."
            elif views and ctr:
                ctr_line = f"You have {views:,} views this month at {ctr * 100:.1f}% CTR."
                loss_line = f"Even 1 extra CTR point is about {max(1, int(views * 0.01))} missed leads every month."
            elif views:
                ctr_line = f"You have {views:,} views this month, but conversion needs attention."
                loss_line = f"A small conversion gap can leak {max(1, int(views * 0.01))} leads every month."
            else:
                ctr_line = "Your profile conversion needs attention this week."
                loss_line = "Every week without action can leak about 10 leads every month."

            lapsed_line = (
                f"Recover {lapsed} lapsed customers with {offer}."
                if lapsed
                else f"Use {offer} to capture unused demand before the dip deepens."
            )
            body = f"{owner}, {ctr_line} {loss_line} {lapsed_line} Act this week. Reply YES"

            return {
                "body": body,
                "cta": "binary_yes_stop",
                "send_as": "vera",
                "suppression_key": (
                    trigger.get("suppression_key")
                    or f"{kind}:{merchant.get('merchant_id', 'unknown')}"
                ),
                "rationale": "Rule-based perf_dip fallback using merchant views, CTR, lapsed customers, and active offer.",
            }

        offer_line = offers[0] if offers else "a simple recovery offer"
        perf_line = f"You have {views:,} views and {calls} calls this month." if views else "Your profile is live on magicpin."
        kind_map = {
            "perf_spike":        "current demand is above baseline",
            "recall_due":        "a due customer can be recovered",
            "festival_upcoming": "festival demand is opening",
            "research_digest":   "new clinical evidence creates patient demand",
            "dormant_with_vera": "your Vera actions are paused while demand still exists",
            "ipl_match_today":   "tonight's match can shift demand to delivery",
            "category_seasonal": "seasonal demand is moving to high-intent items",
            "gbp_unverified":    "unverified GBP is limiting discovery",
            "cde_opportunity":   "this week's CDE slot can unlock patient trust",
        }
        context_line = kind_map.get(kind, "there is unused demand worth acting on")
        lost_leads = max(1, int(views * 0.01)) if views else 10
        lapsed = (
            merchant.get("customer_aggregate", {}).get("lapsed_customers")
            or merchant.get("customer_aggregate", {}).get("lapsed_180d_plus")
            or merchant.get("customer_aggregate", {}).get("lapsed_90d_plus")
            or lost_leads
        )

        body = (
            f"{owner}, {perf_line} but {context_line}. "
            f"That gap can cost about {lost_leads} leads every month. "
            f"Recover {lapsed} lapsed or high-intent customers with {offer_line}. "
            f"Act today. Reply YES"
        )

        return {
            "body": body,
            "cta": "binary_yes_stop",
            "send_as": "vera",
            "suppression_key": (
                trigger.get("suppression_key")
                or f"{kind}:{merchant.get('merchant_id', 'unknown')}"
            ),
            "rationale": (
                f"Rule-based fallback for '{kind}' — Gemini unavailable or context incomplete."
            ),
        }


# ─── System Prompt ────────────────────────────────────────────────────────────

def _build_system_prompt(category: dict, m: dict, send_as: str) -> str:
    voice = category.get("voice", {})
    tone = voice.get("tone", "peer")
    taboos = voice.get("vocab_taboo", [])
    allowed = voice.get("vocab_allowed", [])
    slug = category.get("slug", "")
    peer_stats = category.get("peer_stats", {})

    if m.get("use_hindi"):
        lang = "Use natural Hindi-English code-mix (Hinglish). Mix naturally, don't force full Hindi."
    elif m.get("use_marathi"):
        lang = "Open with a brief Marathi greeting if appropriate, then English body."
    elif m.get("use_telugu"):
        lang = "Primarily English. A brief Telugu greeting is acceptable."
    else:
        lang = "Use clear English."

    if send_as == "merchant_on_behalf":
        role = (
            f"You are Vera, drafting a WhatsApp message ON BEHALF of {m['business_name']} "
            f"to send to one of their customers. "
            f"The message appears to come FROM the merchant, NOT from Vera. "
            f"Do NOT mention Vera or magicpin. "
            f"Sign from: \"{m['business_name']}\" or owner \"{m['owner_name']}\"."
        )
    else:
        role = (
            f"You are Vera, magicpin's AI merchant-growth assistant. "
            f"You are messaging {m['owner_name']} ({m['business_name']}) directly. "
            f"Speak as a knowledgeable peer advisor — not a salesperson, not a bot."
        )

    return f"""You are composing a WhatsApp message for Vera, magicpin's AI assistant.

{role}

CATEGORY: {slug.upper()}
VOICE: {tone}
VOCABULARY ALLOWED: {', '.join(allowed[:8]) if allowed else 'standard industry terms'}
VOCABULARY FORBIDDEN: {', '.join(taboos)} — NEVER use these
PEER BENCHMARKS: {json.dumps(peer_stats)}
{lang}

STRICT RULES:
1. EVERY message MUST follow this structure in order:
   [Problem + contrast] -> [Quantified loss] -> [Recovery opportunity] -> [Time anchor] -> [Binary CTA]
2. The contrast MUST compare current vs potential, e.g. "You're at X; peers/potential is Y."
3. The quantified loss MUST include a number and cadence, e.g. "losing ~60 leads/month."
4. The recovery opportunity MUST name lapsed customers, unused demand, missed walk-ins, refill pool, slots, orders, or another recoverable pool.
5. The time anchor MUST explicitly say "today", "this week", "tonight", "before [date]", or "every month".
6. The final CTA MUST be exactly binary and end with "Reply YES".
7. NO generic phrases: "increase your sales", "grow your business", "boost visibility"
8. USE REAL NUMBERS ONLY. Every claim must cite a number from the provided data.
9. NO preambles — start with the finding or signal immediately.
10. NO fabricated claims — research citations must match the source exactly.
11. 3–5 sentences max. Every sentence must earn its place.
12. TABOO words are absolutely forbidden.
13. CONTRAST FRAMING: "You're at X. Peers are at Y. Closing that gap means Z."

OUTPUT FORMAT — return ONLY valid JSON, no markdown fences, no extra text:
{{
  "body": "the message text",
  "cta": "binary_yes_stop",
  "suppression_key": "dedup-key-string",
  "rationale": "1-2 sentences on why this message and what it achieves"
}}"""


# ─── User Prompt ──────────────────────────────────────────────────────────────

def _build_user_prompt(
    trigger_kind: str,
    m: dict,
    t: dict,
    c: dict,
    category: dict,
    merchant: dict,
    customer: Optional[dict],
) -> str:
    facts = _format_merchant_facts(m, category)
    trigger_facts = _format_trigger_facts(t)
    customer_facts = _format_customer_facts(c) if c else ""
    instruction = _get_trigger_instruction(trigger_kind, m, t, c, category)

    return f"""MERCHANT FACTS:
{facts}

TRIGGER:
{trigger_facts}
{customer_facts}

TASK:
{instruction}

IMPORTANT: Use the real numbers above. Do not invent data. Return only JSON."""


def _format_merchant_facts(m: dict, category: dict) -> str:
    lines = [
        f"Business: {m['business_name']} | Owner: {m['owner_name']} | City: {m['city']}, {m['locality']}",
        f"Subscription: {m['sub_status']} | Days remaining: {m['days_remaining']}",
    ]

    if m.get("views_30d"):
        lines.append(
            f"Performance (30d): Views={m['views_30d']}, Calls={m['calls_30d']}, CTR={m['ctr']:.3f}"
        )
    if m.get("peer_ctr"):
        gap = abs(m.get("ctr_gap_pct", 0) or 0)
        direction = "BELOW" if m.get("ctr_below_peer") else "ABOVE"
        lines.append(f"Peer benchmark CTR: {m['peer_ctr']:.3f} | Merchant is {gap:.1f}% {direction} peer")
    if m.get("views_delta_pct") is not None:
        v = m["views_delta_pct"]
        sign = "+" if v > 0 else ""
        lines.append(f"7d trend: Views {sign}{v*100:.0f}%, Calls {(m.get('calls_delta_pct') or 0)*100:.0f}%")

    if m.get("active_offers"):
        lines.append(f"Active offers: {', '.join(m['active_offers'])}")
    elif m.get("has_no_active_offers"):
        lines.append("Active offers: NONE")

    if m.get("total_unique_ytd"):
        lines.append(f"Customers YTD: {m['total_unique_ytd']} | Lapsed: {m.get('lapsed_count', 'N/A')}")
    if m.get("high_risk_adult_count"):
        lines.append(f"High-risk adult patients: {m['high_risk_adult_count']}")
    if m.get("delivery_orders_30d"):
        lines.append(
            f"Delivery orders (30d): {m['delivery_orders_30d']} | Dine-in: {m.get('dine_in_orders_30d', 'N/A')}"
        )
    if m.get("signal_flags"):
        lines.append(f"Signals: {', '.join(m['signal_flags'])}")
    if m.get("pos_review_themes"):
        pos = m["pos_review_themes"][0]
        lines.append(f"Top positive review theme: '{pos[0]}' ({pos[1]} mentions)")
    if m.get("neg_review_themes"):
        neg = m["neg_review_themes"][0]
        lines.append(f"Top negative review theme: '{neg[0]}' ({neg[1]} mentions)")

    # Pre-computed contrast deltas
    contrasts = []
    peer_stats = category.get("peer_stats", {})
    if m.get("ctr") and m.get("peer_ctr") and m.get("ctr_below_peer"):
        views = m.get("views_30d", 0)
        if views:
            missed = int(views * (m["peer_ctr"] - m["ctr"]))
            contrasts.append(
                f"CTR gap: You convert {m['ctr']:.3f}; peers convert {m['peer_ctr']:.3f}. "
                f"At peer CTR your {views:,} views → ~{missed:+d} more leads/month."
            )
    peer_ret = peer_stats.get("retention_6mo_pct") or peer_stats.get("retention_30d_pct")
    merch_ret = m.get("retention_pct")
    if merch_ret and peer_ret and merch_ret < peer_ret:
        contrasts.append(
            f"Retention gap: Your 6mo retention {merch_ret*100:.0f}% vs peer {peer_ret*100:.0f}%. "
            f"Your {m.get('lapsed_count', 0)} lapsed are the recoverable pool."
        )
    peer_views = peer_stats.get("avg_views_30d")
    merch_views = m.get("views_30d", 0)
    if peer_views and merch_views and merch_views < peer_views:
        contrasts.append(
            f"Visibility gap: {merch_views:,} views/month vs peer avg {peer_views:,}. "
            f"GBP verification or posts could close this 20–30%."
        )
    if contrasts:
        lines.append("\nCONTRAST (use in message framing):")
        for con in contrasts:
            lines.append(f"  • {con}")

    return "\n".join(lines)


def _format_customer_facts(c: dict) -> str:
    if not c:
        return ""
    lines = [
        "\nCUSTOMER FACTS:",
        f"Name: {c.get('customer_name', 'Customer')} | Language: {c.get('language_pref', 'en')}",
        f"State: {c.get('state', 'active')} | Total visits: {c.get('visits_total', 0)}",
    ]
    if c.get("months_since_last_visit"):
        lines.append(f"Months since last visit: {c['months_since_last_visit']}")
    if c.get("services_received"):
        lines.append(f"Services received: {', '.join(c['services_received'][-3:])}")
    if c.get("preferred_slots"):
        lines.append(f"Preferred time: {c['preferred_slots']}")
    return "\n".join(lines)


def _format_trigger_facts(t: dict) -> str:
    lines = [f"Kind: {t['kind']} | Urgency: {t.get('urgency', 2)}/5"]

    if di := t.get("digest_item"):
        lines.append(f"Digest item: \"{di.get('title', '')}\"")
        if di.get("source"):
            lines.append(f"Source: {di['source']}")
        if di.get("trial_n"):
            lines.append(f"Trial size: {di['trial_n']:,} participants")
        if di.get("patient_segment"):
            lines.append(f"Patient segment: {di['patient_segment']}")
        if di.get("summary"):
            lines.append(f"Summary: {di['summary'][:200]}")

    if t.get("teams"):
        is_wknt = t.get("is_weeknight", True)
        occasion = "weeknight (delivery boost expected)" if is_wknt else "weekend (mixed dine-in + delivery)"
        lines.append(
            f"IPL Match: {t['teams']} | Venue: {t.get('venue','')} | "
            f"Time: {t.get('match_time_display', t.get('match_time_iso',''))} | {occasion}"
        )
    if t.get("molecule_list"):
        mols = ", ".join(t["molecule_list"])
        days_left = t.get("days_until_stockout")
        runs_out = t.get("stock_runs_out_iso", "")[:10]
        lines.append(f"Medicines due: {mols}")
        if days_left is not None:
            lines.append(f"Stock runs out: {runs_out} ({days_left} days) — URGENT")
        lines.append(
            "Delivery address saved — can dispatch without customer action"
            if t.get("delivery_address_saved")
            else "Delivery address NOT saved — customer must confirm"
        )
    elif t.get("is_placeholder"):
        lines.append("chronic_refill_due: placeholder — use patient relationship data")

    if t.get("demand_gains"):
        lines.append("Summer demand RISING: " + " | ".join(
            f"{g['item'].upper()} +{g['pct']}%" for g in t["demand_gains"]
        ))
    if t.get("demand_drops"):
        lines.append("Summer demand FALLING: " + " | ".join(
            f"{d['item'].upper()} -{d['pct']}%" for d in t["demand_drops"]
        ))
    if t.get("shelf_action_recommended"):
        lines.append("Action needed: Restock front shelf + update GBP highlights for seasonal SKUs")
    if t.get("uplift_display"):
        lines.append(f"GBP verified: NO | Est. uplift: {t['uplift_display']} more views")
        lines.append(f"Verification path: {t.get('verification_path','postcard_or_phone_call')}")
    if t.get("deadline"):
        lines.append(f"Compliance deadline: {t['deadline']}")
    if t.get("festival_name"):
        lines.append(
            f"Festival: {t['festival_name']} in {t.get('days_away','?')} days ({t.get('festival_date','')})"
        )
    if t.get("available_slots"):
        slots = [s.get("label", "") for s in t["available_slots"][:2]]
        lines.append(f"Available slots: {', '.join(slots)}")
    if t.get("service_due"):
        lines.append(
            f"Service due: {t['service_due']} | Last: {t.get('last_service_date','')} | Due: {t.get('due_date','')}"
        )
    if t.get("batch_numbers"):
        lines.append(f"Recall batches: {', '.join(t['batch_numbers'])} | Molecule: {t.get('molecule','')}")
        if t.get("affected_count"):
            lines.append(f"Affected customers: {t['affected_count']}")
    if t.get("distance_km"):
        lines.append(f"Competitor: {t['distance_km']} km away | Type: {t.get('competitor_type','')}")
    if t.get("milestone_type"):
        lines.append(f"Milestone: {t['milestone_type']} = {t.get('value','')}")
    if t.get("delta_pct") is not None:
        lines.append(f"Performance change: {t['delta_pct']*100:+.0f}% on {t.get('metric','views')}")
    if t.get("context_note"):
        lines.append(f"Context: {t['context_note']}")
    if t.get("intent"):
        lines.append(f"Merchant intent: {t['intent']}")
    if t.get("last_merchant_message"):
        lines.append(f"Last merchant said: \"{t['last_merchant_message']}\"")

    return "\n".join(lines)


def _get_trigger_instruction(kind: str, m: dict, t: dict, c: dict, category: dict) -> str:
    slug = category.get("slug", "")
    name = m["owner_name"]
    biz = m["business_name"]

    instructions = {
        "research_digest": f"""
Compose a message to {name} sharing the digest item above.
STRUCTURE:
1. FINDING: Open with the specific trial result + number
2. CONTRAST: Anchor to their practice — use their high_risk_adult_count or patient aggregate
3. MISSED-OPPORTUNITY: Frame what happens if they don't act (preventable decay, lost revenue)
4. CTA: "Should I pull the abstract + draft a 4-line patient WhatsApp? Reply YES."
- Cite source exactly. Tone: peer-clinical. CTA must end "Reply YES".
""",
        "regulation_change": f"""
Compose an urgent compliance message to {name}.
- Lead with the regulation name and deadline date
- State exactly what action is required
- CTA: "Should I draft the compliance checklist now? Reply YES."
""",
        "perf_spike": f"""
Compose a message capitalizing on {name}'s performance spike.
- State the spike metric and exact percentage from the data
- Recommend one specific NOW action (post, offer push)
- CTA: "Should I draft the post and push the offer? Reply YES."
""",
        "perf_dip": f"""
Compose a message about {name}'s performance dip.
- Name the metric and exact drop percentage
- Use the CONTRAST block to show the gap vs peers
- Give ONE specific fix
- CTA: "Should I fix this right now? Reply YES."
""",
        "seasonal_perf_dip": f"""
Compose a message reframing the seasonal dip for {name}.
- State the real dip, then reframe: 'this is the normal [season] pattern'
- Give the peer seasonal range so they know it's normal
- Pivot to a retention strategy
- CTA: "Should I draft a retention campaign? Reply YES."
""",
        "recall_due": f"""
Compose a recall message FROM {biz} TO {c.get('customer_name', 'the customer')}.
- Open with customer name, state recall is due with exact available slots
- Include active offer price if available
- CTA: "Reply 1 for [slot1], 2 for [slot2]" or "Reply YES to confirm"
- Language: match customer preference.
""",
        "festival_upcoming": f"""
Compose a festival message for {name}.
- Name the festival and exact days-until
- Tie their ACTIVE OFFER to the festival moment (don't invent offers)
- CTA: "Should I push this now? Reply YES."
""",
        "curious_ask_due": f"""
Compose a single focused question to {name} about their business this week.
- Immediately offer a specific deliverable in return
- CTA: one clear, low-stakes YES/NO
""",
        "supply_alert": f"""
Compose an urgent alert to {name}.
- Lead with batch numbers, molecule, and exact affected customer count
- Offer to draft the customer notification immediately
- CTA: "Reply YES to get the draft now."
""",
        "dormant_with_vera": f"""
Compose a re-engagement message to {name}.
- No guilt. Open with one specific signal from their data (CTR gap, stale posts)
- Offer one small, useful action
- CTA: "One action, 5 seconds. Reply YES."
""",
        "competitor_opened": f"""
Compose a message about a new competitor near {name}.
- Acknowledge it factually, pivot immediately to {biz}'s advantages
- Suggest one specific differentiation action
- CTA: "Should I handle this? Reply YES."
""",
        "milestone_reached": f"""
Compose a congratulations + next-milestone message for {name}.
- Celebrate briefly (1 sentence), pivot to what's next immediately
- CTA: "Should I help with the next step? Reply YES."
""",
        "customer_lapsed_soft": f"""
Compose a warm win-back message FROM {biz} TO {c.get('customer_name', 'the customer')}.
- Reference their past service(s) warmly, offer something concrete
- CTA: "Reply YES — no commitment, no auto-charge."
""",
        "customer_lapsed_hard": f"""
Compose a win-back message FROM {biz} TO {c.get('customer_name', 'the customer')}.
- Brief, no judgment. Lead with a no-commitment offer
- CTA: "Reply YES — one tap, we'll take it from there."
""",
        "appointment_tomorrow": f"""
Compose an appointment reminder FROM {biz} TO {c.get('customer_name', 'the customer')}.
- State time and service clearly
- CTA: "Reply CONFIRM or reply RESCHEDULE."
""",
        "active_planning_intent": f"""
{name} has signalled a planning intent. Skip re-qualifying.
Deliver the CONCRETE artifact they asked about immediately.
- End with: "Edit this as needed — should I post it? Reply YES."
""",
        "ipl_match_today": f"""
Compose a message to {name} about tonight's IPL match.
STRUCTURE:
1. Name the match ({t.get('teams','DC vs MI')}) and time ({t.get('match_time_display','7:30 PM')})
2. CONTRARIAN INSIGHT: Weekend IPL = -12% dine-in near venue.
   "Your {m.get('dine_in_orders_30d','~95')}/month dine-in will dip tonight.
   But delivery runs +18-22% on match nights — you're already at {m.get('delivery_orders_30d','180')} orders/month."
3. Tie their ACTIVE OFFER to the delivery moment
4. CTA: "Should I push your offer for delivery tonight? Reply YES."
""",
        "chronic_refill_due": f"""
Compose a refill reminder FROM {biz} TO {c.get('customer_name', 'the customer')}.
- Name the specific medicines: {', '.join(t.get('molecule_list', []))}
- State stock runs out on {t.get('stock_runs_out_iso', 'soon')[:10]}
- If delivery address is saved: offer to send without them leaving home
- CTA: "Reply YES — we'll dispatch before [date]."
- Language: match customer preference.
""",
        "category_seasonal": f"""
Compose a message to {name} about the summer demand shift.
STRUCTURE:
1. Lead with the 2 biggest gains from trigger facts (exact percentages only)
2. CONTRAST: "Your {m.get('views_30d',1850)} views/month — if 10% look for ORS or sunscreen,
   that's ~{int(m.get('views_30d',1850)*0.10)} walk-ins you capture only with the right shelf + GBP"
3. WINDOW: "This peaks May-June — repositioning this week vs next month = 4-6 weeks of capture"
4. CTA: "Should I draft the GBP update + a summer bundle WhatsApp? Reply YES."
""",
        "gbp_unverified": f"""
Compose a message to {name} about their unverified Google Business Profile.
- Lead with the cost: {t.get('uplift_display','30%')} fewer views than verified competitors
- Show what verified would mean: their {m.get('views_30d','N/A')} views * 1.3
- Verification takes 5 minutes via {t.get('verification_path','postcard or phone call')}
- CTA: "Takes 5 minutes. Should I walk you through it? Reply YES."
""",
        "cde_opportunity": f"""
Compose a message to {name} about the CDE/webinar opportunity.
- Use the digest item: title, date, credits, fee
- Anchor to their patient mix
- CTA: "Want the registration link + front-desk note? Reply YES."
""",
    }

    category_fallbacks = {
        "dentists": f"Compose a message to {name} using their CTR vs peer or patient aggregate. Lever: CURIOSITY. CTA: Reply YES.",
        "salons": f"Compose a message to {name} using a seasonal or slot-fill signal. Lever: TIME-BOUND. CTA: Reply YES.",
        "restaurants": f"Compose a message to {name} using delivery vs dine-in performance. Lever: LOSS AVERSION. CTA: Reply YES.",
        "gyms": f"Compose a message to {name} using retention or lapsed-member data. Lever: SOCIAL PROOF. CTA: Reply YES.",
        "pharmacies": f"Compose a message to {name} using repeat rate or seasonal demand. Lever: URGENCY. CTA: Reply YES.",
    }

    return instructions.get(
        kind,
        category_fallbacks.get(
            slug,
            f"Compose a specific message to {name} at {biz}. "
            f"Use one real signal from their data. Apply urgency or loss-aversion framing. "
            f"CTA: Reply YES.",
        ),
    )


# ─── Output Parsing ───────────────────────────────────────────────────────────

def _parse_output(raw: str) -> dict:
    text = raw.strip()
    # Strip markdown fences
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("```").strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r'\{[^{}]*"body"[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass

    log.warning(f"Could not parse Gemini output as JSON. First 300 chars: {raw[:300]}")
    return {
        "body": raw[:600] if raw else "Unable to compose message.",
        "cta": "binary_yes_stop",
        "suppression_key": "",
        "rationale": "Fallback: raw Gemini output used as body",
    }


def _validate_cta(cta: str, trigger_kind: str, is_customer_facing: bool) -> str:
    valid = {"open_ended", "binary_yes_stop", "binary_yes_no", "slot_selection", "none"}
    if cta in valid:
        return cta
    if trigger_kind in ("recall_due", "appointment_tomorrow"):
        return "slot_selection"
    return "binary_yes_stop"
