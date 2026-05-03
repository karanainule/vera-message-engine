"""
SignalExtractor — extracts structured, verifiable facts from context objects.

These facts are injected into the LLM prompt as a structured "facts block"
that the model must use. This ensures specificity (real numbers, real dates)
and prevents hallucination (model can only use what we give it).
"""

from typing import Optional
from datetime import datetime, timezone


def extract_merchant_signals(merchant: dict, category: dict) -> dict:
    """
    Returns a structured dict of merchant-specific facts for prompt injection.
    Every value here is verifiable from the input data.
    """
    identity = merchant.get("identity", {})
    perf = merchant.get("performance", {})
    sub = merchant.get("subscription", {})
    cust_agg = merchant.get("customer_aggregate", {})
    signals = merchant.get("signals", [])
    offers = merchant.get("offers", [])
    reviews = merchant.get("review_themes", [])
    peer_stats = category.get("peer_stats", {})

    # Active offers only
    active_offers = [o for o in offers if o.get("status") == "active"]
    expired_offers = [o for o in offers if o.get("status") == "expired"]

    # CTR gap vs peer
    merchant_ctr = perf.get("ctr", 0)
    peer_ctr = peer_stats.get("avg_ctr", 0)
    ctr_gap_pct = None
    if peer_ctr and merchant_ctr:
        ctr_gap_pct = round((peer_ctr - merchant_ctr) / peer_ctr * 100, 1)

    # Performance deltas
    delta = perf.get("delta_7d", {})
    views_delta_pct = delta.get("views_pct")
    calls_delta_pct = delta.get("calls_pct")

    # Language
    languages = identity.get("languages", ["en"])
    use_hindi = "hi" in languages
    use_marathi = "mr" in languages
    use_telugu = "te" in languages

    # Subscription urgency
    days_remaining = sub.get("days_remaining")
    sub_status = sub.get("status", "active")
    renewal_urgent = days_remaining is not None and days_remaining <= 14

    # Review sentiment
    neg_reviews = [r for r in reviews if r.get("sentiment") == "neg"]
    pos_reviews = [r for r in reviews if r.get("sentiment") == "pos"]

    # Signal flags
    signal_flags = set(s.split(":")[0] for s in signals)

    return {
        "owner_name": identity.get("owner_first_name", identity.get("name", "").split()[0]),
        "business_name": identity.get("name", ""),
        "city": identity.get("city", ""),
        "locality": identity.get("locality", ""),
        "verified": identity.get("verified", False),
        "languages": languages,
        "use_hindi": use_hindi,
        "use_marathi": use_marathi,
        "use_telugu": use_telugu,
        # Performance
        "views_30d": perf.get("views"),
        "calls_30d": perf.get("calls"),
        "directions_30d": perf.get("directions"),
        "ctr": merchant_ctr,
        "peer_ctr": peer_ctr,
        "ctr_gap_pct": ctr_gap_pct,
        "ctr_below_peer": merchant_ctr < peer_ctr if peer_ctr else False,
        "views_delta_pct": views_delta_pct,
        "calls_delta_pct": calls_delta_pct,
        # Subscription
        "sub_status": sub_status,
        "days_remaining": days_remaining,
        "renewal_urgent": renewal_urgent,
        # Offers
        "active_offers": [o["title"] for o in active_offers],
        "expired_offers": [o["title"] for o in expired_offers],
        "has_no_active_offers": len(active_offers) == 0,
        # Customer aggregate
        "total_unique_ytd": cust_agg.get("total_unique_ytd"),
        "lapsed_count": cust_agg.get("lapsed_180d_plus") or cust_agg.get("lapsed_90d_plus"),
        "retention_pct": cust_agg.get("retention_6mo_pct") or cust_agg.get("retention_3mo_pct"),
        "high_risk_adult_count": cust_agg.get("high_risk_adult_count"),
        "delivery_orders_30d": cust_agg.get("delivery_orders_30d"),
        "dine_in_orders_30d": cust_agg.get("dine_in_orders_30d"),
        # Reviews
        "neg_review_themes": [(r["theme"], r.get("occurrences_30d", 0)) for r in neg_reviews],
        "pos_review_themes": [(r["theme"], r.get("occurrences_30d", 0)) for r in pos_reviews],
        # Signal flags
        "signal_flags": list(signal_flags),
        "is_dormant": "dormant_with_vera" in signal_flags,
        "ctr_below_peer_flag": "ctr_below_peer_median" in signal_flags,
        "stale_posts": any("stale_posts" in s for s in signals),
        "is_new_merchant": "new_merchant" in signal_flags,
        "winback_eligible": "winback_eligible" in signal_flags,
    }


def extract_trigger_anchor(trigger: dict, category: dict) -> dict:
    """
    Returns the specific fact that anchors the 'why now' for this trigger.
    Cross-references trigger payload with category digest.
    """
    kind = trigger.get("kind", "")
    payload = trigger.get("payload", {})
    anchor = {
        "kind": kind,
        "urgency": trigger.get("urgency", 2),
        "suppression_key": trigger.get("suppression_key", ""),
    }

    # Look up digest item if trigger references one
    top_item_id = payload.get("top_item_id")
    if top_item_id:
        digest = category.get("digest", [])
        for item in digest:
            if item.get("id") == top_item_id:
                anchor["digest_item"] = item
                break

    # Extract kind-specific payload data
    if kind == "research_digest":
        anchor["category"] = payload.get("category", "")
    elif kind == "regulation_change":
        anchor["deadline"] = payload.get("deadline_iso", "")
        anchor["circular_ref"] = payload.get("circular_ref", "")
    elif kind in ("perf_spike", "perf_dip", "seasonal_perf_dip"):
        anchor["metric"] = payload.get("metric", "views")
        anchor["delta_pct"] = payload.get("delta_pct")
        anchor["context_note"] = payload.get("context_note", "")
    elif kind == "recall_due":
        anchor["service_due"] = payload.get("service_due", "")
        anchor["last_service_date"] = payload.get("last_service_date", "")
        anchor["due_date"] = payload.get("due_date", "")
        anchor["available_slots"] = payload.get("available_slots", [])
    elif kind == "festival_upcoming":
        anchor["festival_name"] = payload.get("festival_name", "")
        anchor["days_away"] = payload.get("days_away")
        anchor["festival_date"] = payload.get("festival_date", "")
    elif kind in ("customer_lapsed_soft", "customer_lapsed_hard"):
        anchor["days_since_visit"] = payload.get("days_since_visit")
        anchor["last_service"] = payload.get("last_service", "")
    elif kind == "supply_alert":
        anchor["batch_numbers"] = payload.get("batch_numbers", [])
        anchor["molecule"] = payload.get("molecule", "")
        anchor["affected_count"] = payload.get("affected_customer_count")
    elif kind == "competitor_opened":
        anchor["distance_km"] = payload.get("distance_km")
        anchor["competitor_type"] = payload.get("competitor_type", "")
    elif kind == "milestone_reached":
        anchor["milestone_type"] = payload.get("milestone_type", "")
        anchor["value"] = payload.get("value")
    elif kind == "ipl_match_today":
        # FIX: correct field names from actual payload schema
        anchor["teams"] = payload.get("match", "")            # payload key is 'match', not 'teams'
        anchor["venue"] = payload.get("venue", "")
        anchor["city"] = payload.get("city", "")
        anchor["match_time_iso"] = payload.get("match_time_iso", "")
        anchor["is_weeknight"] = payload.get("is_weeknight", True)
        # Derive human-readable match time
        raw_time = payload.get("match_time_iso", "")
        if raw_time:
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(raw_time)
                anchor["match_time_display"] = dt.strftime("%I:%M %p")
            except Exception:
                anchor["match_time_display"] = raw_time
        # Key insight: weeknight match = delivery opportunity, weekend = mixed (some go out)
        anchor["occasion_type"] = "delivery_boost" if payload.get("is_weeknight", True) else "mixed_dine_delivery"
    elif kind == "chronic_refill_due":
        # FIX: was completely missing — add full payload extraction
        anchor["molecule_list"] = payload.get("molecule_list", [])
        anchor["last_refill"] = payload.get("last_refill", "")
        anchor["stock_runs_out_iso"] = payload.get("stock_runs_out_iso", "")
        anchor["delivery_address_saved"] = payload.get("delivery_address_saved", False)
        # Derive urgency days
        runs_out = payload.get("stock_runs_out_iso", "")
        if runs_out:
            try:
                from datetime import datetime, timezone
                exp = datetime.fromisoformat(runs_out.replace("Z", "+00:00"))
                now = datetime.now(timezone.utc)
                days_left = max(0, (exp - now).days)
                anchor["days_until_stockout"] = days_left
            except Exception:
                anchor["days_until_stockout"] = None
        # Placeholder payload handling (T08 expanded trigger has no real payload)
        anchor["is_placeholder"] = payload.get("placeholder", False)
    elif kind == "category_seasonal":
        # FIX: was completely missing — extract seasonal trend data
        anchor["season"] = payload.get("season", "")
        anchor["trends"] = payload.get("trends", [])
        anchor["shelf_action_recommended"] = payload.get("shelf_action_recommended", False)
        # Parse trends into readable gains/drops
        gains, drops = [], []
        for trend in payload.get("trends", []):
            if "_+" in trend:
                parts = trend.rsplit("_+", 1)
                gains.append({"item": parts[0].replace("_demand", "").replace("_", " "), "pct": int(parts[1])})
            elif "_-" in trend:
                parts = trend.rsplit("_-", 1)
                drops.append({"item": parts[0].replace("_demand", "").replace("_", " "), "pct": int(parts[1])})
        anchor["demand_gains"] = gains   # e.g. [{"item": "ORS", "pct": 40}]
        anchor["demand_drops"] = drops
    elif kind == "gbp_unverified":
        # FIX: was completely missing — extract GBP verification data
        anchor["verified"] = payload.get("verified", False)
        anchor["verification_path"] = payload.get("verification_path", "postcard_or_phone_call")
        anchor["estimated_uplift_pct"] = payload.get("estimated_uplift_pct", 0)
        # Convert uplift to a human-readable percentage
        uplift = payload.get("estimated_uplift_pct", 0)
        anchor["uplift_display"] = f"{int(uplift * 100)}%" if uplift else "30%"
    elif kind == "cde_opportunity":
        # FIX: was completely missing — look up digest item using 'digest_item_id' (not 'top_item_id')
        digest_item_id = payload.get("digest_item_id", "")
        anchor["digest_item_id"] = digest_item_id
        anchor["credits"] = payload.get("credits", 0)
        anchor["fee"] = payload.get("fee", "")
        if digest_item_id:
            digest = category.get("digest", [])
            for item in digest:
                if item.get("id") == digest_item_id:
                    anchor["digest_item"] = item   # same key as research_digest uses
                    break
    elif kind == "appointment_tomorrow":
        anchor["appointment_time"] = payload.get("appointment_time", "")
        anchor["service"] = payload.get("service", "")
    elif kind == "active_planning_intent":
        anchor["intent"] = payload.get("intent_summary", "")
        anchor["last_merchant_message"] = payload.get("last_merchant_message", "")
    elif kind == "curious_ask_due":
        anchor["cadence"] = payload.get("cadence", "weekly")
        anchor["last_asked"] = payload.get("last_asked", "")
    elif kind == "perf_spike":
        anchor["spike_metric"] = payload.get("metric", "views")
        anchor["spike_pct"] = payload.get("delta_pct", 0)

    return anchor


def extract_customer_signals(customer: dict) -> dict:
    """Extracts facts for customer-facing messages."""
    if not customer:
        return {}
    identity = customer.get("identity", {})
    rel = customer.get("relationship", {})
    prefs = customer.get("preferences", {})
    consent = customer.get("consent", {})
    state = customer.get("state", "active")

    # Compute months since last visit for lapse framing
    last_visit = rel.get("last_visit")
    months_since = None
    if last_visit:
        try:
            lv = datetime.fromisoformat(last_visit.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            diff = now - lv
            months_since = round(diff.days / 30)
        except Exception:
            pass

    return {
        "customer_name": identity.get("name", ""),
        "language_pref": identity.get("language_pref", "en"),
        "use_hindi_mix": "hi" in identity.get("language_pref", ""),
        "state": state,
        "visits_total": rel.get("visits_total", 0),
        "last_visit": last_visit,
        "months_since_last_visit": months_since,
        "first_visit": rel.get("first_visit"),
        "services_received": rel.get("services_received", []),
        "preferred_slots": prefs.get("preferred_slots", ""),
        "channel": prefs.get("channel", "whatsapp"),
        "consent_scope": consent.get("scope", []),
        "is_lapsed_soft": state == "lapsed_soft",
        "is_lapsed_hard": state == "lapsed_hard",
        "is_new": state == "new",
    }
