# Vera Message Engine — Submission README

**Team**: Karan Ainule  
**Model**: claude-sonnet-4-5-20251001 (temperature=0 for determinism)  
**Version**: 1.0.0

---

## Approach

### Architecture
```
ContextStore (in-memory, idempotent versioning)
    ↓
DecisionEngine (trigger ranking, per-tick action selection)
    ↓
SignalExtractor (pulls verifiable facts: CTR gaps, counts, dates, citations)
    ↓
MessageGenerator (trigger-kind router → LLM at temperature=0 → CTA validator)
    ↓
ReplyHandler (auto-reply detection, intent transitions, graceful exit)
```

### Core Design Decisions

**1. Trigger-kind routing over generic prompts**  
Each trigger kind (`research_digest`, `perf_dip`, `recall_due`, etc.) maps to a distinct prompt instruction. This ensures category voice, compulsion lever, and CTA shape are all tuned to the specific situation — not just "compose a message about X."

**2. Signal extraction before prompting**  
Before calling the LLM, `SignalExtractor` pulls structured facts: CTR gap vs peer median, exact customer counts, active offers, review themes, seasonal context. These are injected as a grounded "facts block" the model must use. This prevents hallucination and guarantees specificity.

**3. Determinism via temperature=0**  
Same context version → same output every time. Context is stored by `(scope, context_id, version)` — idempotent, atomic updates.

**4. Multi-turn conversation handling**  
`ReplyHandler` maintains per-conversation state with: auto-reply pattern detection (verbatim match + repetition), accept-intent detection (30+ Hindi/English signals), graceful exit on 3rd unanswered nudge or explicit rejection, and off-topic redirect (politely stays on mission).

**5. Adaptation to injected context**  
When `/v1/context` receives a higher version (new digest items, updated perf), the store atomically replaces. Next `/v1/tick` will use the fresh context automatically — the generator always reads from store, never caches.

---

## What tradeoffs were made

- **In-memory store**: No Redis/SQLite — fast, simple, sufficient for 60-min test window
- **Single LLM call per compose**: No chained calls. Structured facts block + specific instruction = one good call rather than two mediocre ones
- **Suppression at DecisionEngine level**: Once a suppression key is sent, it won't fire again in the same session. Conservative but prevents spam penalties
- **20-action tick cap respected**: Engine stops at 20 actions per tick even if more triggers are available

---

## What additional context would have helped most

1. **Real seasonal beat dates** — knowing which specific Indian festivals fall in the next 30 days would sharpen festival trigger messages
2. **Actual peer stats per city** — the dataset has Delhi-scoped peer CTR; city-specific peer stats for Mumbai, Bangalore, Hyderabad, etc. would improve comparative framing
3. **Merchant's actual WhatsApp conversation history** — the seed data has 1-2 turns; more history would improve reply continuity

---

## Deployment

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8080
```

Environment: `ANTHROPIC_API_KEY` must be set.

## Endpoints
- `GET /v1/healthz`
- `GET /v1/metadata`
- `POST /v1/context`
- `POST /v1/tick`
- `POST /v1/reply`
- `POST /v1/teardown`
