# Vera Message Engine 🚀

**Team**: Karan Ainule
**Model**: Google Gemini (`gemini-2.0-flash`, temperature=0 for determinism)
**Version**: 2.0.0

---

## 🧠 Overview

Vera Message Engine is an AI-powered backend system that generates **high-conversion, context-aware messages** for merchants based on real-time triggers.

The system combines:

* Rule-based decision logic
* Structured signal extraction
* Gemini LLM generation
* Multi-layer fallback guarantees

👉 **Key Guarantee**: `/v1/tick` never returns an empty response.

---

## 🏗 Architecture

```
ContextStore (versioned, in-memory)
    ↓
DecisionEngine (trigger mapping + action selection)
    ↓
SignalExtractor (CTR gap, lapsed users, offers, etc.)
    ↓
MessageGenerator (Gemini + rule-based fallback)
    ↓
ReplyHandler (intent detection + conversation flow)
```

---

## ⚙️ Core Design Decisions

### 1. Trigger → Merchant Mapping

* Event strings like `"perf_dip"` are automatically mapped to the **latest merchant context**
* No need to pass full objects in `/tick`
* Ensures simplicity + robustness

---

### 2. Deterministic AI Output

* Gemini runs at **temperature = 0**
* Same input → same output (important for evaluation)

---

### 3. Structured Message Framework

Every message follows:

```
[Problem + Contrast]
→ [Quantified Loss]
→ [Recovery Opportunity]
→ [Time Anchor]
→ [Reply YES CTA]
```

Example:

> “Your CTR is 2% vs ~5% nearby clinics — that’s ~60 missed patients every month. You already have 120 lapsed patients ready to convert. Activate your ₹299 checkup today and recover them fast. Reply YES”

---

### 4. Multi-layer Fallback System (CRITICAL)

The system guarantees output using 3 layers:

1. Gemini LLM
2. Rule-based fallback (`compose_fallback`)
3. Emergency fallback (hardcoded safe message)

👉 Even if API fails → system still responds

---

### 5. Context Versioning

* Stored as `(scope, context_id, version)`
* New versions overwrite atomically
* Always uses latest data

---

## 🚫 Anti-Failure Guarantees

* ❌ Never returns `{"actions": []}`
* ❌ No dependency on LLM success
* ❌ No hallucinated data (uses extracted signals)
* ❌ No multiple CTAs

---

## 🔧 Tech Stack

* Python
* FastAPI
* Google Gemini API (`google-genai`)
* Python-dotenv

---

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Setup environment

Create `.env`:

```env
GEMINI_API_KEY=your_api_key_here
```

---

### 3. Run server

```bash
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

---

### 4. Open Swagger UI

```
http://127.0.0.1:8000/docs
```

---

## 📡 API Endpoints

* `GET /v1/healthz` → health check
* `GET /v1/metadata` → system info
* `POST /v1/context` → push merchant/customer context
* `POST /v1/tick` → generate actions (core logic)
* `POST /v1/reply` → simulate conversation reply
* `POST /v1/teardown` → reset state

---

## 🧪 Example Flow

### 1. Load context

```json
{
  "scope": "merchant",
  "context_id": "m_001",
  "version": 1,
  "payload": { ... }
}
```

---

### 2. Trigger decision

```json
{
  "now": "2026-05-03T10:05:00Z",
  "events": ["perf_dip"]
}
```

---

### 3. Output

```json
{
  "actions": [
    {
      "message": "Your CTR is 2% vs ~5% nearby clinics — that’s ~60 missed patients every month...",
      "cta": "Reply YES"
    }
  ]
}
```

---

## 🎯 Key Strengths

* Data-driven messaging (CTR, lapsed users, offers)
* High engagement design (urgency + contrast + CTA)
* Robust fallback system
* Deterministic outputs
* Clean API design

---

## 📈 Future Improvements

* City-specific peer benchmarks
* Multi-language personalization (Hindi/Marathi)
* Persistent database (Redis/Postgres)
* Advanced user segmentation

---

## 💬 Final Note

This system is designed not just to work — but to **drive action**.

👉 Every message is engineered for:

* clarity
* urgency
* conversion

---
