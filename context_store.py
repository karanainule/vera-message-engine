"""
ContextStore — in-memory, idempotent, versioned.

Stores all 4 context scopes: category, merchant, customer, trigger.
Versioning: same (scope, context_id) with lower or equal version = no-op.
Higher version atomically replaces.
"""

from typing import Optional, Literal
import threading

VALID_SCOPES = {"category", "merchant", "customer", "trigger"}


class ContextStore:
    def __init__(self):
        self._lock = threading.Lock()
        # (scope, context_id) → {version: int, payload: dict}
        self._store: dict[tuple[str, str], dict] = {}

    def upsert(
        self,
        scope: str,
        context_id: str,
        version: int,
        payload: dict,
    ) -> Literal["stored", "stale", "invalid_scope"]:
        if scope not in VALID_SCOPES:
            return "invalid_scope"
        key = (scope, context_id)
        with self._lock:
            existing = self._store.get(key)
            if existing and existing["version"] >= version:
                return "stale"
            self._store[key] = {"version": version, "payload": payload}
        return "stored"

    def get(self, scope: str, context_id: str) -> Optional[dict]:
        key = (scope, context_id)
        record = self._store.get(key)
        return record["payload"] if record else None

    def get_version(self, scope: str, context_id: str) -> Optional[int]:
        key = (scope, context_id)
        record = self._store.get(key)
        return record["version"] if record else None

    def get_all(self, scope: str) -> dict[str, dict]:
        """Return {context_id: payload} for all items of a given scope."""
        result = {}
        for (s, cid), record in self._store.items():
            if s == scope:
                result[cid] = record["payload"]
        return result

    def get_latest(self, scope: str) -> Optional[dict]:
        """Return the highest-version payload for a scope, with context_id attached."""
        latest_context_id = None
        latest_record = None
        for (s, cid), record in self._store.items():
            if s != scope:
                continue
            if latest_record is None or record["version"] > latest_record["version"]:
                latest_context_id = cid
                latest_record = record

        if latest_record is None:
            return None

        payload = dict(latest_record["payload"])
        payload.setdefault(f"{scope}_id", latest_context_id)
        payload.setdefault("context_id", latest_context_id)
        return payload

    def get_counts(self) -> dict[str, int]:
        counts = {s: 0 for s in VALID_SCOPES}
        for (scope, _) in self._store:
            if scope in counts:
                counts[scope] += 1
        return counts

    def clear(self):
        with self._lock:
            self._store.clear()
