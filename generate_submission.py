"""
generate_submission.py — generates submission.jsonl

Runs compose() for all 30 canonical test pairs.
Uses the MessageGenerator with real LLM calls.
Requires GEMINI_API_KEY to be set.

Usage:
    python generate_submission.py
Outputs:
    submission.jsonl (30 lines)
"""

import json
import os
import sys
import time
from pathlib import Path

# Add vera-bot to path
sys.path.insert(0, str(Path(__file__).parent))

from message_generator import MessageGenerator

EXPANDED_DIR = Path(__file__).parent.parent / "expanded"


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def main():
    # Load test pairs
    test_pairs = load_json(EXPANDED_DIR / "test_pairs.json")["pairs"]
    print(f"Loaded {len(test_pairs)} test pairs")

    gen = MessageGenerator()
    results = []

    for pair in test_pairs:
        test_id = pair["test_id"]
        trigger_id = pair["trigger_id"]
        merchant_id = pair["merchant_id"]
        customer_id = pair.get("customer_id")

        print(f"Composing {test_id}: merchant={merchant_id} trigger={trigger_id}")

        # Load contexts
        merchant_path = EXPANDED_DIR / "merchants" / f"{merchant_id}.json"
        trigger_path = EXPANDED_DIR / "triggers" / f"{trigger_id}.json"

        if not merchant_path.exists():
            print(f"  SKIP: merchant file not found: {merchant_path}")
            continue
        if not trigger_path.exists():
            print(f"  SKIP: trigger file not found: {trigger_path}")
            continue

        merchant = load_json(merchant_path)
        trigger = load_json(trigger_path)

        category_slug = merchant.get("category_slug", "")
        category_path = EXPANDED_DIR / "categories" / f"{category_slug}.json"
        if not category_path.exists():
            print(f"  SKIP: category not found: {category_slug}")
            continue
        category = load_json(category_path)

        customer = None
        if customer_id:
            customer_path = EXPANDED_DIR / "customers" / f"{customer_id}.json"
            if customer_path.exists():
                customer = load_json(customer_path)

        try:
            composed = gen.compose(
                category=category,
                merchant=merchant,
                trigger=trigger,
                customer=customer,
            )

            result = {
                "test_id": test_id,
                "body": composed.get("body", ""),
                "cta": composed.get("cta", "open_ended"),
                "send_as": composed.get("send_as", "vera"),
                "suppression_key": composed.get("suppression_key", ""),
                "rationale": composed.get("rationale", ""),
            }
            results.append(result)
            print(f"  ✓ send_as={result['send_as']} cta={result['cta']}")
            print(f"    body: {result['body'][:120]}...")
            # Small delay to avoid rate limits
            time.sleep(0.5)

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "test_id": test_id,
                "body": f"[Error: {str(e)[:100]}]",
                "cta": "open_ended",
                "send_as": "vera",
                "suppression_key": "",
                "rationale": "Compose error",
            })

    # Write submission.jsonl
    output_path = Path(__file__).parent / "submission.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(results)} results to {output_path}")


if __name__ == "__main__":
    main()
