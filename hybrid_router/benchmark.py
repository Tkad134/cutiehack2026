"""Offline route-policy benchmark; never calls a model or loads credentials."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from hybrid_router.config import Settings
from hybrid_router.request_processor import process_request
from hybrid_router.router import Router


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=Path(__file__).resolve().parents[1] / "benchmarks/router_cases.json")
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text())
    settings = Settings()
    router = Router(settings)
    local = correct = 0
    mismatches = []
    for case in cases:
        request = process_request(case["prompt"], case.get("mode", "auto"), case.get("context"), settings)
        decision = router.route(request)
        actual = decision.destination.value
        local += actual == "local"
        correct += actual == case["expected"]
        if actual != case["expected"]:
            mismatches.append({"id": case["id"], "expected": case["expected"], "actual": actual, "reason": decision.reason})
    count = len(cases)
    print(json.dumps({
        "cases": count,
        "policy_agreement": correct / count if count else 0,
        "local_request_share": local / count if count else 0,
        "mismatches": mismatches,
        "note": "Hand-labeled routing cases; does not measure answer quality, token usage, or dollar savings.",
    }, indent=2))
    return int(bool(mismatches))


if __name__ == "__main__":
    logging.disable(logging.INFO)
    raise SystemExit(main())
