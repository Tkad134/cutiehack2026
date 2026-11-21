"""Routing engine: deterministic, rule-based choice between local and cloud inference.

The router never generates answers. It classifies the request, checks for
complexity indicators and input length, and returns a RoutingDecision with a
human-readable reason. Keyword matches are heuristics, not guarantees.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from hybrid_router.config import Settings
from hybrid_router.models import Destination, Mode, Request, RoutingDecision

log = logging.getLogger("hybrid_router")

STRATEGY_RULES = "rule_based"
STRATEGY_MANUAL = "manual_override"


def _rx(*patterns: str) -> re.Pattern[str]:
    return re.compile("|".join(f"(?:{p})" for p in patterns), re.IGNORECASE)


_TECH_ANALYSIS = _rx(
    r"\bsystem design\b",
    r"\barchitecture (?:review|analysis)\b",
    r"\bdistributed systems?\b",
    r"\bscalab(?:le|ility)\b",
    r"\btrade-?offs?\b",
    r"\bthreat model(?:ing)?\b",
    r"\bsecurity audit\b",
    r"\bin-depth (?:analysis|review)\b",
    r"\bdeep dive\b",
    r"\broot cause analysis\b",
    r"\bformal verification\b",
)

# (category id, display label, pattern). Listed in tie-break priority order.
CATEGORY_RULES: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    ("advanced_technical_analysis", "advanced technical analysis", _TECH_ANALYSIS),
    (
        "math",
        "mathematical reasoning",
        _rx(
            r"\bprove\b", r"\bproofs?\b", r"\btheorem\b", r"\blemma\b", r"\bequations?\b",
            r"\bsolve\b", r"\bcalculate\b", r"\bcompute\b", r"\bintegra(?:l|te)\b",
            r"\bderivatives?\b", r"\bmatri(?:x|ces)\b", r"\bprobability\b", r"\balgebra\b",
            r"\bgeometry\b", r"\bcalculus\b", r"\bmath(?:s|ematics|ematical)?\b",
            r"\d+\s*[-+*/^×÷]\s*\d+",
        ),
    ),
    (
        "summarization",
        "summarization",
        _rx(
            r"\bsummar(?:y|ize|ise|izing|ization)\b", r"\btl;?dr\b", r"\bkey points\b",
            r"\bmain points\b", r"\bcondense\b", r"\brecap\b",
        ),
    ),
    (
        "rewriting",
        "text rewriting",
        _rx(
            r"\brewrite\b", r"\brephrase\b", r"\bparaphrase\b", r"\bproofread\b",
            r"\bfix (?:the |my )?grammar\b", r"\btranslate\b",
            r"\bmake (?:this|it) (?:more )?(?:formal|casual|concise|shorter|clearer|polite)\b",
        ),
    ),
    (
        "creative_writing",
        "creative writing",
        _rx(
            r"\bpoem\b", r"\bpoetry\b", r"\bstory\b", r"\bhaiku\b", r"\blyrics\b",
            r"\bsong\b", r"\bjoke\b", r"\blimerick\b", r"\bscreenplay\b", r"\bfiction\b",
            r"\bslogan\b", r"\bcreative\b",
        ),
    ),
    (
        "coding",
        "coding",
        _rx(
            r"```", r"\bcode\b", r"\bfunction\b", r"\bpython\b", r"\bjavascript\b",
            r"\btypescript\b", r"\bjava\b", r"\bc\+\+", r"\brust\b", r"\bgolang\b",
            r"\bsql\b", r"\bregex\b", r"\bscript\b", r"\bprogram(?:ming)?\b",
            r"\bcompil(?:e|er|ation)\b", r"\bdebug(?:ging)?\b", r"\bbug\b", r"\bapi\b",
            r"\balgorithm\b", r"\bexception\b", r"\btraceback\b", r"^\s*(?:def|class|import) ",
        ),
    ),
    (
        "explanation",
        "basic explanation",
        _rx(
            r"\bexplain\b", r"\bwhat (?:is|are|was|were|does|do)\b",
            r"\bhow (?:does|do|is|are|can|to)\b", r"\bwhy (?:is|are|does|do)\b",
            r"\bdescribe\b", r"\bdefine\b", r"\bdefinition\b", r"\bmeaning of\b",
            r"\bdifference between\b",
        ),
    ),
)
CATEGORY_LABELS = {cid: label for cid, label, _ in CATEGORY_RULES}
CATEGORY_LABELS["general"] = "general question answering"

# Categories that go to the cloud by default even without other indicators.
CLOUD_CATEGORIES = {"advanced_technical_analysis"}

# (indicator label, pattern). Any match routes to the cloud.
COMPLEXITY_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "advanced proof",
        _rx(
            r"\bprove\b", r"\bproofs?\b", r"\btheorem\b", r"\blemma\b", r"\bcorollary\b",
            r"\bby (?:strong )?induction\b", r"\bformally (?:show|verify|derive)\b",
            r"\brigorous(?:ly)?\b",
        ),
    ),
    (
        "advanced mathematics",
        _rx(
            r"\bdifferential equations?\b", r"\beigen(?:values?|vectors?)\b",
            r"\b(?:fourier|laplace) transforms?\b", r"\bmultivariable\b", r"\btopology\b",
            r"\breal analysis\b", r"\babstract algebra\b", r"\bnumber theory\b",
            r"\bderive\b", r"\bderivation\b",
        ),
    ),
    (
        "complex debugging",
        _rx(
            r"\btraceback\b", r"\bstack ?traces?\b", r"\bsegfault\b",
            r"\bsegmentation fault\b", r"\brace conditions?\b", r"\bdeadlocks?\b",
            r"\bmemory leaks?\b", r"\bcore dump(?:ed)?\b", r"\bundefined behaviou?r\b",
            r"\bintermittent(?:ly)? (?:fail|crash)",
        ),
    ),
    (
        "advanced optimization",
        _rx(
            r"\boptimi[sz](?:e|ing|ation)\b[^.?!\n]{0,60}\b(?:performance|algorithm|complexity"
            r"|latency|throughput|quer(?:y|ies)|runtime|memory|code|function|loop|database)\b",
            r"\b(?:performance|algorithm|latency|throughput|quer(?:y|ies)|runtime|memory|code"
            r"|function|loop|database)\b[^.?!\n]{0,60}\boptimi[sz](?:e|ing|ation)\b",
            r"\b(?:time|space) complexity\b", r"\bbig[- ]o\b", r"\basymptotic\b",
            r"\b(?:linear|integer|convex) programming\b", r"\bconvex optimi[sz]ation\b",
        ),
    ),
    (
        "multi-step reasoning",
        _rx(
            r"\bmulti-?step\b", r"\bchain of reasoning\b",
            r"\bstep[- ]by[- ]step (?:proof|derivation|analysis)\b",
        ),
    ),
    ("advanced technical analysis", _TECH_ANALYSIS),
)

_DEBUG_REQUEST = _rx(
    r"\bdebug(?:ging)?\b", r"\bfix\b", r"\bbugs?\b", r"\berrors?\b", r"\bnot working\b",
    r"\bdoesn'?t work\b", r"\bbroken\b", r"\bcrash(?:es|ing)?\b",
)


def classify(text: str) -> str:
    """Return the category id with the most distinct keyword hits (ties -> priority order)."""
    best_id, best_score = "general", 0
    for cid, _label, pattern in CATEGORY_RULES:
        hits = {m.group(0).strip().lower() for m in pattern.finditer(text)}
        if len(hits) > best_score:
            best_id, best_score = cid, len(hits)
    return best_id


def find_complexity_indicators(text: str, debug_lines_threshold: int) -> list[str]:
    """Return human-readable descriptions of matched complexity indicators."""
    found: list[str] = []
    for label, pattern in COMPLEXITY_RULES:
        m = pattern.search(text)
        if m:
            found.append(f"{label} ('{m.group(0).strip()}')")
    non_empty_lines = sum(1 for line in text.splitlines() if line.strip())
    if non_empty_lines >= debug_lines_threshold and _DEBUG_REQUEST.search(text):
        if not any(f.startswith("complex debugging") for f in found):
            found.append(f"complex debugging (debug request with {non_empty_lines} lines of input)")
    return found


class Router:
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()

    def route(self, request: Request) -> RoutingDecision:
        decision = self._decide(request)
        log.info(
            "[%s] routed to %s (%s, category=%s): %s",
            request.request_id, decision.destination.value, decision.strategy,
            decision.category, decision.reason,
        )
        return decision

    def _decide(self, request: Request) -> RoutingDecision:
        rid = request.request_id
        category = classify(request.prompt)
        label = CATEGORY_LABELS[category]

        # 1. Manual overrides bypass the routing algorithm entirely.
        if request.mode is Mode.LOCAL:
            return RoutingDecision(
                rid, Destination.LOCAL,
                f"Local-Only mode selected: automatic routing bypassed (task looks like {label}).",
                STRATEGY_MANUAL, category,
            )
        if request.mode is Mode.CLOUD:
            return RoutingDecision(
                rid, Destination.CLOUD,
                f"Cloud-Only mode selected: automatic routing bypassed (task looks like {label}).",
                STRATEGY_MANUAL, category,
            )

        # 2. Length: input (prompt + context) beyond what the local model should handle.
        total_chars = len(request.prompt) + sum(len(m.content) for m in request.context)
        limit = self.settings.local_max_input_chars
        if total_chars > limit:
            return RoutingDecision(
                rid, Destination.CLOUD,
                f"Input is {total_chars:,} characters including context, over the local "
                f"limit of {limit:,}; sent to the cloud model.",
                STRATEGY_RULES, category,
            )

        # 3. Complexity indicators.
        indicators = find_complexity_indicators(
            request.prompt, self.settings.debug_code_lines_threshold
        )
        if indicators:
            return RoutingDecision(
                rid, Destination.CLOUD,
                f"Classified as {label}; complexity indicators found: "
                f"{'; '.join(indicators)}. Likely beyond the local model, sent to the cloud.",
                STRATEGY_RULES, category,
            )

        # 4. Category defaults.
        if category in CLOUD_CATEGORIES:
            return RoutingDecision(
                rid, Destination.CLOUD,
                f"Classified as {label}, which is routed to the cloud model by policy.",
                STRATEGY_RULES, category,
            )
        return RoutingDecision(
            rid, Destination.LOCAL,
            f"Classified as {label} with no complexity indicators and {total_chars:,} "
            f"characters of input; handled locally to save cloud tokens.",
            STRATEGY_RULES, category,
        )
