"""
Mock LLM provider for development and testing.

Simulates OpenAI-style chat completions with deterministic outputs.
The mock uses keyword matching to produce realistic-looking classifications,
including deliberate mistakes on edge cases to make eval diffs interesting.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from typing import Protocol

from src.models import ClassificationResult, EmailCategory, PromptConfig


class LLMProvider(Protocol):
    """Interface that both mock and real providers implement."""

    async def classify(
        self, prompt_config: PromptConfig, email_text: str
    ) -> tuple[ClassificationResult, float, int]:
        """
        Returns: (result, latency_ms, tokens_used)
        """
        ...


# ---------------------------------------------------------------------------
# Keyword rules for the mock classifier
# ---------------------------------------------------------------------------

_KEYWORD_MAP: dict[str, EmailCategory] = {
    # Billing — financial transactions, pricing, charges
    "charge": EmailCategory.BILLING,
    "charged": EmailCategory.BILLING,
    "refund": EmailCategory.BILLING,
    "invoice": EmailCategory.BILLING,
    "payment": EmailCategory.BILLING,
    "subscription": EmailCategory.BILLING,
    "billing": EmailCategory.BILLING,
    "price": EmailCategory.BILLING,
    "pricing": EmailCategory.BILLING,
    "cost": EmailCategory.BILLING,
    "upgrade": EmailCategory.BILLING,
    "downgrade": EmailCategory.BILLING,
    "discount": EmailCategory.BILLING,
    "coupon": EmailCategory.BILLING,
    "receipt": EmailCategory.BILLING,
    "credit card": EmailCategory.BILLING,
    "card on file": EmailCategory.BILLING,
    "prorate": EmailCategory.BILLING,
    "annual billing": EmailCategory.BILLING,
    "pay annually": EmailCategory.BILLING,
    "seats": EmailCategory.BILLING,
    "team plan": EmailCategory.BILLING,
    "add my colleague": EmailCategory.BILLING,
    "api access": EmailCategory.BILLING,
    "included in the": EmailCategory.BILLING,
    "what's included": EmailCategory.BILLING,
    "free trial": EmailCategory.BILLING,
    # Technical — bugs, errors, performance, features not working
    "error": EmailCategory.TECHNICAL,
    "bug": EmailCategory.TECHNICAL,
    "crash": EmailCategory.TECHNICAL,
    "crashes": EmailCategory.TECHNICAL,
    "broken": EmailCategory.TECHNICAL,
    "slow": EmailCategory.TECHNICAL,
    "loading": EmailCategory.TECHNICAL,
    "500": EmailCategory.TECHNICAL,
    "404": EmailCategory.TECHNICAL,
    "api": EmailCategory.TECHNICAL,
    "export": EmailCategory.TECHNICAL,
    "timeout": EmailCategory.TECHNICAL,
    "glitch": EmailCategory.TECHNICAL,
    "freeze": EmailCategory.TECHNICAL,
    "lag": EmailCategory.TECHNICAL,
    "doesn't work": EmailCategory.TECHNICAL,
    "does not work": EmailCategory.TECHNICAL,
    "not working": EmailCategory.TECHNICAL,
    "won't load": EmailCategory.TECHNICAL,
    "won't work": EmailCategory.TECHNICAL,
    "can't load": EmailCategory.TECHNICAL,
    "blank screen": EmailCategory.TECHNICAL,
    "spinning": EmailCategory.TECHNICAL,
    "dashboard": EmailCategory.TECHNICAL,
    "formatting": EmailCategory.TECHNICAL,
    "notification": EmailCategory.TECHNICAL,
    "notifications": EmailCategory.TECHNICAL,
    "delayed": EmailCategory.TECHNICAL,
    "toggle": EmailCategory.TECHNICAL,
    "persist": EmailCategory.TECHNICAL,
    "duplicate": EmailCategory.TECHNICAL,
    "duplicates": EmailCategory.TECHNICAL,
    "drag and drop": EmailCategory.TECHNICAL,
    "drag": EmailCategory.TECHNICAL,
    "paste": EmailCategory.TECHNICAL,
    "display": EmailCategory.TECHNICAL,
    "render": EmailCategory.TECHNICAL,
    "search function": EmailCategory.TECHNICAL,
    "ssl": EmailCategory.TECHNICAL,
    "certificate": EmailCategory.TECHNICAL,
    "dark mode": EmailCategory.TECHNICAL,
    "spins forever": EmailCategory.TECHNICAL,
    "import": EmailCategory.TECHNICAL,
    "save": EmailCategory.TECHNICAL,
    "hasn't updated": EmailCategory.TECHNICAL,
    "not updated": EmailCategory.TECHNICAL,
    "settings page": EmailCategory.TECHNICAL,
    "still shows": EmailCategory.TECHNICAL,
    "still on the": EmailCategory.TECHNICAL,
    "stopped working": EmailCategory.TECHNICAL,
    "has stopped": EmailCategory.TECHNICAL,
    "isn't working": EmailCategory.TECHNICAL,
    "isnt working": EmailCategory.TECHNICAL,
    "auto-save": EmailCategory.TECHNICAL,
    "memory leak": EmailCategory.TECHNICAL,
    "memory usage": EmailCategory.TECHNICAL,
    "unresponsive": EmailCategory.TECHNICAL,
    "shortcuts": EmailCategory.TECHNICAL,
    "webhook": EmailCategory.TECHNICAL,
    "integration": EmailCategory.TECHNICAL,
    "dropping": EmailCategory.TECHNICAL,
    # Account — login, password, profile, access
    "password": EmailCategory.ACCOUNT,
    "login": EmailCategory.ACCOUNT,
    "log in": EmailCategory.ACCOUNT,
    "sign in": EmailCategory.ACCOUNT,
    "account": EmailCategory.ACCOUNT,
    "profile": EmailCategory.ACCOUNT,
    "delete my": EmailCategory.ACCOUNT,
    "delete my account": EmailCategory.ACCOUNT,
    "deactivate": EmailCategory.ACCOUNT,
    "username": EmailCategory.ACCOUNT,
    "two-factor": EmailCategory.ACCOUNT,
    "2fa": EmailCategory.ACCOUNT,
    "locked out": EmailCategory.ACCOUNT,
    "logged out": EmailCategory.ACCOUNT,
    "session expired": EmailCategory.ACCOUNT,
    "merge two accounts": EmailCategory.ACCOUNT,
    "reset email": EmailCategory.ACCOUNT,
    "sso": EmailCategory.ACCOUNT,
    "ownership": EmailCategory.ACCOUNT,
    "transfer ownership": EmailCategory.ACCOUNT,
    "email address": EmailCategory.ACCOUNT,
    "change my email": EmailCategory.ACCOUNT,
    "my account": EmailCategory.ACCOUNT,
    "cancelled": EmailCategory.ACCOUNT,
    "can't get in": EmailCategory.ACCOUNT,
    "cant get in": EmailCategory.ACCOUNT,
    "deleted all": EmailCategory.ACCOUNT,
    "accidentally deleted": EmailCategory.ACCOUNT,
    "recover": EmailCategory.ACCOUNT,
    "data deletion": EmailCategory.ACCOUNT,
    "revoke": EmailCategory.ACCOUNT,
    "admin": EmailCategory.ACCOUNT,
    "admin access": EmailCategory.ACCOUNT,
    "manage users": EmailCategory.ACCOUNT,
    "workspace": EmailCategory.ACCOUNT,
    # General — feedback, questions, requests, partnerships
    "roadmap": EmailCategory.GENERAL,
    "feature request": EmailCategory.GENERAL,
    "feedback": EmailCategory.GENERAL,
    "great work": EmailCategory.GENERAL,
    "love your": EmailCategory.GENERAL,
    "keep up": EmailCategory.GENERAL,
    "blogger": EmailCategory.GENERAL,
    "press account": EmailCategory.GENERAL,
    "non-profit": EmailCategory.GENERAL,
    "partnership": EmailCategory.GENERAL,
    "research paper": EmailCategory.GENERAL,
    "white paper": EmailCategory.GENERAL,
    "referral": EmailCategory.GENERAL,
    "affiliate": EmailCategory.GENERAL,
    "community forum": EmailCategory.GENERAL,
    "discord": EmailCategory.GENERAL,
    "youtube": EmailCategory.GENERAL,
    "compare": EmailCategory.GENERAL,
    "conference": EmailCategory.GENERAL,
    "demo": EmailCategory.GENERAL,
    "bank account": EmailCategory.GENERAL,
    "nigerian": EmailCategory.GENERAL,
    "transferring": EmailCategory.GENERAL,
    "legitimate": EmailCategory.GENERAL,
    "documentation": EmailCategory.GENERAL,
    "api reference": EmailCategory.GENERAL,
    "api documentation": EmailCategory.GENERAL,
    "school district": EmailCategory.GENERAL,
    "education": EmailCategory.GENERAL,
    "hipaa": EmailCategory.GENERAL,
    "compliance": EmailCategory.GENERAL,
    "podcast": EmailCategory.GENERAL,
    "soc 2": EmailCategory.GENERAL,
    "professor": EmailCategory.GENERAL,
    "academic": EmailCategory.GENERAL,
    "university": EmailCategory.GENERAL,
    "case study": EmailCategory.GENERAL,
    "case studies": EmailCategory.GENERAL,
    "onboarding": EmailCategory.GENERAL,
    "training": EmailCategory.GENERAL,
    "status page": EmailCategory.GENERAL,
    "competitor": EmailCategory.GENERAL,
    "export all my data": EmailCategory.GENERAL,
    "export my data": EmailCategory.GENERAL,
    "locked in": EmailCategory.GENERAL,
    "data portability": EmailCategory.GENERAL,
}

# Words that are too ambiguous to be strong signals on their own.
# These only contribute half weight to avoid false positives.
_WEAK_KEYWORDS: set[str] = {"plan", "account", "verify", "save", "api access", "free trial", "charged"}

_SUMMARY_TEMPLATES = {
    EmailCategory.BILLING: "Customer has a billing-related inquiry regarding their account charges.",
    EmailCategory.TECHNICAL: "Customer is experiencing a technical issue with the product.",
    EmailCategory.ACCOUNT: "Customer needs assistance with their account settings or access.",
    EmailCategory.GENERAL: "Customer has a general inquiry or feedback about the service.",
}


class MockLLMProvider:
    """
    Deterministic mock that simulates realistic LLM classification behavior.

    Features:
    - Keyword-based classification (mimics what an LLM would likely do)
    - Configurable error rate on edge cases
    - Simulated latency jitter
    - Token usage estimation
    - Version-aware behavior: v2 prompts intentionally degrade on some cases
    """

    def __init__(
        self,
        base_latency_ms: float = 150.0,
        latency_jitter_ms: float = 50.0,
        edge_case_error_rate: float = 0.15,
        seed: int = 42,
    ):
        self.base_latency_ms = base_latency_ms
        self.latency_jitter_ms = latency_jitter_ms
        self.edge_case_error_rate = edge_case_error_rate
        self._rng = random.Random(seed)

    async def classify(
        self, prompt_config: PromptConfig, email_text: str
    ) -> tuple[ClassificationResult, float, int]:
        """Simulate an LLM classification call."""
        # Simulate network latency
        latency = self.base_latency_ms + self._rng.gauss(0, self.latency_jitter_ms)
        latency = max(50.0, latency)
        await asyncio.sleep(latency / 1000.0)

        # Classify using keywords
        category = self._classify_by_keywords(email_text)

        # Simulate version-dependent behavior:
        # v2 prompt is weaker — multiple degradation types
        # v3 prompt has a subtle flaw — only sarcasm detection fails
        if prompt_config.version == "v2":
            category = self._apply_v2_degradation(email_text, category)
        elif prompt_config.version >= "v3":
            category = self._apply_v3_degradation(email_text, category)

        # Generate summary
        summary = self._generate_summary(email_text, category)

        # Estimate tokens (rough: ~1.3 tokens per word)
        word_count = len(email_text.split())
        tokens = int(word_count * 1.3) + 80  # +80 for prompt overhead

        confidence = 0.92 if category != EmailCategory.GENERAL else 0.75

        result = ClassificationResult(
            category=category,
            summary=summary,
            confidence=round(confidence, 2),
        )
        return result, round(latency, 2), tokens

    def _classify_by_keywords(self, email_text: str) -> EmailCategory:
        """Match keywords to determine category. Longer phrases matched first."""
        text_lower = email_text.lower()
        scores: dict[EmailCategory, float] = {cat: 0 for cat in EmailCategory}

        # Sort keywords longest-first so multi-word phrases match before
        # their individual words (e.g., "press account" before "account")
        sorted_keywords = sorted(_KEYWORD_MAP.keys(), key=len, reverse=True)
        matched_spans: list[tuple[int, int]] = []

        for keyword in sorted_keywords:
            pos = text_lower.find(keyword)
            if pos == -1:
                continue

            # Skip if this position is already covered by a longer phrase
            end = pos + len(keyword)
            if any(s <= pos and end <= e for s, e in matched_spans):
                continue

            matched_spans.append((pos, end))
            category = _KEYWORD_MAP[keyword]

            # Weak keywords get half weight
            weight = 0.5 if keyword in _WEAK_KEYWORDS else 1.0
            scores[category] += weight

        max_score = max(scores.values())
        if max_score == 0:
            return EmailCategory.GENERAL

        # If there's a tie, prefer account over technical (account issues
        # often manifest as technical symptoms like errors or broken flows)
        for cat in [EmailCategory.BILLING, EmailCategory.ACCOUNT,
                    EmailCategory.TECHNICAL, EmailCategory.GENERAL]:
            if scores[cat] == max_score:
                return cat

        return EmailCategory.GENERAL

    def _apply_v2_degradation(
        self, email_text: str, original_category: EmailCategory
    ) -> EmailCategory:
        """
        v2 prompt is intentionally weaker. Simulates the kind of regressions
        you'd see when someone simplifies a prompt too aggressively.

        The degradation is targeted, not blanket:
        - Upgrade/downgrade emails WITHOUT strong billing context get misclassified
        - Sarcastic emails lose their sarcasm detection
        - Short emails sometimes confuse the model
        """
        text_lower = email_text.lower()

        # v2 misclassifies "upgrade/downgrade" as account — BUT only when
        # the email lacks strong billing signals (dollar amounts, refund, etc.)
        # A real weakened prompt would still get "I want a refund for my
        # downgrade" right, but miss "I want to downgrade — will I get
        # prorated credit?"
        strong_billing = any(
            s in text_lower
            for s in ["$", "refund", "charged", "invoice", "payment", "receipt"]
        )
        if any(w in text_lower for w in ["upgrade", "downgrade"]) and not strong_billing:
            return EmailCategory.ACCOUNT

        # v2 misclassifies sarcastic emails
        if any(w in text_lower for w in ["love how", "great job", "amazing that"]):
            return EmailCategory.GENERAL

        # v2 sometimes confuses short emails
        if len(email_text.split()) < 10:
            if self._rng.random() < 0.3:
                categories = list(EmailCategory)
                categories.remove(original_category)
                return self._rng.choice(categories)

        return original_category

    def _apply_v3_degradation(
        self, email_text: str, original_category: EmailCategory
    ) -> EmailCategory:
        """
        v3 prompt has a subtle flaw: the instruction 'classify based on
        literal tone' causes it to take sarcastic praise at face value.

        This produces exactly 1 regression from v1 — proving the system
        can detect even a single-case quality drop.
        """
        text_lower = email_text.lower()

        # v3 only fails on strong sarcasm that opens with positive language
        if text_lower.startswith("amazing that"):
            return EmailCategory.GENERAL

        return original_category

    def _generate_summary(self, email_text: str, category: EmailCategory) -> str:
        """Generate a plausible summary from the email text."""
        # Take the first sentence, truncate, and prefix with context
        first_sentence = email_text.split(".")[0].strip()
        if len(first_sentence) > 80:
            first_sentence = first_sentence[:77] + "..."

        prefix = {
            EmailCategory.BILLING: "Customer reports billing issue:",
            EmailCategory.TECHNICAL: "Customer reports technical problem:",
            EmailCategory.ACCOUNT: "Customer needs account help:",
            EmailCategory.GENERAL: "Customer inquiry:",
        }
        return f"{prefix[category]} {first_sentence}."
