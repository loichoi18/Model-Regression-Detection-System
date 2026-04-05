"""
Email Classifier — the LLM feature under test.

This is the production feature that the eval pipeline monitors.
It takes an email and returns a structured classification.
The classifier is provider-agnostic: it works with any LLMProvider.
"""

from __future__ import annotations

from src.mock_llm import LLMProvider, MockLLMProvider
from src.models import ClassificationResult, PromptConfig


class EmailClassifier:
    """
    Classifies customer support emails using an LLM.

    Usage:
        config = load_prompt("v1")
        classifier = EmailClassifier(provider=MockLLMProvider())
        result, latency, tokens = await classifier.classify(config, email_text)
    """

    def __init__(self, provider: LLMProvider | None = None):
        self.provider = provider or MockLLMProvider()

    async def classify(
        self, prompt_config: PromptConfig, email_text: str
    ) -> tuple[ClassificationResult, float, int]:
        """
        Classify a single email.

        Returns:
            tuple of (ClassificationResult, latency_ms, tokens_used)
        """
        return await self.provider.classify(prompt_config, email_text)
