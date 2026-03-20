"""LLM client wrapping Anthropic Claude for medical education RAG."""

import os

import anthropic
from dotenv import load_dotenv

load_dotenv()


class LLMClient:
    """Thin wrapper around Anthropic's Messages API."""

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        max_tokens: int = 1024,
    ):
        self.client = anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        )
        self.model = model
        self.max_tokens = max_tokens

    def complete(
        self,
        prompt: str,
        system: str = "You are a helpful medical education assistant.",
        temperature: float = 0.3,
        max_tokens: int | None = None,
    ) -> str:
        """Send a prompt to Claude and return the response text."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens or self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            timeout=30.0,
        )
        return response.content[0].text
