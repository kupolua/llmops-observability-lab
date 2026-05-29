import os
from dataclasses import dataclass
from typing import Any

from anthropic import AsyncAnthropic
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ModelConfig:
    """Один уровень в fallback chain."""

    provider: str  # "anthropic" | "openai"
    model: str
    label: str  # человекочитаемое имя для логов


class FallbackChain:
    """Цепочка моделей: пробуем по порядку, пока одна не сработает."""

    def __init__(self, chain: list[ModelConfig]) -> None:
        if not chain:
            raise ValueError("Fallback chain cannot be empty")
        self._chain = chain
        self._anthropic = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        # OpenAI опционально
        self._openai: Any = None
        if os.environ.get("OPENAI_API_KEY"):
            from openai import AsyncOpenAI

            self._openai = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

    async def _call_anthropic(self, model: str, prompt: str) -> str:
        response = await self._anthropic.messages.create(
            model=model,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        block = response.content[0]
        return block.text if hasattr(block, "text") else ""

    async def _call_openai(self, model: str, prompt: str) -> str:
        if self._openai is None:
            raise RuntimeError("OpenAI client not configured")
        response = await self._openai.chat.completions.create(
            model=model,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""

    async def ask(self, prompt: str) -> tuple[str, str]:
        """Вернуть (ответ, label_модели_которая_сработала)."""
        errors: list[str] = []

        for config in self._chain:
            try:
                print(f"\033[90m  trying {config.label}...\033[0m")
                if config.provider == "anthropic":
                    text = await self._call_anthropic(config.model, prompt)
                elif config.provider == "openai":
                    text = await self._call_openai(config.model, prompt)
                else:
                    raise ValueError(f"Unknown provider: {config.provider}")

                print(f"\033[92m  ✓ {config.label} succeeded\033[0m")
                return text, config.label

            except Exception as e:
                error_msg = f"{config.label} failed: {type(e).__name__}"
                print(f"\033[91m  ✗ {error_msg}\033[0m")
                errors.append(error_msg)
                continue

        raise RuntimeError(f"All models in chain failed: {errors}")


async def main() -> None:

    # Цепочка: Sonnet → Haiku → (если есть) GPT
    chain = [
        ModelConfig("anthropic", "claude-nonexistent-model", "Broken Model"),
        ModelConfig("anthropic", "claude-sonnet-4-5", "Claude Sonnet 4.5"),
        ModelConfig("anthropic", "claude-haiku-4-5-20251001", "Claude Haiku 4.5"),
    ]
    if os.environ.get("OPENAI_API_KEY"):
        chain.append(ModelConfig("openai", "gpt-4o-mini", "GPT-4o-mini"))

    fc = FallbackChain(chain)
    answer, used_model = await fc.ask("Объясни RAG в одном предложении.")
    print(f"\n=== Ответ (от {used_model}) ===\n{answer}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
