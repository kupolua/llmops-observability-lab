from dataclasses import dataclass
from anthropic import AsyncAnthropic
from anthropic.types import Message, MessageParam, TextBlock
from claude_lab.usage import CallCost, calculate_cost


@dataclass
class CallResult:
    """Результат одного вызова Claude."""

    text: str
    raw: Message
    cost: CallCost


class ClaudeClient:
    """Тонкая обёртка вокруг AsyncAnthropic.

    Зачем: централизованная точка для трекинга стоимости, retry,
    логирования. Сейчас просто шаблон, в следующих днях будем
    наращивать функциональность.
    """

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-5") -> None:
        self._client = AsyncAnthropic(api_key=api_key)
        self._model = model
        self._total_cost: float = 0.0
        self._call_count: int = 0

    async def ask(
        self,
        prompt: str,
        max_tokens: int = 1024,
        system: str | None = None,
    ) -> CallResult:
        """Шаг 1: messages"""
        messages: list[MessageParam] = [{"role": "user", "content": prompt}]

        """# Шаг 2: вызов API (с if/else для system)"""
        if system is not None:
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                messages=messages,
                system=system,
            )
        else:
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                messages=messages,
            )

        """# Шаг 3: извлечение текста с проверкой типа блока"""
        block = response.content[0]
        if not isinstance(block, TextBlock):
            raise ValueError(f"Expected TextBlock, got {type(block).__name__}")
        text = block.text

        """# Шаг 4: стоимость"""
        cost = calculate_cost(response.usage)

        """# Шаг 5: обновление счётчиков"""
        self._total_cost += cost.total_cost_usd
        self._call_count += 1

        """# Шаг 6: возврат"""
        return CallResult(text=text, raw=response, cost=cost)

    @property
    def total_cost_usd(self) -> float:
        return round(self._total_cost, 6)

    @property
    def call_count(self) -> int:
        return self._call_count
