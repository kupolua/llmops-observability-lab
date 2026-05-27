from dataclasses import dataclass
from anthropic import AsyncAnthropic
from anthropic.types import Message, TextBlock, MessageParam
from lf_client.usage import CallCost, calculate_cost
from langfuse import Langfuse


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

    def __init__(
        self,
        api_key: str,
        langfuse: Langfuse,  # ← новый параметр
        model: str = "claude-sonnet-4-5",
    ) -> None:
        self._client = AsyncAnthropic(api_key=api_key)
        self._langfuse = langfuse
        self._model = model
        self._total_cost: float = 0.0
        self._call_count: int = 0

    async def ask(
        self,
        prompt: str,
        max_tokens: int = 1024,
        system: str | None = None,
        trace_name: str = "claude-ask",
        user_id: str | None = None,
        session_id: str | None = None,
        tags: list[str] | None = None,
        environment: str = "dev",
    ) -> CallResult:
        messages: list[MessageParam] = [{"role": "user", "content": prompt}]

        # Корневой span — это будет trace в UI
        with self._langfuse.start_as_current_observation(
            as_type="span",
            name=trace_name,
            input=prompt,
        ) as root_span:
            # set_trace_io ставит метаданные на trace целиком
            # Используем update_current_span для расширенных полей
            self._langfuse.update_current_span(
                metadata={
                    "environment": environment,
                    "user_id": user_id,
                    "session_id": session_id,
                    "tags": tags or [],
                    "system": system,
                },
            )

            # Вложенный generation — конкретный вызов LLM
            with self._langfuse.start_as_current_observation(
                as_type="generation",
                name=f"{trace_name}-llm",
                model=self._model,
                input=messages,
            ) as generation:
                try:
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

                    block = response.content[0]
                    if not isinstance(block, TextBlock):
                        raise ValueError(
                            f"Expected TextBlock, got {type(block).__name__}"
                        )
                    text = block.text

                    cost = calculate_cost(response.usage)

                    generation.update(
                        output=text,
                        usage_details={
                            "input": response.usage.input_tokens,
                            "output": response.usage.output_tokens,
                        },
                    )

                    root_span.update(output=text)

                    self._total_cost += cost.total_cost_usd
                    self._call_count += 1

                    return CallResult(text=text, raw=response, cost=cost)

                except Exception as e:
                    generation.update(level="ERROR", status_message=str(e))
                    root_span.update(level="ERROR", status_message=str(e))
                    raise

    @property
    def total_cost_usd(self) -> float:
        return round(self._total_cost, 6)

    @property
    def call_count(self) -> int:
        return self._call_count
