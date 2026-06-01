from dataclasses import dataclass
from typing import Any
from anthropic import AsyncAnthropic
from anthropic.types import (
    Message,
    TextBlock,
    MessageParam,
    TextBlockParam,
    ToolParam,
    ToolUseBlock,
)
from lf_client.usage import CallCost, calculate_cost
from langfuse import Langfuse
from anthropic import APIStatusError
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)


def _is_retryable(exception: BaseException) -> bool:
    if not isinstance(exception, APIStatusError):
        return False
    return exception.response.status_code in {429, 529, 500, 502, 503, 504}


@dataclass
class CallResult:
    """Результат одного вызова Claude."""

    text: str
    raw: Message
    cost: CallCost


@dataclass
class ToolCallResult:
    """Результат вызова Claude с принудительным tool_use.

    tool_input — провалидированный моделью JSON (аргументы инструмента).
    Это структурированный выход, а не свободный текст: парсить нечего.
    """

    tool_input: dict[str, Any]
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
        api_key: str | None = None,
        langfuse: Langfuse | None = None,
        model: str = "claude-sonnet-4-5",
        anthropic_client: AsyncAnthropic | None = None,  # ← новый параметр
    ) -> None:
        if anthropic_client is not None:
            self._client = anthropic_client
        else:
            if api_key is None:
                raise ValueError("Either api_key or anthropic_client must be provided")
            self._client = AsyncAnthropic(api_key=api_key)

        if langfuse is None:
            raise ValueError("langfuse client must be provided")
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
        max_retries: int = 3,
        cache_system: bool = False,
    ) -> CallResult:
        messages: list[MessageParam] = [{"role": "user", "content": prompt}]

        with self._langfuse.start_as_current_observation(
            as_type="span",
            name=trace_name,
            input=prompt,
        ) as root_span:
            self._langfuse.update_current_span(
                metadata={
                    "environment": environment,
                    "user_id": user_id,
                    "session_id": session_id,
                    "tags": tags or [],
                    "system": system,
                    "max_retries": max_retries,
                },
            )

            with self._langfuse.start_as_current_observation(
                as_type="generation",
                name=f"{trace_name}-llm",
                model=self._model,
                input=messages,
            ) as generation:
                attempt_num = 0
                try:
                    # Retry-цикл вокруг настоящего вызова
                    async for attempt in AsyncRetrying(
                        stop=stop_after_attempt(
                            max_retries + 1
                        ),  # +1 потому что первая попытка — это «попытка», а max_retries — это retry-попытки
                        wait=wait_exponential_jitter(initial=1, max=10, jitter=2),
                        retry=retry_if_exception(_is_retryable),
                        reraise=True,
                    ):
                        with attempt:
                            attempt_num = attempt.retry_state.attempt_number
                            if system is not None:
                                if cache_system:
                                    system_param = None
                                elif cache_system:
                                    system_param = [
                                        TextBlockParam(
                                            type="text",
                                            text=system,
                                            cache_control={"type": "ephemeral"},
                                        )
                                    ]
                                else:
                                    system_param = system

                                if system_param is not None:
                                    response = await self._client.messages.create(
                                        model=self._model,
                                        max_tokens=max_tokens,
                                        messages=messages,
                                        system=system_param,
                                    )
                                else:
                                    response = await self._client.messages.create(
                                        model=self._model,
                                        max_tokens=max_tokens,
                                        messages=messages,
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
                        metadata={
                            "attempts": attempt_num
                        },  # ← фиксируем, сколько было попыток
                    )
                    root_span.update(output=text, metadata={"attempts": attempt_num})

                    self._total_cost += cost.total_cost_usd
                    self._call_count += 1

                    return CallResult(text=text, raw=response, cost=cost)

                except Exception as e:
                    generation.update(level="ERROR", status_message=str(e))
                    root_span.update(level="ERROR", status_message=str(e))
                    raise

    async def ask_tool(
        self,
        prompt: str,
        tools: list[ToolParam],
        tool_name: str,
        max_tokens: int = 1024,
        system: str | None = None,
        trace_name: str = "claude-tool",
        user_id: str | None = None,
        session_id: str | None = None,
        tags: list[str] | None = None,
        environment: str = "dev",
        max_retries: int = 3,
    ) -> ToolCallResult:
        """Вызов с принудительным tool_use → структурированный JSON-выход.

        В отличие от ask(), здесь tool_choice заставляет Claude вызвать
        конкретный инструмент tool_name. Возвращаем его аргументы (input)
        как dict — это надёжнее, чем парсить свободный текст.

        Та же обвязка, что и у ask(): один общий span + generation в
        Langfuse, retry на временных ошибках, учёт стоимости.
        """
        messages: list[MessageParam] = [{"role": "user", "content": prompt}]

        with self._langfuse.start_as_current_observation(
            as_type="span",
            name=trace_name,
            input=prompt,
        ) as root_span:
            self._langfuse.update_current_span(
                metadata={
                    "environment": environment,
                    "user_id": user_id,
                    "session_id": session_id,
                    "tags": tags or [],
                    "system": system,
                    "tool_name": tool_name,
                    "max_retries": max_retries,
                },
            )

            with self._langfuse.start_as_current_observation(
                as_type="generation",
                name=f"{trace_name}-llm",
                model=self._model,
                input=messages,
            ) as generation:
                attempt_num = 0
                try:
                    async for attempt in AsyncRetrying(
                        stop=stop_after_attempt(max_retries + 1),
                        wait=wait_exponential_jitter(initial=1, max=10, jitter=2),
                        retry=retry_if_exception(_is_retryable),
                        reraise=True,
                    ):
                        with attempt:
                            attempt_num = attempt.retry_state.attempt_number
                            create_kwargs: dict[str, Any] = {
                                "model": self._model,
                                "max_tokens": max_tokens,
                                "messages": messages,
                                "tools": tools,
                                "tool_choice": {"type": "tool", "name": tool_name},
                            }
                            if system is not None:
                                create_kwargs["system"] = system
                            response = await self._client.messages.create(
                                **create_kwargs
                            )

                    # Ищем блок tool_use с нужным именем.
                    tool_input: dict[str, Any] | None = None
                    for block in response.content:
                        if isinstance(block, ToolUseBlock) and block.name == tool_name:
                            # block.input типизирован как object — это JSON-объект
                            # инструмента; приводим к dict для удобства.
                            if isinstance(block.input, dict):
                                tool_input = block.input
                            break

                    if tool_input is None:
                        raise ValueError(
                            f"Expected tool_use block '{tool_name}', "
                            f"got: {[type(b).__name__ for b in response.content]}"
                        )

                    cost = calculate_cost(response.usage)

                    generation.update(
                        output=tool_input,
                        usage_details={
                            "input": response.usage.input_tokens,
                            "output": response.usage.output_tokens,
                        },
                        metadata={"attempts": attempt_num},
                    )
                    root_span.update(
                        output=tool_input, metadata={"attempts": attempt_num}
                    )

                    self._total_cost += cost.total_cost_usd
                    self._call_count += 1

                    return ToolCallResult(
                        tool_input=tool_input, raw=response, cost=cost
                    )

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
