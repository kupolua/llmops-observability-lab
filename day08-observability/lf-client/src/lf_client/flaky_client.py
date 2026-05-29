import asyncio
import os
import random
import httpx
from typing import Any

from anthropic import APIStatusError, AsyncAnthropic
from dotenv import load_dotenv

load_dotenv()


class FlakyAnthropicClient:
    """Обёртка над AsyncAnthropic с искусственными сбоями для тестирования reliability.

    Полезно для разработки и интеграционных тестов — позволяет проверить retry,
    circuit breakers, fallbacks без необходимости реально ронять Anthropic API.
    """

    def __init__(
        self,
        api_key: str,
        failure_rate: float = 0.3,
        failure_codes: tuple[int, ...] = (429, 529, 500),
    ) -> None:
        self._client = AsyncAnthropic(api_key=api_key)
        self._failure_rate = failure_rate
        self._failure_codes = failure_codes
        self._call_count = 0
        self._injected_failures = 0

    @property
    def messages(self) -> "FlakyMessages":
        return FlakyMessages(self)


class FlakyMessages:
    """Proxy для messages, который иногда «ломается» искусственно."""

    def __init__(self, parent: FlakyAnthropicClient) -> None:
        self._parent = parent

    async def create(self, **kwargs: Any) -> Any:
        self._parent._call_count += 1

        # Случайно бросаем сбой
        if random.random() < self._parent._failure_rate:
            code = random.choice(self._parent._failure_codes)
            self._parent._injected_failures += 1
            print(
                f"\033[91m  [flaky] injecting {code} on call #{self._parent._call_count}\033[0m"
            )
            raise APIStatusError(
                f"Simulated failure {code}",
                response=_fake_response(code),
                body=None,
            )

        # Иначе пропускаем настоящий вызов
        return await self._parent._client.messages.create(**kwargs)


class FlakyAsyncAnthropic:
    """Drop-in замена для AsyncAnthropic с симулированными сбоями."""

    def __init__(
        self,
        api_key: str,
        failure_rate: float = 0.3,
        failure_codes: tuple[int, ...] = (429, 529, 500),
    ) -> None:
        self._real = AsyncAnthropic(api_key=api_key)
        self._failure_rate = failure_rate
        self._failure_codes = failure_codes
        self.call_count = 0
        self.injected_failures = 0

        # Эмулируем структуру AsyncAnthropic: client.messages.create(...)
        self.messages = _FlakyMessages(self)


class _FlakyMessages:
    def __init__(self, parent: FlakyAsyncAnthropic) -> None:
        self._parent = parent

    async def create(self, **kwargs: Any) -> Any:
        self._parent.call_count += 1

        if random.random() < self._parent._failure_rate:
            code = random.choice(self._parent._failure_codes)
            self._parent.injected_failures += 1
            print(
                f"\033[91m  [flaky] injecting {code} on call #{self._parent.call_count}\033[0m"
            )
            raise APIStatusError(
                f"Simulated failure {code}",
                response=_fake_response(code),
                body=None,
            )

        return await self._parent._real.messages.create(**kwargs)


def _fake_response(status_code: int) -> Any:
    """Создать минимальный httpx.Response для APIStatusError."""

    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(
        status_code=status_code,
        headers={"content-type": "application/json"},
        content=b'{"error": {"type": "simulated", "message": "fake error"}}',
        request=request,
    )
    return response


async def main() -> None:
    """Демонстрация: 10 вызовов с 50% failure rate — без retry."""
    client = FlakyAnthropicClient(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        failure_rate=0.5,
    )

    successes = 0
    failures = 0

    for i in range(10):
        try:
            await client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=50,
                messages=[{"role": "user", "content": "Скажи 'привет'"}],
            )
            successes += 1
            print(f"\033[92m  [success] call {i + 1}\033[0m")
        except APIStatusError as e:
            failures += 1
            print(f"\033[91m  [failed]  call {i + 1}: {e}\033[0m")

    print(f"\nИтого: {successes} успехов, {failures} сбоев из 10 вызовов")


if __name__ == "__main__":
    asyncio.run(main())
