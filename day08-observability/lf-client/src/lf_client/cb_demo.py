import asyncio
import os

from dotenv import load_dotenv

from lf_client.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerError,
)
from lf_client.flaky_client import FlakyAsyncAnthropic

from typing import Any

load_dotenv()


async def main() -> None:
    # Полностью «сломанный» клиент — 100% сбоев
    broken = FlakyAsyncAnthropic(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        failure_rate=1.0,  # всегда падает
    )

    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=5.0)

    async def make_call() -> Any:
        return await broken.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=50,
            messages=[{"role": "user", "content": "test"}],
        )

    # 10 попыток подряд
    for i in range(1, 11):
        state_before = cb.state
        try:
            await cb.call(make_call)
            print(f"  call {i}: SUCCESS (state={state_before.value})")
        except CircuitBreakerError:
            print(
                f"\033[93m  call {i}: REJECTED by circuit breaker (state={state_before.value})\033[0m"
            )
        except Exception:
            print(
                f"\033[91m  call {i}: FAILED through (state={state_before.value})\033[0m"
            )

        await asyncio.sleep(0.5)

    print(f"\nФинальное состояние: {cb.state.value}")
    print("Жди 5 секунд (recovery_timeout)...")
    await asyncio.sleep(5)
    print(f"Состояние после ожидания: {cb.state.value}  ← должно быть half_open")


if __name__ == "__main__":
    asyncio.run(main())
