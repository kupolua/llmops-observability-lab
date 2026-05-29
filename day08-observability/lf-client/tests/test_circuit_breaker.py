import asyncio

import pytest

from lf_client.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerError,
    CircuitState,
)


async def _failing() -> None:
    raise RuntimeError("boom")


async def _succeeding() -> str:
    return "ok"


async def test_starts_closed() -> None:
    cb = CircuitBreaker()
    assert cb.state == CircuitState.CLOSED


async def test_opens_after_threshold() -> None:
    cb = CircuitBreaker(failure_threshold=3)
    for _ in range(3):
        with pytest.raises(RuntimeError):
            await cb.call(_failing)
    assert cb.state == CircuitState.OPEN


async def test_rejects_when_open() -> None:
    cb = CircuitBreaker(failure_threshold=2)
    for _ in range(2):
        with pytest.raises(RuntimeError):
            await cb.call(_failing)
    # Теперь цепь открыта — следующий вызов отклонён без вызова функции
    with pytest.raises(CircuitBreakerError):
        await cb.call(_failing)


async def test_success_resets_failures() -> None:
    cb = CircuitBreaker(failure_threshold=3)
    # 2 ошибки
    for _ in range(2):
        with pytest.raises(RuntimeError):
            await cb.call(_failing)
    # успех сбрасывает счётчик
    result = await cb.call(_succeeding)
    assert result == "ok"
    assert cb.state == CircuitState.CLOSED


async def test_half_open_after_timeout() -> None:
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.2)
    for _ in range(2):
        with pytest.raises(RuntimeError):
            await cb.call(_failing)
    assert cb.state == CircuitState.OPEN
    # ждём recovery_timeout
    await asyncio.sleep(0.25)
    assert cb.state == CircuitState.HALF_OPEN  # type: ignore[comparison-overlap]
