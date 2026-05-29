import time
from enum import Enum
from typing import Any


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerError(Exception):
    """Бросается, когда цепь разомкнута и запрос отклонён без вызова."""


class CircuitBreaker:
    """Простой circuit breaker для защиты от длительных отказов.

    CLOSED: запросы проходят, считаем подряд идущие ошибки.
    OPEN: после failure_threshold ошибок — мгновенно отклоняем запросы.
    HALF_OPEN: после recovery_timeout — пробуем один запрос.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
    ) -> None:
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float = 0.0

    @property
    def state(self) -> CircuitState:
        # Если OPEN и прошёл timeout — переходим в HALF_OPEN
        if self._state == CircuitState.OPEN:
            elapsed = time.monotonic() - self._last_failure_time
            if elapsed >= self._recovery_timeout:
                self._state = CircuitState.HALF_OPEN
        return self._state

    def _record_success(self) -> None:
        self._failure_count = 0
        self._state = CircuitState.CLOSED

    def _record_failure(self) -> None:
        self._failure_count += 1
        self._last_failure_time = time.monotonic()
        if self._failure_count >= self._failure_threshold:
            self._state = CircuitState.OPEN

    async def call(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        """Выполнить func под защитой circuit breaker."""
        current_state = self.state  # триггерит переход OPEN → HALF_OPEN

        if current_state == CircuitState.OPEN:
            raise CircuitBreakerError(
                f"Circuit is OPEN (failures: {self._failure_count}). "
                f"Retry after {self._recovery_timeout}s."
            )

        try:
            result = await func(*args, **kwargs)
            self._record_success()
            return result
        except Exception:
            self._record_failure()
            raise
