from unittest.mock import AsyncMock, MagicMock

import pytest
import httpx
from anthropic.types import TextBlock

from lf_client.client import ClaudeClient, _is_retryable

from anthropic import APIStatusError


@pytest.fixture
def mock_response() -> MagicMock:
    """Имитация ответа от Anthropic API."""
    text_block = MagicMock(spec=TextBlock)
    text_block.text = "Hello from mock!"
    text_block.type = "text"

    response = MagicMock()
    response.content = [text_block]
    response.usage = MagicMock(input_tokens=10, output_tokens=20)
    return response


@pytest.fixture
def mock_langfuse() -> MagicMock:
    """Имитация Langfuse-клиента с поддержкой context managers."""
    langfuse = MagicMock()

    # Любой start_as_current_observation возвращает context manager
    # __enter__ возвращает span/generation, __exit__ ничего не делает.
    observation_mock = MagicMock()
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=observation_mock)
    cm.__exit__ = MagicMock(return_value=False)

    langfuse.start_as_current_observation = MagicMock(return_value=cm)
    langfuse.update_current_span = MagicMock()

    return langfuse


async def test_ask_returns_text(
    mock_response: MagicMock,
    mock_langfuse: MagicMock,
) -> None:
    client = ClaudeClient(api_key="fake-key", langfuse=mock_langfuse)
    client._client.messages.create = AsyncMock(return_value=mock_response)

    result = await client.ask("Hi")

    assert result.text == "Hello from mock!"
    assert result.cost.input_tokens == 10
    assert result.cost.output_tokens == 20


async def test_total_cost_accumulates(
    mock_response: MagicMock,
    mock_langfuse: MagicMock,
) -> None:
    client = ClaudeClient(api_key="fake-key", langfuse=mock_langfuse)
    client._client.messages.create = AsyncMock(return_value=mock_response)

    await client.ask("First question")
    await client.ask("Second question")

    assert client.call_count == 2
    assert client.total_cost_usd > 0


async def test_ask_creates_observation_in_langfuse(
    mock_response: MagicMock,
    mock_langfuse: MagicMock,
) -> None:
    """Проверяем, что Langfuse был вызван правильным образом."""
    client = ClaudeClient(api_key="fake-key", langfuse=mock_langfuse)
    client._client.messages.create = AsyncMock(return_value=mock_response)

    await client.ask("Hi", trace_name="my-test-trace")

    # start_as_current_observation должен быть вызван минимум 2 раза:
    # 1) корневой span, 2) generation внутри
    assert mock_langfuse.start_as_current_observation.call_count >= 2


async def test_ask_passes_metadata(
    mock_response: MagicMock,
    mock_langfuse: MagicMock,
) -> None:
    """Проверяем, что metadata передаётся в update_current_span."""
    client = ClaudeClient(api_key="fake-key", langfuse=mock_langfuse)
    client._client.messages.create = AsyncMock(return_value=mock_response)

    await client.ask(
        "Hi",
        user_id="alice",
        session_id="session-42",
        tags=["test", "demo"],
        environment="dev",
    )

    # update_current_span должен быть вызван с metadata
    mock_langfuse.update_current_span.assert_called()
    call_kwargs = mock_langfuse.update_current_span.call_args.kwargs
    metadata = call_kwargs["metadata"]
    assert metadata["user_id"] == "alice"
    assert metadata["session_id"] == "session-42"
    assert metadata["tags"] == ["test", "demo"]
    assert metadata["environment"] == "dev"


async def test_environment_defaults_to_dev(
    mock_response: MagicMock,
    mock_langfuse: MagicMock,
) -> None:
    """Если environment не указан явно — должен быть 'dev'."""
    client = ClaudeClient(api_key="fake-key", langfuse=mock_langfuse)
    client._client.messages.create = AsyncMock(return_value=mock_response)

    await client.ask("Hi")

    call_kwargs = mock_langfuse.update_current_span.call_args.kwargs
    assert call_kwargs["metadata"]["environment"] == "dev"


def _make_api_error(status_code: int) -> APIStatusError:
    """Helper: создать APIStatusError с нужным кодом."""
    fake_request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")

    class FakeResponse:
        def __init__(self, code: int) -> None:
            self.status_code = code
            self.headers: dict[str, str] = {}
            self.request = fake_request

        def json(self) -> dict[str, str]:
            return {"error": "test"}

    return APIStatusError(
        f"Status {status_code}",
        response=FakeResponse(status_code),  # type: ignore[arg-type]
        body=None,
    )


@pytest.mark.parametrize(
    "status_code,expected",
    [
        (429, True),  # rate limit
        (529, True),  # overloaded
        (500, True),  # server error
        (502, True),  # bad gateway
        (503, True),  # service unavailable
        (504, True),  # gateway timeout
        (400, False),  # bad request - не ретраим
        (401, False),  # unauthorized
        (403, False),  # forbidden
        (404, False),  # not found
    ],
)
def test_is_retryable_status_codes(status_code: int, expected: bool) -> None:
    error = _make_api_error(status_code)
    assert _is_retryable(error) is expected


def test_is_retryable_non_api_error() -> None:
    """Обычные исключения не должны ретраиться."""
    assert _is_retryable(ValueError("test")) is False
    assert _is_retryable(KeyError("test")) is False
    assert _is_retryable(RuntimeError("test")) is False
