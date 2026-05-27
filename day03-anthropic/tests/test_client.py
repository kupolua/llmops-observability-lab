from unittest.mock import AsyncMock, MagicMock
import pytest
from claude_lab.client import ClaudeClient
from anthropic.types import TextBlock


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


async def test_ask_returns_text(mock_response: MagicMock) -> None:
    client = ClaudeClient(api_key="fake-key")
    # Подменяем внутренний клиент моком
    client._client.messages.create = AsyncMock(return_value=mock_response)

    result = await client.ask("Hi")

    assert result.text == "Hello from mock!"
    assert result.cost.input_tokens == 10
    assert result.cost.output_tokens == 20


async def test_total_cost_accumulates(mock_response: MagicMock) -> None:
    client = ClaudeClient(api_key="fake-key")
    client._client.messages.create = AsyncMock(return_value=mock_response)

    await client.ask("Hi")
    await client.ask("Hi")

    assert client.call_count == 2
    assert client.total_cost_usd > 0
