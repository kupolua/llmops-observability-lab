import asyncio
import os
from datetime import datetime
from typing import Any

from anthropic import AsyncAnthropic
from anthropic.types import TextBlock, ToolUseBlock
from dotenv import load_dotenv

load_dotenv()


# ---------- 1. Описание инструмента для Claude ----------

TOOLS: list[dict[str, Any]] = [
    {
        "name": "get_current_time",
        "description": "Получить текущее время в формате ISO 8601",
        "input_schema": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "Часовой пояс, например 'Europe/Chisinau'. По умолчанию UTC.",
                }
            },
            "required": [],
        },
    },
]


# ---------- 2. Реальная реализация инструмента ----------

def get_current_time(timezone: str = "UTC") -> str:
    """Реальная функция, которую Claude хочет вызвать."""
    # Для простоты игнорируем timezone, всегда возвращаем UTC
    return datetime.utcnow().isoformat() + "Z"


# ---------- 3. Главная функция: один tool use цикл ----------

async def main() -> None:
    client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # Шаг 1: Отправляем запрос с tools
    print("=== Шаг 1: Спрашиваем Claude ===")
    response = await client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        tools=TOOLS,
        messages=[
            {"role": "user", "content": "Который сейчас час?"},
        ],
    )

    print(f"stop_reason: {response.stop_reason}")
    print(f"content blocks: {len(response.content)}")
    for i, block in enumerate(response.content):
        print(f"  [{i}] {type(block).__name__}: {block}")

    # Шаг 2: Claude должен вернуть tool_use блок
    # stop_reason будет "tool_use" вместо "end_turn"
    if response.stop_reason != "tool_use":
        print("Claude не захотел использовать инструмент, возвращает финальный ответ.")
        return

    # Извлекаем tool_use блок
    tool_use_block: ToolUseBlock | None = None
    for block in response.content:
        if isinstance(block, ToolUseBlock):
            tool_use_block = block
            break

    assert tool_use_block is not None
    print(f"\n=== Шаг 2: Claude хочет вызвать {tool_use_block.name} ===")
    print(f"  args: {tool_use_block.input}")
    print(f"  tool_use_id: {tool_use_block.id}")

    # Шаг 3: Реально исполняем функцию
    assert tool_use_block.name == "get_current_time"
    tool_args = tool_use_block.input
    assert isinstance(tool_args, dict)
    timezone = tool_args.get("timezone", "UTC")
    assert isinstance(timezone, str)

    result = get_current_time(timezone)
    print(f"\n=== Шаг 3: Результат инструмента ===")
    print(f"  {result}")

    # Шаг 4: Отправляем результат обратно Claude
    print("\n=== Шаг 4: Возвращаем результат Claude ===")
    final_response = await client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        tools=TOOLS,
        messages=[
            {"role": "user", "content": "Который сейчас час?"},
            {"role": "assistant", "content": response.content},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_block.id,
                        "content": result,
                    }
                ],
            },
        ],
    )

    # Шаг 5: Финальный ответ Claude
    print(f"stop_reason: {final_response.stop_reason}")
    for block in final_response.content:
        if isinstance(block, TextBlock):
            print(f"\n=== Финальный ответ ===")
            print(block.text)


if __name__ == "__main__":
    asyncio.run(main())