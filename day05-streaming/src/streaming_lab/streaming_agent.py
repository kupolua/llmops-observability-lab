import asyncio
import os
from typing import Any

from anthropic import AsyncAnthropic
from anthropic.types import (
    MessageParam,
    TextBlock,
    ToolParam,
    ToolUseBlock,
    ToolResultBlockParam,
)
from dotenv import load_dotenv

load_dotenv()


# ---------- Tools (повторяем со вчерашнего дня) ----------

TOOLS: list[ToolParam] = [
    {
        "name": "get_weather",
        "description": "Получить текущую погоду в городе",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "Название города"},
            },
            "required": ["city"],
        },
    },
    {
        "name": "get_population",
        "description": "Получить население города в миллионах",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "Название города"},
            },
            "required": ["city"],
        },
    },
]


def get_weather(city: str) -> str:
    fake_data = {
        "Кишинёв": "15°C, облачно",
        "Париж": "12°C, дождь",
        "Киев": "-3°C, снег",
    }
    return fake_data.get(city, "no data")


def get_population(city: str) -> str:
    fake_data = {"Кишинёв": "0.6", "Париж": "2.1", "Киев": "12.5"}
    return f"{fake_data.get(city, 'no data')} млн"


def execute_tool(name: str, args: dict[str, Any]) -> str:
    if name == "get_weather":
        return get_weather(args["city"])
    if name == "get_population":
        return get_population(args["city"])
    return f"Error: unknown tool '{name}'"


# ---------- Streaming-агент ----------


async def run_streaming_agent(user_prompt: str) -> str:
    """Агент с tool use, который стримит текстовые ответы пользователю."""
    client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # messages: list[dict[str, Any]] = [{"role": "user", "content": user_prompt}]
    messages: list[MessageParam] = [{"role": "user", "content": user_prompt}]
    max_iterations = 10

    for iteration in range(max_iterations):
        print(f"\n\033[90m--- Итерация {iteration + 1} ---\033[0m")

        # Стримим ответ Claude
        async with client.messages.stream(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            tools=TOOLS,
            messages=messages,
        ) as stream:
            # Печатаем текст по мере поступления
            async for text_chunk in stream.text_stream:
                print(text_chunk, end="", flush=True)

            # Получаем итоговое собранное сообщение
            final = await stream.get_final_message()

        print()  # перенос строки после стрима

        # Добавляем ответ в историю
        messages.append({"role": "assistant", "content": final.content})

        # Если модель закончила — возвращаем накопленный текст
        if final.stop_reason == "end_turn":
            text_parts: list[str] = []
            for block in final.content:
                if isinstance(block, TextBlock):
                    text_parts.append(block.text)
            return "\n".join(text_parts)

        # Иначе обрабатываем tool_use блоки
        if final.stop_reason == "tool_use":
            # tool_results: list[dict[str, Any]] = []
            tool_results: list[ToolResultBlockParam] = []
            for block in final.content:
                if isinstance(block, ToolUseBlock):
                    args = block.input
                    assert isinstance(args, dict)
                    print(f"\033[36m  → вызываю {block.name}({args})\033[0m")
                    result = execute_tool(block.name, args)
                    print(f"\033[36m    результат: {result}\033[0m")
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        }
                    )

            messages.append({"role": "user", "content": tool_results})

    return "Достигнут лимит итераций"


async def main() -> None:
    answer = await run_streaming_agent(
        "Сравни Париж и Киев — какая там погода и сколько людей живёт? "
        "Дай развёрнутый сравнительный ответ."
    )
    print(f"\n\033[32m=== Финальный ответ ===\033[0m\n{answer}")


if __name__ == "__main__":
    asyncio.run(main())
