import asyncio
import os
from typing import Any

from anthropic import AsyncAnthropic
from anthropic.types import (
    MessageParam,
    TextBlock,
    ToolParam,
    ToolResultBlockParam,
    ToolUseBlock,
)
from dotenv import load_dotenv

load_dotenv()


# ---------- Описания tools ----------

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


# ---------- Реальные функции (хардкод) ----------


def get_weather(city: str) -> str:
    fake_data = {
        "Київ": "8°C, ясно",
        "Львів": "6°C, туман",
        "Париж": "12°C, дощ",
        "Кишинів": "15°C, хмарно",
    }
    return fake_data.get(city, "no data")


def get_population(city: str) -> str:
    fake_data = {
        "Київ": "3.0",
        "Львів": "0.7",
        "Париж": "2.1",
        "Кишинів": "0.6",
    }
    return f"{fake_data.get(city, 'no data')} млн"


def execute_tool(name: str, args: dict[str, Any]) -> str:
    if name == "get_weather":
        return get_weather(args["city"])
    if name == "get_population":
        return get_population(args["city"])
    return f"Error: unknown tool '{name}'"


# ---------- Агент без observability ----------


async def run_agent(user_prompt: str, max_iterations: int = 10) -> str:
    client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    messages: list[MessageParam] = [{"role": "user", "content": user_prompt}]

    for iteration in range(max_iterations):
        print(f"\n\033[90m--- Iteration {iteration + 1} ---\033[0m")

        response = await client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            tools=TOOLS,
            messages=messages,
        )

        # Печатаем текст ответа (если есть)
        for block in response.content:
            if isinstance(block, TextBlock):
                print(block.text)

        messages.append({"role": "assistant", "content": response.content})

        # Если модель закончила — возвращаем финальный текст
        if response.stop_reason == "end_turn":
            for block in response.content:
                if isinstance(block, TextBlock):
                    return block.text
            return ""

        # Иначе обрабатываем tool_use блоки
        if response.stop_reason == "tool_use":
            tool_results: list[ToolResultBlockParam] = []
            for block in response.content:
                if isinstance(block, ToolUseBlock):
                    args = block.input
                    assert isinstance(args, dict)
                    print(f"\033[36m  → tool {block.name}({args})\033[0m")
                    result = execute_tool(block.name, args)
                    print(f"\033[36m    result: {result}\033[0m")
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        }
                    )
            messages.append({"role": "user", "content": tool_results})

    return "max iterations reached"


async def main() -> None:
    answer = await run_agent(
        "Порівняй Київ та Париж — яка там погода і скільки людей живе?"
    )
    print(f"\n\033[32m=== Final ===\033[0m\n{answer}")


if __name__ == "__main__":
    asyncio.run(main())
