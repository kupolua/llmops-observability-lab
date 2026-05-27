import asyncio
import os
from typing import Any

from anthropic import AsyncAnthropic
from anthropic.types import TextBlock, ToolUseBlock
from dotenv import load_dotenv

load_dotenv()


# ---------- Описания tools ----------

TOOLS: list[dict[str, Any]] = [
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


# ---------- Реализации (хардкод для простоты) ----------

def get_weather(city: str) -> str:
    fake_data = {
        "Кишинёв": "15°C, облачно",
        "Париж": "12°C, дождь",
        "Москва": "-3°C, снег",
    }
    return fake_data.get(city, "нет данных")


def get_population(city: str) -> str:
    fake_data = {
        "Кишинёв": "0.6",
        "Париж": "2.1",
        "Москва": "12.5",
    }
    return f"{fake_data.get(city, 'нет данных')} млн"


# ---------- Диспетчер: маппинг имени → реальная функция ----------

def execute_tool(name: str, args: dict[str, Any]) -> str:
    """Выполнить инструмент по имени с переданными аргументами."""
    if name == "get_weather":
        return get_weather(args["city"])
    elif name == "get_population":
        return get_population(args["city"])
    else:
        return f"Error: unknown tool '{name}'"


# ---------- Цикл агента ----------

async def run_agent(user_prompt: str) -> str:
    """Запустить агента до получения финального ответа.

    Может потребоваться несколько итераций tool use.
    """
    client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # История диалога. Будем дописывать сюда новые сообщения.
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": user_prompt},
    ]

    max_iterations = 10
    for iteration in range(max_iterations):
        print(f"\n--- Итерация {iteration + 1} ---")

        response = await client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            tools=TOOLS,
            messages=messages,
        )

        print(f"stop_reason: {response.stop_reason}")

        # Сохраняем ответ ассистента в историю
        messages.append({"role": "assistant", "content": response.content})

        # Если Claude закончил без tool use — это финальный ответ
        if response.stop_reason == "end_turn":
            for block in response.content:
                if isinstance(block, TextBlock):
                    return block.text
            return ""  # на всякий случай

        # Обработка tool_use: собираем все вызовы инструментов из ответа
        tool_results = []

        for block in response.content:
            if isinstance(block, ToolUseBlock):
                print(f"  -> Вызов инструмента: {block.name}, аргументы: {block.input}")

                # Выполняем инструмент и получаем результат
                result = execute_tool(block.name, block.input)

                print(f"  <- Результат: {result}")

                # Формируем tool_result-блок для этого вызова
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

        # Добавляем ОДНО сообщение role=user со всеми результатами
        if tool_results:
            messages.append({
                "role": "user",
                "content": tool_results,
            })

    return "Достигнут лимит итераций"


async def main() -> None:
    answer = await run_agent(
        "Сравни Париж и Москву — какая там погода и сколько людей живёт?"
    )
    print(f"\n=== Финальный ответ ===\n{answer}")


if __name__ == "__main__":
    asyncio.run(main())
