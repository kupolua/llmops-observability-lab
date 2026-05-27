import asyncio
import os

from anthropic import AsyncAnthropic
from anthropic.types import (
    MessageParam,
    TextBlock,
    ToolResultBlockParam,
    ToolUseBlock,
)
from dotenv import load_dotenv
from langfuse import get_client

# Переиспользуем tools и функции из соседнего модуля
from lf_client.streaming_agent import TOOLS, execute_tool

load_dotenv()


async def run_observed_agent(
    user_prompt: str,
    max_iterations: int = 10,
    trace_name: str = "agent-run",
) -> str:
    """Запустить агента, трекая каждую итерацию и каждый tool call в Langfuse."""
    anthropic_client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    langfuse = get_client()

    messages: list[MessageParam] = [{"role": "user", "content": user_prompt}]

    # Корневой trace — обертка для всего запуска агента
    with langfuse.start_as_current_observation(
        as_type="span",
        name=trace_name,
        input=user_prompt,
    ) as root_span:
        langfuse.update_current_span(
            metadata={"environment": "dev", "agent_type": "weather-comparison"},
        )

        final_text = ""

        for iteration in range(max_iterations):
            print(f"\n\033[90m--- Iteration {iteration + 1} ---\033[0m")

            # Generation: один вызов LLM
            with langfuse.start_as_current_observation(
                as_type="generation",
                name=f"iteration-{iteration + 1}",
                model="claude-sonnet-4-5",
                input=messages,
            ) as gen:
                response = await anthropic_client.messages.create(
                    model="claude-sonnet-4-5",
                    max_tokens=1024,
                    tools=TOOLS,
                    messages=messages,
                )

                # Извлекаем текстовую часть для отображения
                response_text = ""
                for block in response.content:
                    if isinstance(block, TextBlock):
                        response_text += block.text

                gen.update(
                    output=response_text or "(tool_use only)",
                    usage_details={
                        "input": response.usage.input_tokens,
                        "output": response.usage.output_tokens,
                    },
                    metadata={"stop_reason": response.stop_reason},
                )

                if response_text:
                    print(response_text)

            # Добавляем ответ ассистента в историю
            messages.append({"role": "assistant", "content": response.content})

            # Если модель закончила — выходим
            if response.stop_reason == "end_turn":
                final_text = response_text
                break

            # Иначе обрабатываем все tool_use блоки
            if response.stop_reason == "tool_use":
                tool_results: list[ToolResultBlockParam] = []

                for block in response.content:
                    if isinstance(block, ToolUseBlock):
                        args = block.input
                        assert isinstance(args, dict)

                        # Span на каждый tool call
                        with langfuse.start_as_current_observation(
                            as_type="span",
                            name=f"tool: {block.name}",
                            input=args,
                        ) as tool_span:
                            print(f"\033[36m  → tool {block.name}({args})\033[0m")
                            result = execute_tool(block.name, args)
                            print(f"\033[36m    result: {result}\033[0m")

                            tool_span.update(
                                output=result,
                                metadata={"tool_use_id": block.id},
                            )

                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": result,
                            }
                        )

                messages.append({"role": "user", "content": tool_results})

        else:
            final_text = "max iterations reached"

        # Записываем финальный output на корневой span
        root_span.update(output=final_text)

    langfuse.flush()
    return final_text


async def main() -> None:
    answer = await run_observed_agent(
        "Я плануюю поїздку. Розкажи мені про погоду і населення трьох міст: Київ, Львів, Кишинів. Зроби висновок, де найкомфортніше зараз.",
        trace_name="weather-comparison-demo",
    )
    print(f"\n\033[32m=== Final ===\033[0m\n{answer}")
    print("\n→ Откройте Langfuse: http://localhost:3000")


if __name__ == "__main__":
    asyncio.run(main())
