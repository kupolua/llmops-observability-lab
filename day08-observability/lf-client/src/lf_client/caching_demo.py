import asyncio
import os

from anthropic import AsyncAnthropic
from dotenv import load_dotenv

load_dotenv()


# Большой системный промпт (>1024 токенов нужно для кэша)
# Симулируем "тяжёлый" контекст — например, инструкции технолога
LARGE_SYSTEM_PROMPT = """Ты — опытный технолог отделочных работ.
""" + (
    """
Ты анализируешь объекты, выявляешь дефекты, строишь технологические процессы.
Ты работаешь со штукатуркой, малярными работами, гипсокартоном, напольными покрытиями.
Ты учитываешь условия эксплуатации, совместимость материалов, геометрию поверхностей.
Ты задаёшь уточняющие вопросы, если данных недостаточно.
"""
    * 30
)  # повторяем, чтобы гарантированно превысить 1024 токена


async def call_without_cache(client: AsyncAnthropic, question: str) -> dict[str, int]:
    response = await client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=100,
        system=LARGE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": question}],
    )
    usage = response.usage
    return {
        "input": usage.input_tokens,
        "output": usage.output_tokens,
        "cache_read": getattr(usage, "cache_read_input_tokens", 0) or 0,
        "cache_write": getattr(usage, "cache_creation_input_tokens", 0) or 0,
    }


async def call_with_cache(client: AsyncAnthropic, question: str) -> dict[str, int]:
    response = await client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=100,
        system=[
            {
                "type": "text",
                "text": LARGE_SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},  # ← вот магия
            }
        ],
        messages=[{"role": "user", "content": question}],
    )
    usage = response.usage
    return {
        "input": usage.input_tokens,
        "output": usage.output_tokens,
        "cache_read": getattr(usage, "cache_read_input_tokens", 0) or 0,
        "cache_write": getattr(usage, "cache_creation_input_tokens", 0) or 0,
    }


def print_usage(label: str, u: dict[str, int]) -> None:
    print(f"\n{label}:")
    print(f"  input tokens:       {u['input']}")
    print(f"  output tokens:      {u['output']}")
    print(f"  cache READ tokens:  {u['cache_read']}")
    print(f"  cache WRITE tokens: {u['cache_write']}")


async def main() -> None:
    client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    questions = [
        "Какие основные этапы штукатурных работ?",
        "Что важно при выборе грунтовки?",
        "Как избежать трещин на стыках гипсокартона?",
    ]

    print("=" * 60)
    print("БЕЗ КЭША — каждый запрос платит за весь системный промпт")
    print("=" * 60)
    for q in questions:
        u = await call_without_cache(client, q)
        print_usage(f"Q: {q[:40]}", u)

    print("\n")
    print("=" * 60)
    print("С КЭШЕМ — первый пишет кэш, остальные читают дёшево")
    print("=" * 60)
    for q in questions:
        u = await call_with_cache(client, q)
        print_usage(f"Q: {q[:40]}", u)


if __name__ == "__main__":
    asyncio.run(main())
