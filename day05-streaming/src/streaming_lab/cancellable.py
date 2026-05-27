import asyncio
import os

from anthropic import AsyncAnthropic
from dotenv import load_dotenv

load_dotenv()


async def cancellable_stream(prompt: str, max_chars: int = 200) -> str:
    """Стримит ответ и отменяет его, как только накопилось max_chars символов."""
    client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    collected = ""

    async with client.messages.stream(
        model="claude-sonnet-4-5",
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        async for text_chunk in stream.text_stream:
            collected += text_chunk
            print(text_chunk, end="", flush=True)

            if len(collected) >= max_chars:
                print(
                    f"\n\033[31m[отменено: достигнут лимит {max_chars} символов]\033[0m"
                )
                # Получаем то, что успели собрать
                return collected

    return collected


async def main() -> None:
    result = await cancellable_stream(
        "Напиши очень длинное эссе про историю программирования, от Ады Лавлейс до наших дней.",
        max_chars=300,
    )
    print(f"\nСобрано символов: {len(result)}")


if __name__ == "__main__":
    asyncio.run(main())
