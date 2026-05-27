import asyncio
import os

from anthropic import AsyncAnthropic
from dotenv import load_dotenv

load_dotenv()


async def main() -> None:
    client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    async with client.messages.stream(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": "Расскажи в трёх абзацах про историю Молдовы.",
            }
        ],
    ) as stream:
        async for text_chunk in stream.text_stream:
            print(text_chunk, end="", flush=True)
        print()  # перенос строки в конце

        # После окончания стрима можно получить полный ответ
        final_message = await stream.get_final_message()
        print(
            f"\n---\nИтого токенов: {final_message.usage.input_tokens} in / {final_message.usage.output_tokens} out"
        )
        print(f"stop_reason: {final_message.stop_reason}")


if __name__ == "__main__":
    asyncio.run(main())
