import asyncio
import os
from dotenv import load_dotenv
from anthropic import AsyncAnthropic

load_dotenv()


async def stream_response(prompt: str) -> None:
    client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    async with client.messages.stream(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        async for text in stream.text_stream:
            print(text, end="", flush=True)
    print()  # перенос строки в конце


if __name__ == "__main__":
    asyncio.run(stream_response("Объясни, что такое vector embeddings, простым языком"))
