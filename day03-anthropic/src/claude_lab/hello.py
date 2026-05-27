import asyncio
import os
from dotenv import load_dotenv
from anthropic import AsyncAnthropic


load_dotenv()

# TODO: вызови client.messages.create(...)
# с model="claude-sonnet-4-5", max_tokens=1024,
# и messages=[{"role": "user", "content": "Привет, расскажи о себе в трёх предложениях"}]
#
# Распечатай response.content[0].text


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
    asyncio.run(stream_response("Привет, расскажи о себе в трёх предложениях"))
