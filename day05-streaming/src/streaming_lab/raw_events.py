import asyncio
import os

from anthropic import AsyncAnthropic
from dotenv import load_dotenv

load_dotenv()


async def main() -> None:
    client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    async with client.messages.stream(
        model="claude-sonnet-4-5",
        max_tokens=256,
        messages=[
            {
                "role": "user",
                "content": "Назови три преимущества Python в одно предложение каждое.",
            }
        ],
    ) as stream:
        async for event in stream:
            event_type = type(event).__name__
            print(f"[{event_type}]")

            # Для разных типов событий показываем разные поля
            if hasattr(event, "delta"):
                print(f"  delta: {event.delta}")
            if hasattr(event, "content_block"):
                print(f"  content_block: {event.content_block}")
            if hasattr(event, "usage"):
                print(f"  usage: {event.usage}")
            if hasattr(event, "message"):
                print(f"  message_id: {event.message.id}")


if __name__ == "__main__":
    asyncio.run(main())
