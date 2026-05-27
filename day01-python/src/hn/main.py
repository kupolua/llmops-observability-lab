import asyncio
import httpx
from pydantic import BaseModel
from pydantic import TypeAdapter

_StoryIds = TypeAdapter(list[int])


class Story(BaseModel):
    title: str | None = None
    by: str | None = None
    descendants: int = 0


async def fetch_top_story_ids(client: httpx.AsyncClient) -> list[int]:
    r = await client.get("https://hacker-news.firebaseio.com/v0/topstories.json")
    r.raise_for_status()
    return _StoryIds.validate_python(r.json())


async def fetch_story(client: httpx.AsyncClient, story_id: int) -> Story:
    r = await client.get(f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json")
    r.raise_for_status()
    return Story.model_validate(r.json())


async def fetch_top_stories(
        client: httpx.AsyncClient, limit: int = 10
) -> list[Story]:
    """Высокоуровневая функция: ID → детали, всё параллельно."""
    ids = await fetch_top_story_ids(client)
    tasks = [fetch_story(client, sid) for sid in ids[:limit]]
    return await asyncio.gather(*tasks)


def format_story(story: Story) -> str:
    title = story.title or "(без заголовка)"
    by = story.by or "anonymous"
    return f"{title}\n  by {by} · {story.descendants} comments"


async def main() -> None:
    async with httpx.AsyncClient(timeout=10.0) as client:
        stories = await fetch_top_stories(client, limit=10)

    for s in stories:
        print(format_story(s) + "\n")


if __name__ == "__main__":
    asyncio.run(main())