import pytest
import httpx
from pytest_httpx import HTTPXMock

from hn.main import (
    Story,
    fetch_top_story_ids,
    fetch_story,
    fetch_top_stories,
    format_story,
)


# ---------- Тесты чистых функций (без I/O) ----------


def test_format_story_with_all_fields() -> None:
    story = Story(title="Hello", by="alice", descendants=5)
    result = format_story(story)
    assert result == "Hello\n  by alice · 5 comments"


def test_format_story_with_missing_title() -> None:
    story = Story(title=None, by="alice", descendants=5)
    result = format_story(story)
    assert "(без заголовка)" in result


def test_format_story_with_missing_author() -> None:
    story = Story(title="Hello", by=None, descendants=0)
    result = format_story(story)
    assert "anonymous" in result


def test_story_defaults() -> None:
    """Пустой JSON должен валидироваться: все поля опциональны."""
    story = Story.model_validate({})
    assert story.title is None
    assert story.by is None
    assert story.descendants == 0

def test_format_story_with_zero_comments() -> None:
    """format_story должен корректно выводить 0 для пустых комментариев."""
    story = Story(title="Hello", by="alice", descendants=0)
    result = format_story(story)
    assert "0 comments" in result


# ---------- Тесты функций с I/O (с моками) ----------


async def test_fetch_top_story_ids(
        httpx_mock: HTTPXMock, http_client: httpx.AsyncClient
) -> None:
    httpx_mock.add_response(
        url="https://hacker-news.firebaseio.com/v0/topstories.json",
        json=[100, 200, 300],
    )
    ids = await fetch_top_story_ids(http_client)
    assert ids == [100, 200, 300]


async def test_fetch_story(
        httpx_mock: HTTPXMock, http_client: httpx.AsyncClient
) -> None:
    httpx_mock.add_response(
        url="https://hacker-news.firebaseio.com/v0/item/42.json",
        json={"title": "Answer", "by": "deep_thought", "descendants": 7},
    )
    story = await fetch_story(http_client, 42)
    assert story.title == "Answer"
    assert story.by == "deep_thought"
    assert story.descendants == 7


async def test_fetch_top_stories_full_flow(
        httpx_mock: HTTPXMock, http_client: httpx.AsyncClient
) -> None:
    """Интеграционный тест: ID → детали → готовый список Story."""
    httpx_mock.add_response(
        url="https://hacker-news.firebaseio.com/v0/topstories.json",
        json=[1, 2, 3],
    )
    httpx_mock.add_response(
        url="https://hacker-news.firebaseio.com/v0/item/1.json",
        json={"title": "First", "by": "a", "descendants": 1},
    )
    httpx_mock.add_response(
        url="https://hacker-news.firebaseio.com/v0/item/2.json",
        json={"title": "Second", "by": "b", "descendants": 2},
    )
    httpx_mock.add_response(
        url="https://hacker-news.firebaseio.com/v0/item/3.json",
        json={"title": "Third", "by": "c", "descendants": 3},
    )
    stories = await fetch_top_stories(http_client, limit=3)
    assert len(stories) == 3
    assert stories[0].title == "First"
    assert stories[2].descendants == 3


async def test_fetch_story_handles_http_error(
        httpx_mock: HTTPXMock, http_client: httpx.AsyncClient
) -> None:
    """4xx/5xx должны бросать HTTPStatusError."""
    httpx_mock.add_response(
        url="https://hacker-news.firebaseio.com/v0/item/999.json",
        status_code=500,
    )
    # async with httpx.AsyncClient() as client:
    with pytest.raises(httpx.HTTPStatusError):
        await fetch_story(http_client, 999)
