from typing import Any

import pytest

from lf_client.streaming_agent import execute_tool, get_population, get_weather


@pytest.mark.parametrize(
    "city,expected_substring",
    [
        ("Київ", "°C"),
        ("Львів", "°C"),
        ("Париж", "°C"),
        ("Атлантида", "no data"),
    ],
)
def test_get_weather(city: str, expected_substring: str) -> None:
    assert expected_substring in get_weather(city)


@pytest.mark.parametrize(
    "city,expected_substring",
    [
        ("Київ", "млн"),
        ("Львів", "млн"),
        ("Кишинів", "млн"),
        ("Атлантида", "no data"),
    ],
)
def test_get_population(city: str, expected_substring: str) -> None:
    assert expected_substring in get_population(city)


@pytest.mark.parametrize(
    "name,args,expected_substring",
    [
        ("get_weather", {"city": "Київ"}, "°C"),
        ("get_population", {"city": "Київ"}, "млн"),
        ("unknown_tool", {}, "Error"),
    ],
)
def test_execute_tool(name: str, args: dict[str, Any], expected_substring: str) -> None:
    result = execute_tool(name, args)
    assert expected_substring in result
