import pytest

from streaming_lab.streaming_agent import (
    execute_tool,
    get_weather,
    get_population,
)


def test_get_weather_known() -> None:
    assert "°C" in get_weather("Киев")


def test_get_weather_unknown() -> None:
    assert get_weather("Атлантида") == "no data"


def test_get_population_known() -> None:
    assert "млн" in get_population("Париж")


@pytest.mark.parametrize(
    "name,args,expected_substring",
    [
        ("get_weather", {"city": "Париж"}, "°C"),
        ("get_population", {"city": "Киев"}, "млн"),
        ("unknown_tool", {}, "Error"),
    ],
)
# def test_execute_tool(name: str, args: dict, expected_substring: str) -> None:
def test_execute_tool(name: str, args: dict[str, str], expected_substring: str) -> None:
    result = execute_tool(name, args)
    assert expected_substring in result
