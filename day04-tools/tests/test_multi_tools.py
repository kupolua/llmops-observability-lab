import pytest

from tools_lab.multi_tools import execute_tool, get_weather, get_population


# ---------- Простые тесты на чистые функции ----------

def test_get_weather_known_city() -> None:
    assert "°C" in get_weather("Париж")


def test_get_weather_unknown_city() -> None:
    assert get_weather("Атлантида") == "нет данных"


def test_get_population_known_city() -> None:
    assert "млн" in get_population("Москва")


# ---------- Тесты на диспетчер execute_tool ----------

def test_execute_tool_weather() -> None:
    result = execute_tool("get_weather", {"city": "Париж"})
    assert "°C" in result


def test_execute_tool_population() -> None:
    result = execute_tool("get_population", {"city": "Кишинёв"})
    assert "млн" in result


def test_execute_tool_unknown() -> None:
    result = execute_tool("nonexistent_tool", {})
    assert "error" in result.lower() or "unknown" in result.lower()