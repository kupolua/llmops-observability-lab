from unittest.mock import MagicMock

from lf_client.usage import calculate_cost


def test_calculate_cost_with_cache_read() -> None:
    usage = MagicMock()
    usage.input_tokens = 20
    usage.output_tokens = 100
    usage.cache_read_input_tokens = 5000
    usage.cache_creation_input_tokens = 0

    cost = calculate_cost(usage)

    assert cost.cache_read_tokens == 5000
    assert cost.cache_write_tokens == 0
    # cache read дешёвый — общая стоимость должна быть мала
    assert cost.total_cost_usd < 0.01


def test_calculate_cost_with_cache_write() -> None:
    usage = MagicMock()
    usage.input_tokens = 20
    usage.output_tokens = 100
    usage.cache_read_input_tokens = 0
    usage.cache_creation_input_tokens = 5000

    cost = calculate_cost(usage)

    assert cost.cache_write_tokens == 5000
    # cache write дороже read
    assert cost.total_cost_usd > 0


def test_calculate_cost_without_cache() -> None:
    """Базовый случай: без кэша cache-поля должны быть 0."""
    usage = MagicMock()
    usage.input_tokens = 1000
    usage.output_tokens = 500
    usage.cache_read_input_tokens = 0
    usage.cache_creation_input_tokens = 0

    cost = calculate_cost(usage)

    assert cost.cache_read_tokens == 0
    assert cost.cache_write_tokens == 0
    assert cost.input_tokens == 1000
    assert cost.output_tokens == 500
    assert cost.total_cost_usd > 0
