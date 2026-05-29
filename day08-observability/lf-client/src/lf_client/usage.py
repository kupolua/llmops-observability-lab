from anthropic.types import Usage
from pydantic import BaseModel


# Цены за миллион токенов (USD) для Claude Sonnet 4.5
SONNET_INPUT_PRICE = 3.00
SONNET_OUTPUT_PRICE = 15.00
SONNET_CACHE_WRITE_PRICE = 3.75
SONNET_CACHE_READ_PRICE = 0.30


class CallCost(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    input_cost_usd: float
    output_cost_usd: float
    total_cost_usd: float


def calculate_cost(usage: Usage) -> CallCost:
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0

    input_cost = usage.input_tokens / 1_000_000 * SONNET_INPUT_PRICE
    output_cost = usage.output_tokens / 1_000_000 * SONNET_OUTPUT_PRICE
    cache_read_cost = cache_read / 1_000_000 * SONNET_CACHE_READ_PRICE
    cache_write_cost = cache_write / 1_000_000 * SONNET_CACHE_WRITE_PRICE

    total = input_cost + output_cost + cache_read_cost + cache_write_cost

    return CallCost(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
        input_cost_usd=round(input_cost, 6),
        output_cost_usd=round(output_cost, 6),
        total_cost_usd=round(total, 6),
    )
