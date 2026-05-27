from anthropic.types import Usage
from pydantic import BaseModel


# Цены за миллион токенов (USD) для Claude Sonnet 4.5
SONNET_INPUT_PRICE = 3.00
SONNET_OUTPUT_PRICE = 15.00


class CallCost(BaseModel):
    input_tokens: int
    output_tokens: int
    input_cost_usd: float
    output_cost_usd: float
    total_cost_usd: float


def calculate_cost(usage: Usage) -> CallCost:
    input_cost = usage.input_tokens / 1_000_000 * SONNET_INPUT_PRICE
    output_cost = usage.output_tokens / 1_000_000 * SONNET_OUTPUT_PRICE
    return CallCost(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        input_cost_usd=round(input_cost, 6),
        output_cost_usd=round(output_cost, 6),
        total_cost_usd=round(input_cost + output_cost, 6),
    )
