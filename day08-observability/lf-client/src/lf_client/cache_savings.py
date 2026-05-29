# Цены Anthropic для Sonnet (за миллион токенов)
PRICE_INPUT = 3.00
PRICE_OUTPUT = 15.00
PRICE_CACHE_WRITE = 3.75  # на 25% дороже обычного input
PRICE_CACHE_READ = 0.30  # в 10 раз дешевле


def cost_usd(
    input_tokens: int,
    output_tokens: int,
    cache_read: int = 0,
    cache_write: int = 0,
) -> float:
    return (
        input_tokens / 1_000_000 * PRICE_INPUT
        + output_tokens / 1_000_000 * PRICE_OUTPUT
        + cache_read / 1_000_000 * PRICE_CACHE_READ
        + cache_write / 1_000_000 * PRICE_CACHE_WRITE
    )


def simulate_scenario(
    system_prompt_tokens: int,
    output_tokens: int,
    num_requests: int,
) -> None:
    """Сравнить стоимость с кэшем и без на N запросах."""
    print(
        f"\nСценарий: системный промпт {system_prompt_tokens} токенов, "
        f"{num_requests} запросов, ~{output_tokens} output каждый\n"
    )

    # Без кэша: каждый запрос платит за весь системный промпт
    no_cache = num_requests * cost_usd(
        input_tokens=system_prompt_tokens + 20,  # +20 на вопрос
        output_tokens=output_tokens,
    )

    # С кэшем: первый пишет, остальные читают
    first = cost_usd(
        input_tokens=20,
        output_tokens=output_tokens,
        cache_write=system_prompt_tokens,
    )
    rest = (num_requests - 1) * cost_usd(
        input_tokens=20,
        output_tokens=output_tokens,
        cache_read=system_prompt_tokens,
    )
    with_cache = first + rest

    savings = no_cache - with_cache
    savings_pct = (savings / no_cache * 100) if no_cache > 0 else 0

    print(f"  Без кэша:  ${no_cache:.4f}")
    print(f"  С кэшем:   ${with_cache:.4f}")
    print(f"  Экономия:  ${savings:.4f} ({savings_pct:.1f}%)")


def main() -> None:
    # Разные масштабы
    simulate_scenario(system_prompt_tokens=5000, output_tokens=200, num_requests=100)
    simulate_scenario(system_prompt_tokens=5000, output_tokens=200, num_requests=1000)
    simulate_scenario(system_prompt_tokens=20000, output_tokens=200, num_requests=1000)


if __name__ == "__main__":
    main()
