import asyncio
import os

from dotenv import load_dotenv
from langfuse import get_client

from lf_client.client import ClaudeClient
from lf_client.flaky_client import FlakyAsyncAnthropic

load_dotenv()


async def main() -> None:
    langfuse = get_client()

    # Создаём flaky-клиент один раз
    flaky_anthropic = FlakyAsyncAnthropic(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        failure_rate=0.5,
    )

    # Прокидываем его в ClaudeClient через DI
    client = ClaudeClient(
        langfuse=langfuse,
        anthropic_client=flaky_anthropic,  # type: ignore[arg-type]
    )

    questions = [
        "Скажи 'раз' одним словом",
        "Скажи 'два' одним словом",
        "Скажи 'три' одним словом",
        "Скажи 'четыре' одним словом",
        "Скажи 'пять' одним словом",
    ]

    successes = 0
    failures = 0

    for i, q in enumerate(questions, 1):
        print(f"\n--- Вопрос {i}: {q} ---")
        try:
            result = await client.ask(
                q,
                trace_name=f"resilient-{i}",
                user_id="pavlo",
                tags=["reliability-test"],
                max_retries=5,
            )
            print(f"\033[92m  [success] {result.text.strip()}\033[0m")
            successes += 1
        except Exception as e:
            print(f"\033[91m  [failed]  {type(e).__name__}: {e}\033[0m")
            failures += 1

    print(f"\nИтого: {successes} успехов, {failures} провалов")
    print(
        f"Симулированные сбои: {flaky_anthropic.injected_failures} "
        f"из {flaky_anthropic.call_count} попыток"
    )

    langfuse.flush()


if __name__ == "__main__":
    asyncio.run(main())
