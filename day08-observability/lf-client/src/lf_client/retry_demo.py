import asyncio
import os

from anthropic import APIStatusError
from dotenv import load_dotenv
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from lf_client.flaky_client import FlakyAnthropicClient

load_dotenv()


def is_retryable_anthropic_error(exception: BaseException) -> bool:
    """Решаем, стоит ли ретраить эту ошибку."""
    if not isinstance(exception, APIStatusError):
        return False
    # Ретраим только временные ошибки: rate limits, overload, server errors
    return exception.response.status_code in {429, 529, 500, 502, 503, 504}


async def call_with_retry(client: FlakyAnthropicClient, prompt: str) -> str:
    """Вызов с retry, exponential backoff и jitter."""

    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(5),
        wait=wait_exponential_jitter(initial=1, max=10, jitter=2),
        retry=retry_if_exception(is_retryable_anthropic_error),
        reraise=True,
    ):
        with attempt:
            attempt_num = attempt.retry_state.attempt_number
            print(f"\033[93m  [attempt {attempt_num}]\033[0m", end=" ")

            response = await client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=100,
                messages=[{"role": "user", "content": prompt}],
            )

            text_block = response.content[0]
            return text_block.text if hasattr(text_block, "text") else ""

    return ""  # unreachable, для mypy


async def main() -> None:
    # Высокий failure_rate — чтобы retry точно сработал
    client = FlakyAnthropicClient(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        failure_rate=0.6,
    )

    questions = [
        "Скажи 'один' одним словом",
        "Скажи 'два' одним словом",
        "Скажи 'три' одним словом",
        "Скажи 'четыре' одним словом",
        "Скажи 'пять' одним словом",
    ]

    successes = 0
    final_failures = 0

    for i, q in enumerate(questions, 1):
        print(f"\n--- Вопрос {i}: {q} ---")
        try:
            answer = await call_with_retry(client, q)
            print(f"\033[92m  [final answer] {answer.strip()}\033[0m")
            successes += 1
        except RetryError:
            print("\033[91m  [exhausted] retry лимит исчерпан\033[0m")
            final_failures += 1

    print(
        f"\nИтого: {successes} успехов, {final_failures} провалов после retry "
        f"({client._injected_failures} симулированных сбоев всего)"
    )


if __name__ == "__main__":
    asyncio.run(main())
