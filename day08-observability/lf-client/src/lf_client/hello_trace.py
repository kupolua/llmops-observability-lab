import os

from anthropic import Anthropic
from dotenv import load_dotenv
from langfuse import get_client

load_dotenv()


def main() -> None:
    # Клиенты
    anthropic_client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    # get_client() читает LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST из env
    langfuse = get_client()

    prompt = "Скажи 'привет' тремя разными способами"

    # Контекстный менеджер — это новый способ создания generation
    with langfuse.start_as_current_observation(
        as_type="generation",
        name="claude-greeting",
        model="claude-sonnet-4-5",
        input=[{"role": "user", "content": prompt}],
    ) as generation:
        # Реальный вызов Anthropic
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )

        # Извлекаем текст из первого блока
        text_block = response.content[0]
        text = text_block.text if hasattr(text_block, "text") else ""

        # Записываем результат и токены
        generation.update(
            output=text,
            usage_details={
                "input": response.usage.input_tokens,
                "output": response.usage.output_tokens,
            },
        )

    # Принудительный flush — без него скрипт может завершиться до отправки
    langfuse.flush()

    print(text)
    print(
        f"\nТокенов: {response.usage.input_tokens} in / {response.usage.output_tokens} out"
    )
    print("Открой http://localhost:3000 — там твой первый trace")


if __name__ == "__main__":
    main()
