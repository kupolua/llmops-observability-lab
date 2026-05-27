import asyncio
import os
from dotenv import load_dotenv
from langfuse import Langfuse

from lf_client.client import ClaudeClient

load_dotenv()


async def main() -> None:
    langfuse = Langfuse(
        public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
        secret_key=os.environ["LANGFUSE_SECRET_KEY"],
        host=os.environ["LANGFUSE_HOST"],
    )

    client = ClaudeClient(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        langfuse=langfuse,
    )

    # Делаем 5 разных вызовов
    questions = [
        "Что такое HNSW?",
        "Объясни косинусное сходство в одном предложении.",
        "Какие популярные vector databases?",
        "В чём разница между RAG и fine-tuning?",
        "Что такое prompt caching?",
    ]

    for q in questions:
        result = await client.ask(q, trace_name=f"qa-{q[:30]}")
        print(f"\n=== {q} ===\n{result.text[:200]}...")
        print(f"  cost: ${result.cost.total_cost_usd}")

    print(f"\n--- Итого: {client.call_count} вызовов, ${client.total_cost_usd} ---")

    # Важно: flush перед выходом
    langfuse.flush()


if __name__ == "__main__":
    asyncio.run(main())
