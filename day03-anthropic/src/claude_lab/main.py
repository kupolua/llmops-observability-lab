import asyncio
import os
from dotenv import load_dotenv
from claude_lab.client import ClaudeClient


async def main() -> None:
    load_dotenv()
    client = ClaudeClient(api_key=os.environ["ANTHROPIC_API_KEY"])

    result1 = await client.ask("Скажи 'привет' на трёх языках")
    print(result1.text)
    print(f"  cost: ${result1.cost.total_cost_usd}")

    result2 = await client.ask(
        "Что такое RAG?",
        system="Отвечай очень кратко, в одном предложении.",
    )
    print(result2.text)
    print(f"  cost: ${result2.cost.total_cost_usd}")

    print(f"\nИтого: {client.call_count} вызовов, ${client.total_cost_usd}")


if __name__ == "__main__":
    asyncio.run(main())
