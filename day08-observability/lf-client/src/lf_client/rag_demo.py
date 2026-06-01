"""Smoke test для GroundedGenerator.

Ключевая проверка дня: на запросе про parallel tool calls система должна
ОТКАЗАТЬСЯ (is_refusal=True). Это success criteria для confidence threshold:
если корпус не содержит ответа, мы не галлюцинируем, а честно отказываемся.
"""

import asyncio
import os

from dotenv import load_dotenv
from langfuse import get_client

from lf_client.client import ClaudeClient
from lf_client.grounded_generator import GroundedGenerator
from lf_client.retrieval import TwoStageRetriever

load_dotenv()


async def main() -> None:
    langfuse = get_client()
    retriever = TwoStageRetriever()
    client = ClaudeClient(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        langfuse=langfuse,
    )

    rag = GroundedGenerator(retriever=retriever, client=client)

    queries = [
        "How does prompt chaining work?",
        "When should I use parallel tool calls?",  # UNANSWERABLE!
        "What is tool_choice and when to use force?",
    ]

    for q in queries:
        print(f"\n{'=' * 60}")
        print(f"QUERY: {q}")
        print(f"{'=' * 60}")

        result = await rag.answer(q)

        if result.is_refusal:
            print(f"REFUSED: {result.refusal_reason}")
            print(f"(top confidence was {result.top_confidence:.3f})")
        else:
            print(f"ANSWER: {result.answer}")
            print(f"\nSources used: {[c.source for c in result.chunks_used]}")
            print(f"Top confidence: {result.top_confidence:.3f}")

    print(f"\n{'=' * 60}")
    print(f"LLM cost: ${client.total_cost_usd:.6f} ({client.call_count} calls)")
    print(f"Retrieval cost: ${retriever.total_cost_usd:.6f}")
    langfuse.flush()


if __name__ == "__main__":
    asyncio.run(main())
