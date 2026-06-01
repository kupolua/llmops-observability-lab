"""Декомпозиция compositional-запросов в под-запросы через LLM.

Зачем: запрос вроде «В чём разница между X и Y?» плохо ищется одним
retrieval-вызовом — релевантные чанки про X и про Y лежат в разных местах
корпуса. Разбиваем такой запрос на ["What is X?", "What is Y?"] и ищем
каждый по отдельности (multi-query retrieval).

Решение «компонуемый ли запрос и как его разбить» принимает Claude через
принудительный tool_use — это даёт структурированный JSON вместо парсинга
свободного текста.
"""

from dataclasses import dataclass

from anthropic.types import ToolParam

from lf_client.client import ClaudeClient

# Системный промпт учит Claude отличать compositional-запросы от простых.
SYSTEM_PROMPT = """You are a query analyzer. Determine if a query requires \
multiple retrievals.

A query is COMPOSITIONAL if it asks to compare, contrast, or combine \
information about MULTIPLE distinct concepts/entities.

Examples:
- "Difference between X and Y?" -> compositional -> ["What is X?", "What is Y?"]
- "How does X work?" -> not compositional -> []
- "Compare X, Y, and Z" -> compositional -> ["What is X?", "What is Y?", "What is Z?"]
- "What is X used for?" -> not compositional -> []

Return your decision via the decompose_query tool."""

# Схема инструмента: tool_choice заставит Claude вернуть ровно этот JSON.
DECOMPOSE_TOOL: ToolParam = {
    "name": "decompose_query",
    "description": "Report whether the query is compositional and split it.",
    "input_schema": {
        "type": "object",
        "properties": {
            "is_compositional": {"type": "boolean"},
            "subqueries": {
                "type": "array",
                "items": {"type": "string"},
                "description": "2-4 sub-queries if compositional, empty otherwise",
            },
            "reasoning": {
                "type": "string",
                "description": "brief explanation",
            },
        },
        "required": ["is_compositional", "subqueries", "reasoning"],
    },
}


@dataclass
class DecomposedQuery:
    """Результат декомпозиции."""

    original: str
    subqueries: list[str]  # пустой если декомпозиция не нужна
    is_compositional: bool
    reasoning: str  # пояснение почему именно так


class QueryDecomposer:
    """Декомпозирует compositional queries в под-запросы через LLM."""

    def __init__(self, client: ClaudeClient) -> None:
        self._client = client

    async def decompose(self, query: str) -> DecomposedQuery:
        """Решает, нужна ли декомпозиция, и делает её.

        Возвращает DecomposedQuery. Если запрос простой — subqueries пустой,
        is_compositional=False. Вызывающий код сам решает: искать по
        original или по subqueries.
        """
        result = await self._client.ask_tool(
            prompt=query,
            tools=[DECOMPOSE_TOOL],
            tool_name="decompose_query",
            system=SYSTEM_PROMPT,
            trace_name="query-decompose",
            tags=["rag", "decomposition"],
        )

        data = result.tool_input
        is_compositional = bool(data.get("is_compositional", False))
        raw_subqueries = data.get("subqueries", []) or []
        reasoning = str(data.get("reasoning", ""))

        # Защита: если модель пометила запрос как простой, но прислала
        # под-запросы (или наоборот) — приводим к согласованному виду.
        subqueries = [str(s) for s in raw_subqueries] if is_compositional else []

        return DecomposedQuery(
            original=query,
            subqueries=subqueries,
            is_compositional=is_compositional,
            reasoning=reasoning,
        )


async def main() -> None:
    import os

    from dotenv import load_dotenv
    from langfuse import get_client

    load_dotenv()

    langfuse = get_client()
    client = ClaudeClient(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        langfuse=langfuse,
    )
    decomposer = QueryDecomposer(client)

    queries = [
        "Difference between evaluator-optimizer and orchestrator-workers?",
        "How does prompt chaining work?",
        "Compare prompt chaining, routing, and parallelization.",
    ]

    for query in queries:
        print(f"\n{'=' * 70}")
        print(f"QUERY: {query}")
        print(f"{'-' * 70}")
        result = await decomposer.decompose(query)
        print(f"compositional: {result.is_compositional}")
        print(f"reasoning:     {result.reasoning}")
        if result.subqueries:
            print("subqueries:")
            for i, sq in enumerate(result.subqueries, start=1):
                print(f"  {i}. {sq}")

    print(f"\n{'=' * 70}")
    print(f"LLM cost: ${client.total_cost_usd:.6f} ({client.call_count} calls)")
    langfuse.flush()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
