"""Демо: baseline RAG vs multi-query RAG — до и после.

Цель — визуально показать, что даёт автоматическая декомпозиция запросов.
Один и тот же GroundedGenerator запускаем с двумя retriever'ами:

- baseline: TwoStageRetriever — один retrieval-вызов на запрос;
- multi:    MultiQueryRetriever — compositional-запросы режутся на под-запросы,
            каждый ищется отдельно, результаты мёржатся и реранкаются.

Что ожидаем увидеть:

1. Compositional ("Difference between X and Y?") — baseline тянет чанки в
   основном про один из концептов, multi-query достаёт чанки про ОБА →
   ответ Claude становится полноценным сравнением.
2. Simple ("How does prompt chaining work?") — decomposer вернёт
   is_compositional=False, multi-retriever пройдёт обычным путём. Разницы
   с baseline практически нет.
3. Exact term ("What is tool_choice?") — тоже не compositional. Без изменений.

Это заодно проверка того, что routing (compositional vs simple) работает.
"""

import asyncio
import os

from dotenv import load_dotenv
from langfuse import get_client

from lf_client.client import ClaudeClient
from lf_client.grounded_generator import GroundedAnswer, GroundedGenerator
from lf_client.multi_query_retrieval import MultiQueryRetriever
from lf_client.query_decomposer import QueryDecomposer
from lf_client.retrieval import TwoStageRetriever

load_dotenv()


# Три запроса разных типов — покрывают все ветки routing'а.
QUERIES = [
    # Compositional — тут multi-query должен дать видимый выигрыш.
    "Difference between evaluator-optimizer and orchestrator-workers?",
    # Simple — decomposer вернёт is_compositional=False.
    "How does prompt chaining work?",
    # Exact term — тоже не compositional.
    "What is tool_choice?",
]


def _print_routing(result: GroundedAnswer) -> None:
    """Показать решение routing'а напрямую из MultiQueryResult.

    Видно без догадок по ответу: счёл ли decomposer запрос compositional,
    на какие под-запросы разбил, сколько чанков дал каждый и как сработала
    дедупликация. Для baseline (обычный retriever) meta = None — пропускаем.
    """
    meta = result.retrieval_meta
    if meta is None:
        return
    print(f"routing: is_compositional={meta.is_compositional}")
    if meta.is_compositional:
        print(f"  subqueries: {meta.subqueries}")
        print(f"  per-subquery hits: {meta.per_subquery_counts}")
        print(
            f"  merged: {meta.num_before_dedup} -> {meta.num_after_dedup} after dedup"
        )


def _print_result(label: str, result: GroundedAnswer) -> None:
    """Единый формат вывода для baseline и multi-query."""
    print(f"\n--- {label} ---")
    _print_routing(result)
    if result.is_refusal:
        print(f"[REFUSAL] reason: {result.refusal_reason}")
        print(f"top_confidence: {result.top_confidence:.3f}")
        return
    sources = [c.source for c in result.chunks_used]
    print(f"Sources ({len(sources)}): {sources}")
    print(f"top_confidence: {result.top_confidence:.3f}")
    print(f"Answer (first 200 chars): {result.answer[:200]}...")


async def main() -> None:
    langfuse = get_client()

    # Setup: общий retriever и client, поверх — два RAG-пайплайна.
    retriever = TwoStageRetriever()
    client = ClaudeClient(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        langfuse=langfuse,
    )
    decomposer = QueryDecomposer(client=client)
    multi_retriever = MultiQueryRetriever(retriever, decomposer)

    rag_baseline = GroundedGenerator(retriever=retriever, client=client)
    rag_multi = GroundedGenerator(retriever=multi_retriever, client=client)

    for query in QUERIES:
        print(f"\n{'=' * 70}")
        print(f"QUERY: {query}")
        print(f"{'=' * 70}")

        # Запускаем оба пайплайна по очереди (не параллельно — так вывод
        # читается линейно, и нагляднее сравнивать ответы глазами).
        baseline_result = await rag_baseline.answer(query)
        _print_result("BASELINE (single-query retrieval)", baseline_result)

        multi_result = await rag_multi.answer(query)
        _print_result("MULTI-QUERY", multi_result)

    print(f"\n{'=' * 70}")
    print(f"LLM cost: ${client.total_cost_usd:.6f} ({client.call_count} calls)")
    print(f"Retrieval cost: ${retriever.total_cost_usd:.6f}")
    langfuse.flush()


if __name__ == "__main__":
    asyncio.run(main())
