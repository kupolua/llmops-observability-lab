"""Multi-query retrieval с автоматической декомпозицией.

Идея: compositional-запрос («В чём разница между X и Y?») плохо ищется одним
retrieval-вызовом — чанки про X и про Y лежат в разных частях корпуса.
Поэтому:

1. QueryDecomposer решает, компонуемый ли запрос, и режет его на под-запросы.
2. Каждый под-запрос ищем отдельно (top_k_per_subquery чанков).
3. Дедупим (один чанк мог прийти от нескольких под-запросов).
4. Финальный rerank по ОРИГИНАЛЬНому запросу — поднимает наверх то, что
   релевантно вопросу в целом.
5. Возвращаем top-N + метаданные для трейса.

Простые (не compositional) запросы идут обычным путём — один retrieve.
"""

import asyncio
from dataclasses import dataclass, field

from dotenv import load_dotenv

from lf_client.query_decomposer import QueryDecomposer
from lf_client.retrieval import RetrievedChunk, TwoStageRetriever

load_dotenv()


@dataclass
class MultiQueryResult:
    """Результат multi-query retrieval + метаданные для отладки/трейса."""

    original: str
    is_compositional: bool
    subqueries: list[str]
    chunks: list[RetrievedChunk]  # финальный список после dedup + rerank
    # сколько чанков пришло от каждого под-запроса (до дедупликации)
    per_subquery_counts: dict[str, int] = field(default_factory=dict)
    num_before_dedup: int = 0
    num_after_dedup: int = 0


def _chunk_key(chunk: RetrievedChunk) -> tuple[str, str]:
    """Ключ дедупликации: один и тот же чанк = (source, section_path)."""
    return (chunk.source, chunk.section_path)


class MultiQueryRetriever:
    """Retrieval с автоматической декомпозицией compositional queries."""

    def __init__(
        self,
        retriever: TwoStageRetriever,
        decomposer: QueryDecomposer,
        top_k_per_subquery: int = 3,
        final_top_k: int = 5,
    ) -> None:
        self._retriever = retriever
        self._decomposer = decomposer
        self._top_k_per_subquery = top_k_per_subquery
        self._final_top_k = final_top_k

    async def retrieve(self, query: str) -> MultiQueryResult:
        """Декомпозирует → retrieves каждый sub-query → merges → top-N."""
        # 1. Decompose
        decomposed = await self._decomposer.decompose(query)

        # Простой запрос — обычный retrieval, без лишних вызовов.
        if not decomposed.is_compositional or not decomposed.subqueries:
            chunks = await asyncio.to_thread(
                self._retriever.retrieve, query, top_k=self._final_top_k
            )
            return MultiQueryResult(
                original=query,
                is_compositional=False,
                subqueries=[],
                chunks=chunks,
                per_subquery_counts={},
                num_before_dedup=len(chunks),
                num_after_dedup=len(chunks),
            )

        # 2. Retrieve каждый sub-query.
        all_chunks: list[RetrievedChunk] = []
        per_subquery_counts: dict[str, int] = {}
        for subq in decomposed.subqueries:
            chunks = await asyncio.to_thread(
                self._retriever.retrieve, subq, top_k=self._top_k_per_subquery
            )
            per_subquery_counts[subq] = len(chunks)
            all_chunks.extend(chunks)

        num_before_dedup = len(all_chunks)

        # 3. Дедупликация по (source, section_path) с сохранением порядка.
        seen: set[tuple[str, str]] = set()
        deduped: list[RetrievedChunk] = []
        for chunk in all_chunks:
            key = _chunk_key(chunk)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(chunk)

        # 4. Финальный rerank по ОРИГИНАЛЬНому запросу + 5. top-N.
        final_chunks = await asyncio.to_thread(
            self._retriever.rerank_chunks,
            query,
            deduped,
            self._final_top_k,
        )

        return MultiQueryResult(
            original=query,
            is_compositional=True,
            subqueries=decomposed.subqueries,
            chunks=final_chunks,
            per_subquery_counts=per_subquery_counts,
            num_before_dedup=num_before_dedup,
            num_after_dedup=len(deduped),
        )


async def main() -> None:
    import os

    from langfuse import get_client

    from lf_client.client import ClaudeClient

    langfuse = get_client()
    retriever = TwoStageRetriever()
    client = ClaudeClient(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        langfuse=langfuse,
    )
    decomposer = QueryDecomposer(client)
    multi = MultiQueryRetriever(retriever, decomposer)

    queries = [
        "Difference between evaluator-optimizer and orchestrator-workers?",
        "How does prompt chaining work?",
    ]

    for query in queries:
        print(f"\n{'=' * 70}")
        print(f"QUERY: {query}")
        print(f"{'-' * 70}")
        result = await multi.retrieve(query)

        print(f"compositional: {result.is_compositional}")
        if result.is_compositional:
            print(f"subqueries: {result.subqueries}")
            print(f"per-subquery hits: {result.per_subquery_counts}")
            print(
                f"merged: {result.num_before_dedup} -> "
                f"{result.num_after_dedup} after dedup"
            )
        print(f"final top-{len(result.chunks)}:")
        for i, chunk in enumerate(result.chunks, start=1):
            score = chunk.rerank_score if chunk.rerank_score is not None else 0.0
            print(f"  [{i}] rerank={score:.3f}  {chunk.source} / {chunk.section_path}")

    print(f"\n{'=' * 70}")
    print(f"LLM cost: ${client.total_cost_usd:.6f} ({client.call_count} calls)")
    print(f"Retrieval cost: ${retriever.total_cost_usd:.6f}")
    langfuse.flush()


if __name__ == "__main__":
    asyncio.run(main())
