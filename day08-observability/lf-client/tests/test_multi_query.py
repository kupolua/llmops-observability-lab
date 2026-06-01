from unittest.mock import AsyncMock, MagicMock

from lf_client.multi_query_retrieval import MultiQueryRetriever, _chunk_key
from lf_client.query_decomposer import DecomposedQuery, QueryDecomposer
from lf_client.retrieval import RetrievedChunk


def _chunk(
    text: str = "some text",
    source: str = "a.ipynb",
    section_path: str = "Section",
) -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        source=source,
        section_path=section_path,
        dense_score=0.5,
        rerank_score=0.9,
    )


def _decomposer_with_tool_input(tool_input: dict[str, object]) -> QueryDecomposer:
    """QueryDecomposer поверх замоканного client.ask_tool.

    decompose() читает только result.tool_input, поэтому ask_tool —
    AsyncMock, возвращающий объект с нужным tool_input.
    """
    client = MagicMock()
    client.ask_tool = AsyncMock(return_value=MagicMock(tool_input=tool_input))
    return QueryDecomposer(client)


def _make_multi(
    decomposed: DecomposedQuery,
    retrieve_returns: list[list[RetrievedChunk]],
    final_top_k: int = 5,
) -> tuple[MultiQueryRetriever, MagicMock, MagicMock]:
    """MultiQueryRetriever с замоканными decomposer и retriever.

    decomposer.decompose — AsyncMock (async).
    retriever.retrieve / rerank_chunks — sync MagicMock (так их зовёт код
    через asyncio.to_thread). retrieve_returns — список результатов по одному
    на каждый ожидаемый вызов retrieve (side_effect). rerank_chunks по
    умолчанию identity, чтобы в финальном списке было видно ровно то, что
    осталось после дедупа.
    """
    decomposer = MagicMock()
    decomposer.decompose = AsyncMock(return_value=decomposed)

    retriever = MagicMock()
    retriever.retrieve = MagicMock(side_effect=retrieve_returns)
    retriever.rerank_chunks = MagicMock(
        side_effect=lambda query, chunks, top_k: chunks[:top_k]
    )

    multi = MultiQueryRetriever(
        retriever, decomposer, top_k_per_subquery=3, final_top_k=final_top_k
    )
    return multi, retriever, decomposer.decompose


# --- QueryDecomposer ------------------------------------------------------
async def test_decomposer_simple_query() -> None:
    decomposer = _decomposer_with_tool_input(
        {
            "is_compositional": False,
            "subqueries": [],
            "reasoning": "single concept",
        }
    )

    result = await decomposer.decompose("How does prompt chaining work?")

    assert result.is_compositional is False
    assert result.subqueries == []
    assert result.original == "How does prompt chaining work?"


async def test_decomposer_compositional_query() -> None:
    decomposer = _decomposer_with_tool_input(
        {
            "is_compositional": True,
            "subqueries": ["What is X?", "What is Y?"],
            "reasoning": "compares two concepts",
        }
    )

    result = await decomposer.decompose("Difference between X and Y?")

    assert result.is_compositional is True
    assert len(result.subqueries) >= 2


async def test_decomposer_strips_subqueries_when_not_compositional() -> None:
    """Guard: модель сказала simple, но прислала subqueries — игнорируем их."""
    decomposer = _decomposer_with_tool_input(
        {
            "is_compositional": False,
            "subqueries": ["leftover"],
            "reasoning": "inconsistent",
        }
    )

    result = await decomposer.decompose("How does X work?")

    assert result.is_compositional is False
    assert result.subqueries == []


# --- MultiQueryRetriever --------------------------------------------------
async def test_multi_retriever_passthrough_simple() -> None:
    decomposed = DecomposedQuery(
        original="How does X work?",
        subqueries=[],
        is_compositional=False,
        reasoning="simple",
    )
    chunks = [_chunk(source="a.ipynb"), _chunk(source="b.ipynb")]
    multi, retriever, _ = _make_multi(decomposed, [chunks])

    result = await multi.retrieve("How does X work?")

    # Simple query → ровно один retrieve, без декомпозиции и без re-rank.
    retriever.retrieve.assert_called_once()
    retriever.rerank_chunks.assert_not_called()
    assert result.is_compositional is False
    assert result.subqueries == []
    assert result.chunks == chunks


async def test_multi_retriever_compositional() -> None:
    decomposed = DecomposedQuery(
        original="Compare X, Y and Z",
        subqueries=["What is X?", "What is Y?", "What is Z?"],
        is_compositional=True,
        reasoning="three concepts",
    )
    # Каждый sub-query возвращает свой непересекающийся чанк.
    side_effect = [
        [_chunk(source="x.ipynb")],
        [_chunk(source="y.ipynb")],
        [_chunk(source="z.ipynb")],
    ]
    multi, retriever, _ = _make_multi(decomposed, side_effect)

    result = await multi.retrieve("Compare X, Y and Z")

    # retrieve вызван по разу на каждый sub-query.
    assert retriever.retrieve.call_count == 3
    retriever.rerank_chunks.assert_called_once()
    assert result.is_compositional is True
    assert result.per_subquery_counts == {
        "What is X?": 1,
        "What is Y?": 1,
        "What is Z?": 1,
    }
    assert result.num_before_dedup == 3
    assert result.num_after_dedup == 3


async def test_multi_retriever_dedup() -> None:
    decomposed = DecomposedQuery(
        original="Difference between X and Y?",
        subqueries=["What is X?", "What is Y?"],
        is_compositional=True,
        reasoning="two concepts",
    )
    shared = _chunk(source="shared.ipynb", section_path="Common")
    shared_dup = _chunk(source="shared.ipynb", section_path="Common")  # тот же ключ
    only_x = _chunk(source="x.ipynb", section_path="X")
    only_y = _chunk(source="y.ipynb", section_path="Y")
    side_effect = [
        [only_x, shared],  # sub-query 1
        [shared_dup, only_y],  # sub-query 2 — shared пересекается
    ]
    multi, retriever, _ = _make_multi(decomposed, side_effect)

    result = await multi.retrieve("Difference between X and Y?")

    # До дедупа 4 чанка, после — 3 (shared схлопнулся).
    assert result.num_before_dedup == 4
    assert result.num_after_dedup == 3

    keys = [_chunk_key(c) for c in result.chunks]
    assert len(keys) == len(set(keys))  # дублей нет
    assert ("shared.ipynb", "Common") in keys
