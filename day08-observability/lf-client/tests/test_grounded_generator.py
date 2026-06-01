from unittest.mock import AsyncMock, MagicMock

from lf_client.grounded_generator import (
    REFUSAL_SENTENCE,
    GroundedGenerator,
    _format_context,
)
from lf_client.retrieval import RetrievedChunk


def _chunk(
    text: str = "Prompt chaining decomposes a task into steps.",
    source: str = "basic_workflows.ipynb",
    section_path: str = "Workflows",
    dense_score: float = 0.5,
    rerank_score: float | None = 0.9,
) -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        source=source,
        section_path=section_path,
        dense_score=dense_score,
        rerank_score=rerank_score,
    )


def _make_generator(
    chunks: list[RetrievedChunk],
    answer_text: str = "Prompt chaining runs steps in sequence [1].",
    confidence_threshold: float = 0.4,
) -> tuple[GroundedGenerator, MagicMock, AsyncMock]:
    """GroundedGenerator с замоканными retriever и client.

    retriever.retrieve — обычный sync MagicMock (так его и зовёт код).
    client.ask — AsyncMock, потому что метод асинхронный.
    """
    retriever = MagicMock()
    retriever.retrieve = MagicMock(return_value=chunks)

    client = MagicMock()
    client.ask = AsyncMock(return_value=MagicMock(text=answer_text))

    generator = GroundedGenerator(
        retriever, client, confidence_threshold=confidence_threshold
    )
    return generator, retriever.retrieve, client.ask


# 1. Refusal on empty chunks ------------------------------------------------
async def test_refusal_on_empty_chunks() -> None:
    generator, _, ask = _make_generator([])

    result = await generator.answer("anything")

    assert result.is_refusal is True
    assert result.refusal_reason == "no_chunks_retrieved"
    assert result.chunks_used == []
    assert result.top_confidence == 0.0
    # Пустой retrieval → Claude вообще не зовём.
    ask.assert_not_awaited()


# 2. Refusal on low confidence ---------------------------------------------
async def test_refusal_on_low_confidence() -> None:
    generator, _, ask = _make_generator(
        [_chunk(rerank_score=0.1)], confidence_threshold=0.4
    )

    result = await generator.answer("off-topic question")

    assert result.is_refusal is True
    assert result.refusal_reason is not None
    assert result.refusal_reason.startswith("low_confidence")
    assert result.top_confidence == 0.1
    # Confidence check ДО генерации — экономим токены, не зовём Claude.
    ask.assert_not_awaited()


# 3. Generation runs on high confidence ------------------------------------
async def test_generation_runs_on_high_confidence() -> None:
    generator, _, ask = _make_generator([_chunk(rerank_score=0.9)])

    result = await generator.answer("How does prompt chaining work?")

    assert result.is_refusal is False
    assert result.refusal_reason is None
    assert result.top_confidence == 0.9
    # Высокая уверенность → Claude вызван ровно один раз.
    ask.assert_awaited_once()


# 4. Prompt format ----------------------------------------------------------
def test_context_format_has_numbered_sources() -> None:
    chunks = [
        _chunk(text="first chunk", source="a.ipynb", section_path="A"),
        _chunk(text="second chunk", source="b.ipynb", section_path="B"),
    ]
    formatted = _format_context(chunks)

    assert "[1] (a.ipynb / A):" in formatted
    assert "[2] (b.ipynb / B):" in formatted
    assert "first chunk" in formatted
    assert "second chunk" in formatted


async def test_prompt_passed_to_claude_contains_numbered_context() -> None:
    generator, _, ask = _make_generator(
        [_chunk(text="chaining info", source="wf.ipynb", section_path="Chaining")]
    )

    await generator.answer("How does prompt chaining work?")

    assert ask.await_args is not None
    prompt = ask.await_args.kwargs["prompt"]
    assert "[1] (wf.ipynb / Chaining):" in prompt
    assert "chaining info" in prompt


# 5. Sources tracked correctly ---------------------------------------------
async def test_sources_tracked_match_retriever_output() -> None:
    chunks = [
        _chunk(source="a.ipynb", rerank_score=0.9),
        _chunk(source="b.ipynb", rerank_score=0.8),
        _chunk(source="c.ipynb", rerank_score=0.7),
    ]
    generator, retrieve, _ = _make_generator(chunks)

    result = await generator.answer("How does prompt chaining work?")

    # chunks_used — ровно то, что вернул retriever, в том же порядке.
    assert result.chunks_used == chunks
    assert [c.source for c in result.chunks_used] == [
        "a.ipynb",
        "b.ipynb",
        "c.ipynb",
    ]
    retrieve.assert_called_once()


# bonus: модель сама признаёт незнание → помечаем как refusal -----------------
async def test_model_self_refusal_is_flagged() -> None:
    generator, _, _ = _make_generator(
        [_chunk(rerank_score=0.9)], answer_text=REFUSAL_SENTENCE
    )

    result = await generator.answer("a question the context cannot answer")

    assert result.is_refusal is True
    assert result.refusal_reason == "model_refused_no_answer_in_context"
