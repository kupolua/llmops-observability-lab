import pytest

from lf_client.eval_data import EvalQuery, GroundTruthRelevant
from lf_client.eval_metrics import evaluate_query, is_chunk_relevant, summarize_strategy
from lf_client.retrieval import RetrievedChunk


def make_chunk(source: str, section_path: str) -> RetrievedChunk:
    return RetrievedChunk(
        text="dummy",
        source=source,
        section_path=section_path,
        dense_score=0.5,
        rerank_score=None,
    )


# ---------- is_chunk_relevant ----------


def test_is_relevant_exact_match() -> None:
    chunk = make_chunk("basic_workflows.ipynb", "Basic Multi-LLM Workflows")
    relevant = [
        GroundTruthRelevant(
            file_suffix="basic_workflows.ipynb",
            section_prefix="Basic Multi-LLM Workflows",
        )
    ]
    assert is_chunk_relevant(chunk, relevant)


def test_is_relevant_prefix_match() -> None:
    """Чанк с более глубокой секцией должен матчить родительский префикс."""
    chunk = make_chunk(
        "orchestrator_workers.ipynb",
        "Orchestrator-Workers Workflow > Introduction > When to use",
    )
    relevant = [
        GroundTruthRelevant(
            file_suffix="orchestrator_workers.ipynb",
            section_prefix="Orchestrator-Workers Workflow",
        )
    ]
    assert is_chunk_relevant(chunk, relevant)


def test_is_relevant_wrong_file() -> None:
    chunk = make_chunk("wrong_file.ipynb", "Basic Multi-LLM Workflows")
    relevant = [
        GroundTruthRelevant(
            file_suffix="basic_workflows.ipynb",
            section_prefix="Basic Multi-LLM Workflows",
        )
    ]
    assert not is_chunk_relevant(chunk, relevant)


def test_is_relevant_wrong_section() -> None:
    chunk = make_chunk("basic_workflows.ipynb", "Something Different")
    relevant = [
        GroundTruthRelevant(
            file_suffix="basic_workflows.ipynb",
            section_prefix="Basic Multi-LLM Workflows",
        )
    ]
    assert not is_chunk_relevant(chunk, relevant)


def test_is_relevant_multiple_markers_any_match() -> None:
    """Если хотя бы один маркер совпадает — релевантно."""
    chunk = make_chunk("file_b.ipynb", "Section B")
    relevant = [
        GroundTruthRelevant(file_suffix="file_a.ipynb", section_prefix="Section A"),
        GroundTruthRelevant(file_suffix="file_b.ipynb", section_prefix="Section B"),
    ]
    assert is_chunk_relevant(chunk, relevant)


# ---------- evaluate_query ----------


def test_evaluate_perfect_recall() -> None:
    """Все релевантные в top-k."""
    eval_q = EvalQuery(
        query="test",
        relevant=[
            GroundTruthRelevant(file_suffix="a.ipynb", section_prefix="X"),
            GroundTruthRelevant(file_suffix="b.ipynb", section_prefix="Y"),
        ],
    )
    chunks = [
        make_chunk("a.ipynb", "X"),
        make_chunk("b.ipynb", "Y"),
        make_chunk("c.ipynb", "Z"),  # нерелевантный
    ]
    result = evaluate_query(eval_q, chunks, k=3)
    assert result.recall_at_k == 1.0
    assert result.precision_at_k == pytest.approx(2 / 3)
    assert result.reciprocal_rank == 1.0  # первый релевантный на ранге 1
    assert result.first_relevant_rank == 1


def test_evaluate_no_relevant_found() -> None:
    eval_q = EvalQuery(
        query="test",
        relevant=[GroundTruthRelevant(file_suffix="a.ipynb", section_prefix="X")],
    )
    chunks = [
        make_chunk("wrong.ipynb", "Y"),
        make_chunk("wrong.ipynb", "Z"),
    ]
    result = evaluate_query(eval_q, chunks, k=3)
    assert result.recall_at_k == 0.0
    assert result.precision_at_k == 0.0
    assert result.reciprocal_rank == 0.0
    assert result.first_relevant_rank is None


def test_evaluate_rank_2() -> None:
    """Релевантный на 2-м месте — RR=0.5."""
    eval_q = EvalQuery(
        query="test",
        relevant=[GroundTruthRelevant(file_suffix="a.ipynb", section_prefix="X")],
    )
    chunks = [
        make_chunk("wrong.ipynb", "Y"),
        make_chunk("a.ipynb", "X"),
        make_chunk("other.ipynb", "Z"),
    ]
    result = evaluate_query(eval_q, chunks, k=3)
    assert result.reciprocal_rank == 0.5
    assert result.first_relevant_rank == 2


def test_evaluate_unanswerable_returns_zeros() -> None:
    eval_q = EvalQuery(query="test", is_unanswerable=True)
    chunks = [make_chunk("anything.ipynb", "anywhere")]
    result = evaluate_query(eval_q, chunks, k=3)
    assert result.is_unanswerable
    assert result.recall_at_k == 0.0


def test_summarize_excludes_unanswerable() -> None:
    """Unanswerable не должны влиять на средние."""
    results = [
        # answerable, recall=1.0
        evaluate_query(
            EvalQuery(
                query="q1",
                relevant=[
                    GroundTruthRelevant(file_suffix="a.ipynb", section_prefix="X")
                ],
            ),
            [make_chunk("a.ipynb", "X")],
            k=3,
        ),
        # unanswerable
        evaluate_query(
            EvalQuery(query="q2", is_unanswerable=True),
            [make_chunk("anything.ipynb", "anywhere")],
            k=3,
        ),
    ]
    summary = summarize_strategy("test", results, k=3)
    # Среднее считается только по answerable (q1) → recall=1.0
    assert summary.mean_recall == 1.0
