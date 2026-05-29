"""Метрики для оценки качества retrieval.

recall@k:    нашли ли всех релевантных в top-k
precision@k: какая доля top-k релевантна
MRR:         средний обратный ранг первого релевантного
"""

from dataclasses import dataclass

from lf_client.eval_data import EvalQuery, GroundTruthRelevant
from lf_client.retrieval import RetrievedChunk


def is_chunk_relevant(
    chunk: RetrievedChunk,
    relevant_markers: list[GroundTruthRelevant],
) -> bool:
    """Проверить, релевантен ли чанк хотя бы одному ground truth маркеру.

    Логика: prefix-match. Источник содержит file_suffix, секция начинается
    с section_prefix.
    """
    for marker in relevant_markers:
        file_match = marker.file_suffix in chunk.source
        section_match = chunk.section_path.startswith(marker.section_prefix)
        if file_match and section_match:
            return True
    return False


@dataclass
class QueryEvalResult:
    """Результат оценки одного запроса."""

    query: str
    is_unanswerable: bool
    num_relevant_in_top_k: int
    num_relevant_total: int
    first_relevant_rank: int | None  # None если не найдено
    recall_at_k: float
    precision_at_k: float
    reciprocal_rank: float


def evaluate_query(
    eval_query: EvalQuery,
    retrieved_chunks: list[RetrievedChunk],
    k: int = 3,
) -> QueryEvalResult:
    """Посчитать метрики для одного запроса."""
    top_k = retrieved_chunks[:k]

    if eval_query.is_unanswerable:
        return QueryEvalResult(
            query=eval_query.query,
            is_unanswerable=True,
            num_relevant_in_top_k=0,
            num_relevant_total=0,
            first_relevant_rank=None,
            recall_at_k=0.0,
            precision_at_k=0.0,
            reciprocal_rank=0.0,
        )

    # RECALL: сколько ground truth маркеров покрыто (хоть один чанк нашёлся)
    markers_found = sum(
        1
        for marker in eval_query.relevant
        if any(_chunk_matches_marker(chunk, marker) for chunk in top_k)
    )

    # PRECISION: сколько чанков в top-k релевантны (хоть одному маркеру)
    chunks_relevant = sum(
        1 for chunk in top_k if is_chunk_relevant(chunk, eval_query.relevant)
    )

    # MRR: ранг первого релевантного чанка
    first_rank: int | None = None
    for rank, chunk in enumerate(top_k, start=1):
        if is_chunk_relevant(chunk, eval_query.relevant):
            first_rank = rank
            break

    num_relevant_total = len(eval_query.relevant)
    recall = markers_found / num_relevant_total if num_relevant_total > 0 else 0.0
    precision = chunks_relevant / k if k > 0 else 0.0
    rr = 1.0 / first_rank if first_rank else 0.0

    return QueryEvalResult(
        query=eval_query.query,
        is_unanswerable=False,
        num_relevant_in_top_k=chunks_relevant,
        num_relevant_total=num_relevant_total,
        first_relevant_rank=first_rank,
        recall_at_k=recall,
        precision_at_k=precision,
        reciprocal_rank=rr,
    )


def _chunk_matches_marker(
    chunk: RetrievedChunk,
    marker: GroundTruthRelevant,
) -> bool:
    """Internal helper: чанк матчит конкретный маркер."""
    file_match = marker.file_suffix in chunk.source
    section_match = chunk.section_path.startswith(marker.section_prefix)
    return file_match and section_match


@dataclass
class StrategyEvalSummary:
    """Сводка по одной стратегии retrieval'а."""

    strategy_name: str
    k: int
    mean_recall: float
    mean_precision: float
    mrr: float
    per_query: list[QueryEvalResult]


def summarize_strategy(
    strategy_name: str,
    per_query_results: list[QueryEvalResult],
    k: int,
) -> StrategyEvalSummary:
    """Усреднить метрики по всем запросам (исключая unanswerable)."""
    answerable = [r for r in per_query_results if not r.is_unanswerable]

    if not answerable:
        return StrategyEvalSummary(
            strategy_name=strategy_name,
            k=k,
            mean_recall=0.0,
            mean_precision=0.0,
            mrr=0.0,
            per_query=per_query_results,
        )

    return StrategyEvalSummary(
        strategy_name=strategy_name,
        k=k,
        mean_recall=sum(r.recall_at_k for r in answerable) / len(answerable),
        mean_precision=sum(r.precision_at_k for r in answerable) / len(answerable),
        mrr=sum(r.reciprocal_rank for r in answerable) / len(answerable),
        per_query=per_query_results,
    )
