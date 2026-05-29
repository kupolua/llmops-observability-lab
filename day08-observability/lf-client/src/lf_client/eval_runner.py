"""Запуск всех стратегий retrieval на eval-датасете.

Сравнивает:
1. dense only
2. dense + rerank
3. hybrid (dense + BM25 via RRF)
4. hybrid + rerank
"""

from dotenv import load_dotenv

from lf_client.eval_data import EVAL_DATASET
from lf_client.eval_metrics import (
    QueryEvalResult,
    StrategyEvalSummary,
    evaluate_query,
    summarize_strategy,
)
from lf_client.hybrid_search import HybridResult, HybridSearcher
from lf_client.retrieval import RetrievedChunk, TwoStageRetriever

load_dotenv()


K = 3
TOP_N_CANDIDATES = 20


def hybrid_to_retrieved(results: list[HybridResult]) -> list[RetrievedChunk]:
    """Преобразовать HybridResult в RetrievedChunk (общий интерфейс для метрик)."""
    return [
        RetrievedChunk(
            text=r.text,
            source=r.source,
            section_path=r.section_path,
            dense_score=0.0,  # для hybrid это не применимо
            rerank_score=None,
        )
        for r in results
    ]


def evaluate_strategy(
    strategy_name: str,
    chunks_per_query: dict[str, list[RetrievedChunk]],
) -> StrategyEvalSummary:
    """Прогнать метрики для одной стратегии."""
    per_query: list[QueryEvalResult] = []
    for eval_q in EVAL_DATASET:
        chunks = chunks_per_query[eval_q.query]
        result = evaluate_query(eval_q, chunks, k=K)
        per_query.append(result)
    return summarize_strategy(strategy_name, per_query, k=K)


def main() -> None:
    print(f"\n{'=' * 70}")
    print(f"RAG EVALUATION (k={K}, candidates={TOP_N_CANDIDATES})")
    print(f"{'=' * 70}\n")

    # Инициализируем retrievers
    two_stage = TwoStageRetriever()
    hybrid = HybridSearcher()

    # Прогоняем все запросы по всем стратегиям
    strategies: dict[str, dict[str, list[RetrievedChunk]]] = {
        "1. dense only": {},
        "2. dense + rerank": {},
        "3. hybrid (RRF)": {},
        "4. hybrid + rerank": {},
    }

    for eval_q in EVAL_DATASET:
        q = eval_q.query
        print(f"Querying: {q[:60]}...")

        # Strategy 1: dense only
        strategies["1. dense only"][q] = two_stage.retrieve(
            q, top_k=K, use_rerank=False
        )

        # Strategy 2: dense + rerank
        strategies["2. dense + rerank"][q] = two_stage.retrieve(
            q, top_k=K, top_n_candidates=TOP_N_CANDIDATES, use_rerank=True
        )

        # Strategy 3: hybrid via RRF, no rerank
        strategies["3. hybrid (RRF)"][q] = hybrid_to_retrieved(
            hybrid.search(
                q, top_k=K, top_n_per_method=TOP_N_CANDIDATES, use_rerank=False
            )
        )

        # Strategy 4: hybrid + rerank
        strategies["4. hybrid + rerank"][q] = hybrid_to_retrieved(
            hybrid.search(
                q, top_k=K, top_n_per_method=TOP_N_CANDIDATES, use_rerank=True
            )
        )

    # Считаем метрики по каждой стратегии
    summaries: list[StrategyEvalSummary] = []
    for name, chunks_per_query in strategies.items():
        summary = evaluate_strategy(name, chunks_per_query)
        summaries.append(summary)

    # Сводная таблица
    print(f"\n{'=' * 70}")
    print("SUMMARY (averages exclude unanswerable queries)")
    print(f"{'=' * 70}")
    print(f"{'Strategy':<22} {'Recall@3':>10} {'Precision@3':>13} {'MRR':>8}")
    print(f"{'-' * 70}")
    for s in summaries:
        print(
            f"{s.strategy_name:<22} "
            f"{s.mean_recall:>10.3f} "
            f"{s.mean_precision:>13.3f} "
            f"{s.mrr:>8.3f}"
        )

    # Детальный pos-by-query анализ
    print(f"\n{'=' * 70}")
    print("PER-QUERY BREAKDOWN")
    print(f"{'=' * 70}")

    for i, eval_q in enumerate(EVAL_DATASET):
        print(f"\n--- Q{i + 1}: {eval_q.query} ---")
        if eval_q.is_unanswerable:
            print(f"  [UNANSWERABLE — note: {eval_q.note}]")
            continue

        print(f"  ground truth: {len(eval_q.relevant)} relevant chunk(s)")
        for s in summaries:
            r = s.per_query[i]
            rank_str = (
                f"rank={r.first_relevant_rank}"
                if r.first_relevant_rank
                else "NOT FOUND"
            )
            print(
                f"  {s.strategy_name:<22} "
                f"recall={r.recall_at_k:.2f}  "
                f"prec={r.precision_at_k:.2f}  "
                f"{rank_str}"
            )


if __name__ == "__main__":
    main()
