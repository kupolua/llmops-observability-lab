from dotenv import load_dotenv

from lf_client.retrieval import TwoStageRetriever

load_dotenv()


QUERIES = [
    "How does prompt chaining work?",
    "When should I use parallel tool calls?",
    "Difference between evaluator-optimizer and orchestrator-workers?",
    "How to validate JSON output from Claude?",
    "What is tool_choice and when to use force?",
]


def main() -> None:
    retriever = TwoStageRetriever()

    for query in QUERIES:
        print("=" * 70)
        print(f"QUERY: {query}")
        print("=" * 70)

        # БЕЗ реранкера
        print("\n--- БЕЗ реранкера (только dense) ---")
        no_rerank = retriever.retrieve(query, top_k=3, use_rerank=False)
        for rank, c in enumerate(no_rerank, 1):
            short_source = c.source.split("/")[-1]
            print(
                f"  [{rank}] dense={c.dense_score:.3f}  {short_source} / {c.section_path[:50]}"
            )

        # С реранкером
        print("\n--- С реранкером (dense top-20 → rerank top-3) ---")
        with_rerank = retriever.retrieve(
            query, top_k=3, top_n_candidates=20, use_rerank=True
        )
        for rank, c in enumerate(with_rerank, 1):
            short_source = c.source.split("/")[-1]
            rerank_str = f"{c.rerank_score:.3f}" if c.rerank_score is not None else "?"
            print(
                f"  [{rank}] rerank={rerank_str}  {short_source} / {c.section_path[:50]}"
            )

        print()

    print(f"\nИтого стоимость: ${retriever.total_cost_usd:.6f}")


if __name__ == "__main__":
    main()
