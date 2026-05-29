from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from lf_client.embedding_service import EmbeddingService

load_dotenv()


def search(
    query: str,
    top_k: int = 3,
    category_filter: str | None = None,
) -> None:
    embedder = EmbeddingService()
    client = QdrantClient(url="http://localhost:6333")

    qvec = embedder.embed([query], input_type="query")[0]

    query_filter = None
    if category_filter:
        query_filter = Filter(
            must=[
                FieldCondition(key="category", match=MatchValue(value=category_filter))
            ]
        )

    results = client.query_points(
        collection_name="cookbook",
        query=qvec.tolist(),
        limit=top_k,
        query_filter=query_filter,
    ).points

    filter_label = f" (filter: category={category_filter})" if category_filter else ""
    print(f"\n=== {query}{filter_label} ===")
    for rank, point in enumerate(results, 1):
        if point.payload is None:
            continue
        score = point.score
        source = point.payload.get("source", "?")
        section = point.payload.get("section_path", "?")
        text = point.payload.get("text", "")
        print(f"\n  [{rank}] score={score:.3f}")
        print(f"  source: {source}")
        print(f"  section: {section}")
        print(f"  text: {text[:250]}...")


def main() -> None:
    # Реальные доменные вопросы
    queries = [
        "How does prompt chaining work?",
        "When should I use parallel tool calls?",
        "Difference between evaluator-optimizer and orchestrator-workers?",
        "How to validate JSON output from Claude?",
        "What is tool_choice and when to use force?",
    ]

    for q in queries:
        search(q, top_k=3)

    # Демонстрация фильтра — поиск ТОЛЬКО по tool_use
    print("\n\n" + "=" * 60)
    print("ПОИСК С ФИЛЬТРОМ — только в tool_use/")
    print("=" * 60)
    search("Claude tools", top_k=3, category_filter="tool_use")


if __name__ == "__main__":
    main()
