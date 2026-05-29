from dotenv import load_dotenv
from qdrant_client import QdrantClient

from lf_client.embedding_service import EmbeddingService

load_dotenv()


def main() -> None:
    embedder = EmbeddingService()
    client = QdrantClient(url="http://localhost:6333")

    collection_name = "construction_test"

    queries = [
        "как подготовить стену перед штукатуркой?",
        "чем обрабатывать стыки гипсокартона?",
        "при какой температуре можно красить?",
        "как избежать трещин в полу?",
    ]

    # Эмбеддим все запросы батчем
    query_vectors = embedder.embed(queries, input_type="query")

    for query, qvec in zip(queries, query_vectors, strict=True):
        # Поиск в Qdrant
        results = client.query_points(
            collection_name=collection_name,
            query=qvec.tolist(),
            limit=3,
        ).points

        print(f"\n=== Запрос: {query} ===")
        for rank, point in enumerate(results, 1):
            score = point.score
            text = point.payload["text"] if point.payload else "?"
            print(f"  {rank}. [{score:.3f}] {text}")


if __name__ == "__main__":
    main()
