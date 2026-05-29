import uuid

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from lf_client.embedding_service import EmbeddingService

load_dotenv()


CORPUS_WITH_CATEGORIES = [
    ("Штукатурка наносится в три слоя: обрызг, грунт, накрывка.", "штукатурка"),
    ("Маяки устанавливаются по уровню перед оштукатуриванием.", "штукатурка"),
    ("Наливной пол заливается по маякам и прокатывается игольчатым валиком.", "полы"),
    ("Стяжка пола армируется сеткой при толщине более 50мм.", "полы"),
    ("Гипсокартон монтируется на каркас из профилей CD и UD.", "гипсокартон"),
    ("Швы ГКЛ проклеиваются лентой и шпаклюются.", "гипсокартон"),
]


def main() -> None:
    embedder = EmbeddingService()
    client = QdrantClient(url="http://localhost:6333")
    collection_name = "construction_categorized"

    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )

    texts = [item[0] for item in CORPUS_WITH_CATEGORIES]
    categories = [item[1] for item in CORPUS_WITH_CATEGORIES]
    vectors = embedder.embed(texts, input_type="document")

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vec.tolist(),
            payload={"text": txt, "category": cat},
        )
        for txt, cat, vec in zip(texts, categories, vectors, strict=True)
    ]
    client.upsert(collection_name=collection_name, points=points)
    print(f"Загружено {len(points)} точек\n")

    query = "как наносить штукатурку?"
    qvec = embedder.embed([query], input_type="query")[0]

    # Поиск БЕЗ фильтра
    print(f"=== '{query}' — БЕЗ фильтра ===")
    results = client.query_points(
        collection_name=collection_name,
        query=qvec.tolist(),
        limit=4,
    ).points
    for r in results:
        cat = r.payload["category"] if r.payload else "?"
        txt = r.payload["text"] if r.payload else "?"
        print(f"  [{r.score:.3f}] ({cat}) {txt}")

    # Поиск С фильтром — только категория "штукатурка"
    print(f"\n=== '{query}' — ТОЛЬКО category=штукатурка ===")
    results_filtered = client.query_points(
        collection_name=collection_name,
        query=qvec.tolist(),
        limit=4,
        query_filter=Filter(
            must=[FieldCondition(key="category", match=MatchValue(value="штукатурка"))]
        ),
    ).points
    for r in results_filtered:
        cat = r.payload["category"] if r.payload else "?"
        txt = r.payload["text"] if r.payload else "?"
        print(f"  [{r.score:.3f}] ({cat}) {txt}")


if __name__ == "__main__":
    main()
