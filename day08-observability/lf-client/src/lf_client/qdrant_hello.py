from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


def main() -> None:
    client = QdrantClient(url="http://localhost:6333")

    collection_name = "test_collection"

    # Удалим, если уже существует (для идемпотентности скрипта)
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)

    # Создаём коллекцию
    # size=1024 — размерность векторов Voyage voyage-3.5 (помнишь из дня 6)
    # Distance.COSINE — косинусное сходство
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )

    print(f"Коллекция '{collection_name}' создана")

    # Информация о коллекции
    info = client.get_collection(collection_name)
    print(f"  статус: {info.status}")
    print(f"  векторов: {info.points_count}")
    print(f"  размерность: {info.config.params.vectors.size}")  # type: ignore[union-attr]

    print("\nОткрой http://localhost:6333/dashboard — увидишь коллекцию")


if __name__ == "__main__":
    main()
