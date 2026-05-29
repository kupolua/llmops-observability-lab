import uuid

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from lf_client.embedding_service import EmbeddingService

load_dotenv()


# Тестовый "корпус" — пока не строительные нормы, просто проверка pipeline
TEST_CORPUS = [
    "Штукатурные работы начинаются с подготовки основания и установки маяков.",
    "Грунтовка глубокого проникновения укрепляет основание и снижает впитываемость.",
    "Гипсокартонные листы крепятся к металлическому каркасу саморезами.",
    "Стыки гипсокартона армируются серпянкой и шпаклюются в два слоя.",
    "Наливные полы требуют предварительной гидроизоляции основания.",
    "Финишная шпаклёвка наносится тонким слоем под покраску или обои.",
    "Деформационные швы предотвращают трещинообразование в стяжке.",
    "Малярные работы выполняются при температуре не ниже +5 градусов.",
]


def main() -> None:
    embedder = EmbeddingService()
    client = QdrantClient(url="http://localhost:6333")

    collection_name = "construction_test"

    # Пересоздаём коллекцию
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )

    # 1. Эмбеддим корпус (батчем — помнишь день 6)
    print("Эмбеддим корпус...")
    vectors = embedder.embed(TEST_CORPUS, input_type="document")
    print(f"  получили {len(vectors)} векторов размерности {vectors.shape[1]}")

    # 2. Формируем points для Qdrant
    points = []
    for i, (text, vector) in enumerate(zip(TEST_CORPUS, vectors, strict=True)):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector.tolist(),
                payload={
                    "text": text,
                    "doc_type": "test",
                    "chunk_index": i,
                },
            )
        )

    # 3. Загружаем в Qdrant
    client.upsert(collection_name=collection_name, points=points)
    print(f"  загружено {len(points)} точек в '{collection_name}'")

    # Проверяем
    info = client.get_collection(collection_name)
    print(f"\nВ коллекции теперь: {info.points_count} векторов")
    print(f"Стоимость эмбеддинга: ${embedder.total_cost_usd:.6f}")
    print("\nОткрой http://localhost:6333/dashboard — увидишь точки в коллекции")


if __name__ == "__main__":
    main()
