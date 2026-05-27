from dotenv import load_dotenv

from emb_lab.service import EmbeddingService
from emb_lab.similarity import find_nearest

load_dotenv()


CORPUS = [
    "Python — это язык программирования с динамической типизацией.",
    "TypeScript добавляет статическую типизацию к JavaScript.",
    "Borscht — традиционный украинский суп со свёклой.",
    "Salo — традиционная украинская закуска из свиного жира.",
    "Docker позволяет упаковывать приложения в контейнеры.",
    "Kubernetes оркестрирует контейнеры в распределённых системах.",
    "Эмбеддинги превращают текст в вектора чисел.",
    "HNSW — алгоритм приближённого поиска ближайших соседей.",
]


def main() -> None:
    service = EmbeddingService()

    print("Эмбеддим корпус...")
    corpus_vectors = service.embed(CORPUS, input_type="document")
    print(f"  готово: {len(CORPUS)} документов, размерность {corpus_vectors.shape[1]}")

    queries = [
        "какие есть языки программирования?",
        "что готовят на Украине?",
        "как развернуть приложение?",
        "как работает векторный поиск?",
    ]

    print("Эмбеддим запросы...")
    query_vectors = service.embed(queries, input_type="query")

    for query, query_vector in zip(queries, query_vectors, strict=True):
        results = find_nearest(query_vector, corpus_vectors, top_k=3)
        print(f"\n=== Запрос: {query} ===")
        for rank, (idx, score) in enumerate(results, 1):
            print(f"  {rank}. [{score:.3f}] {CORPUS[idx]}")

    print("\n--- Итого ---")
    print(f"Токенов потрачено: {service.total_tokens}")
    print(f"Стоимость: ${service.total_cost_usd:.6f}")


if __name__ == "__main__":
    main()
