import numpy as np
from numpy.typing import NDArray


def cosine_similarity(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    """Косинусное сходство двух векторов.

    1.0 = идентичные направления
    0.0 = перпендикулярные (никакой связи)
    -1.0 = противоположные
    """
    dot = float(np.dot(a, b))
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    return dot / (norm_a * norm_b)


def find_nearest(
    query_vector: NDArray[np.float32],
    document_vectors: NDArray[np.float32],
    top_k: int = 3,
) -> list[tuple[int, float]]:
    """Найти top_k ближайших документов к запросу.

    Возвращает список [(индекс_документа, сходство), ...]
    отсортированный по убыванию сходства.
    """
    # Вычисляем сходство со всеми документами сразу (векторизованно)
    similarities = [
        cosine_similarity(query_vector, doc_vec) for doc_vec in document_vectors
    ]

    # Получаем индексы топ-K, отсортированные по сходству
    indexed = list(enumerate(similarities))
    indexed.sort(key=lambda x: x[1], reverse=True)
    return indexed[:top_k]
