import numpy as np
import pytest
from numpy.typing import NDArray

from emb_lab.similarity import cosine_similarity, find_nearest


def vec(data: list[float]) -> NDArray[np.float32]:
    """Helper: create a typed float32 vector from a Python list."""
    return np.array(data, dtype=np.float32)


# ---------- cosine_similarity ----------


def test_cosine_identical_vectors() -> None:
    v = vec([1.0, 2.0, 3.0])
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_orthogonal_vectors() -> None:
    a = vec([1.0, 0.0])
    b = vec([0.0, 1.0])
    assert cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosine_opposite_vectors() -> None:
    a = vec([1.0, 0.0])
    b = vec([-1.0, 0.0])
    assert cosine_similarity(a, b) == pytest.approx(-1.0)


def test_cosine_partial_similarity() -> None:
    """Угол 60° между векторами даёт косинус 0.5."""
    a = vec([1.0, 0.0])
    b = vec([0.5, 0.866])  # ≈ cos(60°), sin(60°)
    assert cosine_similarity(a, b) == pytest.approx(0.5, abs=0.01)


# ---------- find_nearest ----------


def test_find_nearest_returns_top_k() -> None:
    query = vec([1.0, 0.0, 0.0])
    docs = np.array(
        [
            [1.0, 0.0, 0.0],  # 0: идентичен запросу
            [0.0, 1.0, 0.0],  # 1: перпендикулярен
            [0.9, 0.1, 0.0],  # 2: очень похож
            [-1.0, 0.0, 0.0],  # 3: противоположен
        ],
        dtype=np.float32,
    )

    results = find_nearest(query, docs, top_k=2)
    assert len(results) == 2

    # Первый результат — идентичный документ (индекс 0)
    assert results[0][0] == 0
    # Второй — очень похожий (индекс 2)
    assert results[1][0] == 2


def test_find_nearest_sorted_descending() -> None:
    """Результаты должны быть отсортированы по убыванию сходства."""
    query = vec([1.0, 0.0])
    docs = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.5, 0.5],
        ],
        dtype=np.float32,
    )

    results = find_nearest(query, docs, top_k=3)
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)


def test_find_nearest_respects_top_k() -> None:
    """top_k=1 возвращает один результат, даже если документов много."""
    query = vec([1.0, 0.0])
    docs = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.5, 0.5],
        ],
        dtype=np.float32,
    )

    results = find_nearest(query, docs, top_k=1)
    assert len(results) == 1
