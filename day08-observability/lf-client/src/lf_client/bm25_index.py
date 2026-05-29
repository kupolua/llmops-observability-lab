import re

from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    """Простая токенизация: lowercase + слова."""
    return re.findall(r"\w+", text.lower())


class BM25Index:
    """BM25-индекс по чанкам из Qdrant."""

    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        collection: str = "cookbook",
    ) -> None:
        self._client = QdrantClient(url=qdrant_url)
        self._collection = collection
        self._chunks: list[dict] = []  # type: ignore[type-arg]
        self._bm25: BM25Okapi | None = None

    def build(self) -> None:
        """Загрузить все чанки из Qdrant и построить BM25-индекс."""
        # Получаем ВСЕ точки из коллекции (для маленького корпуса это OK)
        # В production использовали бы scroll API с батчами
        points, _ = self._client.scroll(
            collection_name=self._collection,
            limit=10000,  # хватит для нашего корпуса
            with_payload=True,
            with_vectors=False,
        )

        self._chunks = []
        tokenized_corpus: list[list[str]] = []
        for p in points:
            if p.payload is None:
                continue
            text = str(p.payload.get("text", ""))
            self._chunks.append(
                {
                    "text": text,
                    "source": p.payload.get("source", ""),
                    "section_path": p.payload.get("section_path", ""),
                }
            )
            tokenized_corpus.append(tokenize(text))

        self._bm25 = BM25Okapi(tokenized_corpus)
        print(f"BM25 index: {len(self._chunks)} chunks")

    def search(self, query: str, top_k: int = 20) -> list[tuple[int, float]]:
        """Поиск по BM25. Возвращает [(chunk_idx, score), ...]."""
        if self._bm25 is None:
            raise RuntimeError("Call build() first")

        scores = self._bm25.get_scores(tokenize(query))
        indexed = list(enumerate(scores))
        indexed.sort(key=lambda x: x[1], reverse=True)
        return indexed[:top_k]

    def get_chunk(self, index: int) -> dict:  # type: ignore[type-arg]
        return self._chunks[index]


def main() -> None:
    index = BM25Index()
    index.build()

    queries = [
        "tool_choice force",
        "validate JSON output",
        "prompt chaining",
    ]

    for q in queries:
        print(f"\n=== '{q}' (BM25) ===")
        results = index.search(q, top_k=3)
        for rank, (idx, score) in enumerate(results, 1):
            chunk = index.get_chunk(idx)
            short_source = chunk["source"].split("/")[-1]
            print(f"  [{rank}] score={score:.2f}  {short_source}")
            print(f"      {chunk['text'][:120]}...")


if __name__ == "__main__":
    main()
