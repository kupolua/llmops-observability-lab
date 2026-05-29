from dataclasses import dataclass

from dotenv import load_dotenv
from qdrant_client import QdrantClient

from lf_client.embedding_service import EmbeddingService
from lf_client.reranker import Reranker

load_dotenv()


@dataclass
class RetrievedChunk:
    """Чанк, полученный из retrieval с полной информацией."""

    text: str
    source: str
    section_path: str
    dense_score: float
    rerank_score: float | None  # None если не было реранкинга


class TwoStageRetriever:
    """Two-stage retrieval: dense → rerank.

    Stage 1: Qdrant находит top_n_candidates кандидатов (быстро, неточно).
    Stage 2: Reranker переоценивает их и возвращает top_k (медленно, точно).
    """

    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        collection: str = "cookbook",
    ) -> None:
        self._client = QdrantClient(url=qdrant_url)
        self._embedder = EmbeddingService()
        self._reranker = Reranker()
        self._collection = collection

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        top_n_candidates: int = 20,
        use_rerank: bool = True,
    ) -> list[RetrievedChunk]:
        """Получить top_k чанков для запроса."""
        # Stage 1: dense retrieval
        qvec = self._embedder.embed([query], input_type="query")[0]
        points = self._client.query_points(
            collection_name=self._collection,
            query=qvec.tolist(),
            limit=top_n_candidates if use_rerank else top_k,
        ).points

        candidates: list[RetrievedChunk] = []
        for p in points:
            if p.payload is None:
                continue
            candidates.append(
                RetrievedChunk(
                    text=str(p.payload.get("text", "")),
                    source=str(p.payload.get("source", "")),
                    section_path=str(p.payload.get("section_path", "")),
                    dense_score=p.score,
                    rerank_score=None,
                )
            )

        if not use_rerank or not candidates:
            return candidates[:top_k]

        # Stage 2: rerank
        documents = [c.text for c in candidates]
        rerank_results = self._reranker.rerank(query, documents, top_k=top_k)

        # Восстанавливаем чанки в новом порядке + score реранкера
        reranked: list[RetrievedChunk] = []
        for r in rerank_results:
            candidate = candidates[r.index]
            reranked.append(
                RetrievedChunk(
                    text=candidate.text,
                    source=candidate.source,
                    section_path=candidate.section_path,
                    dense_score=candidate.dense_score,
                    rerank_score=r.score,
                )
            )
        return reranked

    @property
    def total_cost_usd(self) -> float:
        return round(self._embedder.total_cost_usd + self._reranker.total_cost_usd, 6)


def main() -> None:
    retriever = TwoStageRetriever()
    query = "When should I use parallel tool calls?"

    print(f"Запрос: {query}\n")
    results = retriever.retrieve(query, top_k=3, top_n_candidates=20)

    for rank, chunk in enumerate(results, 1):
        print(
            f"[{rank}] rerank={chunk.rerank_score:.3f}, dense={chunk.dense_score:.3f}"
        )
        print(f"    source: {chunk.source}")
        print(f"    section: {chunk.section_path}")
        print(f"    text: {chunk.text[:150]}...\n")

    print(f"\nИтого стоимость: ${retriever.total_cost_usd:.6f}")


if __name__ == "__main__":
    main()
