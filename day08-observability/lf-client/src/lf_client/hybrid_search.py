from dataclasses import dataclass

from dotenv import load_dotenv

from lf_client.bm25_index import BM25Index
from lf_client.embedding_service import EmbeddingService
from lf_client.reranker import Reranker
from qdrant_client import QdrantClient

load_dotenv()


@dataclass
class HybridResult:
    text: str
    source: str
    section_path: str
    rrf_score: float
    dense_rank: int | None
    bm25_rank: int | None


class HybridSearcher:
    """Гибридный поиск: dense + BM25 → RRF → опционально rerank."""

    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        collection: str = "cookbook",
        rrf_k: int = 60,
    ) -> None:
        self._qdrant = QdrantClient(url=qdrant_url)
        self._embedder = EmbeddingService()
        self._reranker = Reranker()
        self._collection = collection
        self._rrf_k = rrf_k
        self._bm25 = BM25Index(qdrant_url=qdrant_url, collection=collection)
        self._bm25.build()

    def _dense_search(self, query: str, top_n: int) -> list[dict]:  # type: ignore[type-arg]
        qvec = self._embedder.embed([query], input_type="query")[0]
        points = self._qdrant.query_points(
            collection_name=self._collection,
            query=qvec.tolist(),
            limit=top_n,
        ).points
        results = []
        for p in points:
            if p.payload is None:
                continue
            results.append(
                {
                    "text": str(p.payload.get("text", "")),
                    "source": str(p.payload.get("source", "")),
                    "section_path": str(p.payload.get("section_path", "")),
                }
            )
        return results

    def _bm25_search(self, query: str, top_n: int) -> list[dict]:  # type: ignore[type-arg]
        results = self._bm25.search(query, top_k=top_n)
        return [self._bm25.get_chunk(idx) for idx, _ in results]

    def search(
        self,
        query: str,
        top_k: int = 5,
        top_n_per_method: int = 20,
        use_rerank: bool = False,
    ) -> list[HybridResult]:
        # Получаем top_n от каждого метода
        dense_results = self._dense_search(query, top_n_per_method)
        bm25_results = self._bm25_search(query, top_n_per_method)

        # Собираем уникальные документы по тексту, считаем RRF score
        # (в реальной системе использовали бы chunk_id, но тут хватит text)
        scores: dict[str, dict] = {}  # type: ignore[type-arg]

        for rank, doc in enumerate(dense_results, 1):
            key = doc["text"]
            if key not in scores:
                scores[key] = {
                    "doc": doc,
                    "rrf": 0.0,
                    "dense_rank": None,
                    "bm25_rank": None,
                }
            scores[key]["rrf"] += 1.0 / (self._rrf_k + rank)
            scores[key]["dense_rank"] = rank

        for rank, doc in enumerate(bm25_results, 1):
            key = doc["text"]
            if key not in scores:
                scores[key] = {
                    "doc": doc,
                    "rrf": 0.0,
                    "dense_rank": None,
                    "bm25_rank": None,
                }
            scores[key]["rrf"] += 1.0 / (self._rrf_k + rank)
            scores[key]["bm25_rank"] = rank

        # Сортируем по RRF score
        sorted_docs = sorted(scores.values(), key=lambda x: x["rrf"], reverse=True)
        top_candidates = sorted_docs[: top_k if not use_rerank else 20]

        if use_rerank:
            # Финальный rerank
            documents = [c["doc"]["text"] for c in top_candidates]
            rerank_results = self._reranker.rerank(query, documents, top_k=top_k)
            final: list[HybridResult] = []
            for r in rerank_results:
                source_data = top_candidates[r.index]
                final.append(
                    HybridResult(
                        text=source_data["doc"]["text"],
                        source=source_data["doc"]["source"],
                        section_path=source_data["doc"]["section_path"],
                        rrf_score=source_data["rrf"],
                        dense_rank=source_data["dense_rank"],
                        bm25_rank=source_data["bm25_rank"],
                    )
                )
            return final

        return [
            HybridResult(
                text=c["doc"]["text"],
                source=c["doc"]["source"],
                section_path=c["doc"]["section_path"],
                rrf_score=c["rrf"],
                dense_rank=c["dense_rank"],
                bm25_rank=c["bm25_rank"],
            )
            for c in top_candidates[:top_k]
        ]


def main() -> None:
    searcher = HybridSearcher()

    queries = [
        "tool_choice force",  # должен сработать BM25 (точный термин)
        "prompt chaining",  # должен сработать dense (концепт)
        "validate JSON output",  # неясно — хороший тест на гибрид
    ]

    for q in queries:
        print(f"\n=== '{q}' (hybrid) ===")
        results = searcher.search(q, top_k=3, top_n_per_method=10)
        for rank, r in enumerate(results, 1):
            short = r.source.split("/")[-1]
            print(
                f"  [{rank}] rrf={r.rrf_score:.4f}, dense_rank={r.dense_rank}, bm25_rank={r.bm25_rank}"
            )
            print(f"      {short} / {r.section_path[:50]}")
            print(f"      {r.text[:120]}...")


if __name__ == "__main__":
    main()
