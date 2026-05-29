import os
from dataclasses import dataclass

from dotenv import load_dotenv

from voyageai import Client  # type: ignore[attr-defined]

load_dotenv()


@dataclass
class RerankResult:
    """Результат реранкинга — кандидат с новым score."""

    index: int  # индекс в исходном списке кандидатов
    score: float  # релевантность от 0 до 1
    text: str


class Reranker:
    """Обёртка над Voyage Rerank с трекингом стоимости."""

    PRICE_PER_MILLION_TOKENS = 0.05  # voyage-rerank-2, проверь актуальные цены

    def __init__(self, api_key: str | None = None, model: str = "rerank-2") -> None:
        key = api_key or os.environ["VOYAGE_API_KEY"]

        self._client = Client(api_key=key)
        self._model = model
        self._total_tokens: int = 0
        self._total_cost_usd: float = 0.0

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
    ) -> list[RerankResult]:
        """Переранжировать документы по релевантности к запросу."""
        if not documents:
            return []

        response = self._client.rerank(
            query=query,
            documents=documents,
            model=self._model,
            top_k=top_k,
        )

        # Трекинг
        tokens = response.total_tokens
        self._total_tokens += tokens
        self._total_cost_usd += tokens / 1_000_000 * self.PRICE_PER_MILLION_TOKENS

        return [
            RerankResult(
                index=r.index,
                score=r.relevance_score,
                text=r.document if hasattr(r, "document") else documents[r.index],
            )
            for r in response.results
        ]

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    @property
    def total_cost_usd(self) -> float:
        return round(self._total_cost_usd, 6)


def main() -> None:
    """Демонстрация: переранжировать несколько кандидатов."""
    reranker = Reranker()

    query = "How does prompt chaining work?"
    candidates = [
        "Prompt chaining decomposes a task into sequential subtasks, where each step builds on previous results.",
        "Build system prompt with tool catalog: list all available tools by name.",
        "Tool choice parameter controls how Claude decides to call tools.",
        "Evaluator-optimizer workflow uses two LLMs: one generates, one evaluates.",
        "Orchestrator analyzes the task and dispatches subtasks to worker LLMs.",
    ]

    print(f"Запрос: {query}\n")
    print("=== Кандидаты ДО реранкинга (в исходном порядке) ===")
    for i, c in enumerate(candidates):
        print(f"  [{i}] {c[:80]}")

    results = reranker.rerank(query, candidates, top_k=3)

    print("\n=== ПОСЛЕ реранкинга (top-3) ===")
    for r in results:
        print(f"  [orig idx={r.index}] score={r.score:.3f}")
        print(f"      {r.text[:80]}")

    print(f"\nСтоимость: ${reranker.total_cost_usd:.6f}")


if __name__ == "__main__":
    main()
