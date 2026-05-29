import os
from typing import Literal

import numpy as np
from voyageai import Client  # type: ignore[attr-defined]
from numpy.typing import NDArray
from pydantic import BaseModel

VOYAGE_PRICES = {
    "voyage-3.5": 0.06,
    "voyage-3.5-lite": 0.02,
    "voyage-3-large": 0.18,
}


class EmbeddingCost(BaseModel):
    total_tokens: int
    cost_usd: float


class EmbeddingService:
    """Обёртка над Voyage AI с трекингом стоимости."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "voyage-3.5",
    ) -> None:
        key = api_key or os.environ["VOYAGE_API_KEY"]
        self._client = Client(api_key=key)
        self._model = model
        self._total_tokens: int = 0
        self._total_cost_usd: float = 0.0

    def embed(
        self,
        texts: list[str],
        input_type: Literal["document", "query"],
    ) -> NDArray[np.float32]:
        """Получить эмбеддинги для списка текстов."""
        response = self._client.embed(
            texts,
            model=self._model,
            input_type=input_type,
        )

        # Трекинг
        tokens = response.total_tokens
        self._total_tokens += tokens
        price_per_million = VOYAGE_PRICES.get(self._model, 0.0)
        self._total_cost_usd += tokens / 1_000_000 * price_per_million

        return np.array(response.embeddings, dtype=np.float32)

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    @property
    def total_cost_usd(self) -> float:
        return round(self._total_cost_usd, 6)
