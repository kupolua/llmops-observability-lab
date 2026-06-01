"""RAG-пайплайн с заземлённой (grounded) генерацией.

Полный путь: query → retrieval → confidence check → prompt → generation.

Главная идея — бороться с галлюцинациями двумя способами:
1. Confidence check ДО генерации: если retrieval вернул мусор (top-1 score
   ниже порога), мы отказываемся отвечать и НЕ зовём Claude. Это экономит
   токены и не даёт модели фантазировать на пустом контексте.
2. Системный промпт, который требует опираться только на контекст,
   признавать незнание и цитировать источники по номерам [1], [2], ...
"""

import asyncio
import os
from dataclasses import dataclass

from dotenv import load_dotenv
from langfuse import get_client

from lf_client.client import ClaudeClient
from lf_client.multi_query_retrieval import MultiQueryResult, MultiQueryRetriever
from lf_client.retrieval import RetrievedChunk, TwoStageRetriever

load_dotenv()


# Системный промпт — главный инструмент против галлюцинаций.
# Требует grounding, явный отказ при незнании и цитирование источников.
SYSTEM_PROMPT = """You are a grounded question-answering assistant for the \
Anthropic Cookbook. You answer questions strictly from the context provided \
to you.

Rules you MUST follow:
1. Use ONLY the information in the provided context. Never rely on outside \
knowledge or assumptions.
2. If the context does not contain enough information to answer the question, \
do NOT guess. Reply with exactly this sentence and nothing else:
   "I don't have enough information in the provided context to answer this."
3. Cite the sources you used with their bracket numbers inline, e.g. [1], [2], \
right after the claims they support.
4. Be concise and factual. Do not add information that is not grounded in the \
context."""

# Если модель не нашла ответа в контексте, она вернёт ровно эту фразу.
# Используем её, чтобы пометить ответ как отказ (is_refusal=True).
REFUSAL_SENTENCE = (
    "I don't have enough information in the provided context to answer this."
)


@dataclass
class GroundedAnswer:
    """Результат RAG-генерации."""

    answer: str  # текст ответа
    is_refusal: bool  # True если система отказалась отвечать
    refusal_reason: str | None  # почему отказалась (None если ответила)
    chunks_used: list[RetrievedChunk]  # чанки, которые пошли в контекст
    top_confidence: float  # score top-1 чанка (для отладки)
    # Метаданные retrieval'а, если использовался MultiQueryRetriever
    # (subqueries, dedup-счётчики и т.д.). None для обычного retriever'а.
    retrieval_meta: MultiQueryResult | None = None


def _confidence(chunk: RetrievedChunk) -> float:
    """Уверенность по чанку: rerank_score важнее, иначе dense_score.

    После реранкинга rerank_score — это нормализованный (0..1) score
    кросс-энкодера, самый осмысленный сигнал. Если реранкинга не было,
    падаем на dense_score из Qdrant.
    """
    return chunk.rerank_score if chunk.rerank_score is not None else chunk.dense_score


def _format_context(chunks: list[RetrievedChunk]) -> str:
    """Собрать пронумерованный контекст для цитирования.

    Формат каждого чанка:
        [1] (source / section_path):
        текст чанка...
    Номера [1], [2] нужны, чтобы Claude мог ссылаться на источники в ответе.
    """
    blocks: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        header = f"[{i}] ({chunk.source} / {chunk.section_path}):"
        blocks.append(f"{header}\n{chunk.text}")
    return "\n\n".join(blocks)


class GroundedGenerator:
    """RAG-пайплайн с заземлённой генерацией.

    1. Retrieval через TwoStageRetriever ИЛИ MultiQueryRetriever
    2. Confidence check — если top-1 < threshold → refuse (без вызова LLM)
    3. Сборка промпта с пронумерованным контекстом для цитирования
    4. Generation через ClaudeClient (observability + retry «из коробки»)

    Retriever полиморфен, но интерфейсы у двух реализаций разные:
    - TwoStageRetriever.retrieve — синхронный, принимает top_k, отдаёт
      list[RetrievedChunk];
    - MultiQueryRetriever.retrieve — асинхронный, top_k не принимает (свой
      final_top_k), отдаёт MultiQueryResult c полем .chunks.
    Различия прячем в _retrieve_chunks, наружу — единый list[RetrievedChunk].
    """

    def __init__(
        self,
        retriever: TwoStageRetriever | MultiQueryRetriever,
        client: ClaudeClient,
        confidence_threshold: float = 0.4,
        top_k: int = 3,
    ) -> None:
        self._retriever = retriever
        self._client = client
        self._confidence_threshold = confidence_threshold
        self._top_k = top_k

    async def _retrieve_chunks(
        self, query: str
    ) -> tuple[list[RetrievedChunk], MultiQueryResult | None]:
        """Нормализует разные retriever'ы к общему list[RetrievedChunk].

        Вторым элементом возвращает метаданные multi-query retrieval'а
        (subqueries, dedup-счётчики) — или None для обычного retriever'а.
        """
        if isinstance(self._retriever, MultiQueryRetriever):
            # Уже асинхронный; top_k задаётся через его собственный
            # final_top_k, поэтому наш self._top_k тут не применяется.
            result = await self._retriever.retrieve(query)
            return result.chunks, result

        # TwoStageRetriever.retrieve синхронный и ходит по сети
        # (Qdrant + reranker), поэтому уводим его в поток, чтобы не
        # блокировать event loop.
        chunks = await asyncio.to_thread(
            self._retriever.retrieve, query, top_k=self._top_k
        )
        return chunks, None

    async def answer(self, query: str) -> GroundedAnswer:
        """Полный pipeline: query → retrieved chunks → answer."""
        # Stage 1: retrieval (см. _retrieve_chunks — прячет разницу
        # между TwoStageRetriever и MultiQueryRetriever).
        chunks, retrieval_meta = await self._retrieve_chunks(query)

        # Stage 2a: пустой retrieval — отказываемся сразу.
        if not chunks:
            return GroundedAnswer(
                answer=REFUSAL_SENTENCE,
                is_refusal=True,
                refusal_reason="no_chunks_retrieved",
                chunks_used=[],
                top_confidence=0.0,
                retrieval_meta=retrieval_meta,
            )

        # Stage 2b: confidence check ДО генерации.
        top_score = _confidence(chunks[0])
        if top_score < self._confidence_threshold:
            return GroundedAnswer(
                answer=REFUSAL_SENTENCE,
                is_refusal=True,
                refusal_reason=(
                    f"low_confidence (top={top_score:.3f} < "
                    f"threshold={self._confidence_threshold})"
                ),
                chunks_used=[],
                top_confidence=top_score,
                retrieval_meta=retrieval_meta,
            )

        # Stage 3: сборка промпта с цитированием.
        context = _format_context(chunks)
        user_message = (
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer using only the context above and cite sources by number."
        )

        # Stage 4: generation через ClaudeClient.
        # ClaudeClient.ask пока не принимает произвольную metadata, поэтому
        # ключевые диагностические поля прокидываем через tags — они видны
        # в Langfuse и позволяют фильтровать трейсы.
        result = await self._client.ask(
            prompt=user_message,
            system=SYSTEM_PROMPT,
            trace_name="rag-grounded-answer",
            tags=[
                "rag",
                "production",
                f"num_chunks={len(chunks)}",
                f"top_confidence={top_score:.3f}",
            ],
        )

        # Модель тоже может «отказаться», если контекст не содержит ответа.
        answer_text = result.text.strip()
        model_refused = REFUSAL_SENTENCE.lower() in answer_text.lower()

        return GroundedAnswer(
            answer=answer_text,
            is_refusal=model_refused,
            refusal_reason="model_refused_no_answer_in_context"
            if model_refused
            else None,
            chunks_used=chunks,
            top_confidence=top_score,
            retrieval_meta=retrieval_meta,
        )


async def main() -> None:
    langfuse = get_client()
    retriever = TwoStageRetriever()
    client = ClaudeClient(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        langfuse=langfuse,
    )
    generator = GroundedGenerator(retriever, client)

    queries = [
        "When should I use parallel tool calls?",  # отвечаемый
        "What is the airspeed velocity of an unladen swallow?",  # вне корпуса
    ]

    for query in queries:
        print(f"\n{'=' * 70}")
        print(f"Q: {query}")
        print(f"{'-' * 70}")

        answer = await generator.answer(query)

        if answer.is_refusal:
            print(f"[REFUSAL] reason: {answer.refusal_reason}")
            print(f"  top_confidence: {answer.top_confidence:.3f}")
        else:
            print(answer.answer)
            print(f"\n  top_confidence: {answer.top_confidence:.3f}")
            print(f"  sources ({len(answer.chunks_used)}):")
            for i, chunk in enumerate(answer.chunks_used, start=1):
                print(f"    [{i}] {chunk.source} / {chunk.section_path}")

    print(f"\n{'=' * 70}")
    print(f"LLM cost: ${client.total_cost_usd:.6f} ({client.call_count} calls)")
    print(f"Retrieval cost: ${retriever.total_cost_usd:.6f}")
    langfuse.flush()


if __name__ == "__main__":
    asyncio.run(main())
