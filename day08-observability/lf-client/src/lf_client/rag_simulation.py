import asyncio
import os
import time

from dotenv import load_dotenv
from langfuse import get_client

from lf_client.client import ClaudeClient

load_dotenv()


def fake_embed(text: str) -> list[float]:
    """Симуляция: эмбеддинг текста (на самом деле — sleep)."""
    time.sleep(0.05)
    return [0.1, 0.2, 0.3]


def fake_search(query_vec: list[float]) -> list[str]:
    """Симуляция: поиск в vector DB."""
    time.sleep(0.08)
    return [
        "RAG расшифровывается как Retrieval-Augmented Generation.",
        "RAG используется для добавления внешних знаний в LLM.",
        "Типичный RAG: эмбеддинг → поиск → инъекция в промпт → LLM.",
    ]


def fake_rerank(query: str, docs: list[str]) -> list[str]:
    """Симуляция: реранкинг найденных документов."""
    time.sleep(0.15)
    return docs  # для простоты возвращаем как есть


async def answer_question(client: ClaudeClient, question: str) -> str:
    langfuse = get_client()

    # Корневой trace для всей задачи
    with langfuse.start_as_current_observation(
        as_type="span",
        name="answer-user-question",
        input=question,
    ) as root:
        langfuse.update_current_span(
            metadata={
                "environment": "dev",
                "tags": ["rag-pipeline"],
            },
        )

        # Span 1: эмбеддинг
        with langfuse.start_as_current_observation(name="embed-query") as embed_span:
            query_vec = fake_embed(question)
            embed_span.update(output={"dim": len(query_vec)})

        # Span 2: поиск
        with langfuse.start_as_current_observation(name="vector-search") as search_span:
            docs = fake_search(query_vec)
            search_span.update(output={"hits": len(docs)})

        # Span 3: реранкер
        with langfuse.start_as_current_observation(name="rerank") as rerank_span:
            reranked = fake_rerank(question, docs)
            rerank_span.update(output={"top_k": len(reranked)})

        # Generation: вызов LLM с контекстом
        context = "\n".join(f"- {d}" for d in reranked)
        prompt = f"Используя только эти источники:\n{context}\n\nОтветь на вопрос: {question}"

        result = await client.ask(
            prompt,
            trace_name="rag-llm-call",
        )

        root.update(output=result.text)
        return result.text


async def main() -> None:
    langfuse = get_client()
    client = ClaudeClient(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        langfuse=langfuse,
    )

    answer = await answer_question(client, "Что такое RAG?")
    print(f"\n=== Ответ ===\n{answer}\n")

    langfuse.flush()
    print("Открой Langfuse → Traces → найди 'answer-user-question'")


if __name__ == "__main__":
    asyncio.run(main())
