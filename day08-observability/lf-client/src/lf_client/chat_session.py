import asyncio
import os
import uuid

from dotenv import load_dotenv
from langfuse import get_client

from lf_client.client import ClaudeClient

load_dotenv()


async def main() -> None:
    langfuse = get_client()
    client = ClaudeClient(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        langfuse=langfuse,
    )

    # Уникальный ID сессии — для группировки traces в Langfuse
    session_id = f"session-{uuid.uuid4().hex[:8]}"
    user_id = "pavlo"  # представь, что это реальный пользователь

    print(f"\nSession: {session_id}\nUser: {user_id}\n")

    # Сценарий: пользователь учит RAG, задаёт связанные вопросы
    questions = [
        "Что такое RAG в одном предложении?",
        "А зачем нужен реранкер в RAG?",
        "Какие популярные библиотеки реранкеров?",
    ]

    for i, q in enumerate(questions, 1):
        print(f"--- Вопрос {i}: {q} ---")

        # Каждый вопрос — отдельный trace, но с одной session
        result = await client.ask(
            q,
            trace_name=f"qa-{i}",
            user_id=user_id,
            session_id=session_id,
            tags=["chat", "rag-learning"],
            environment="dev",
        )

        print(f"{result.text[:200]}...\n")

    langfuse.flush()
    print(f"\nОткрой Langfuse → Sessions → найди {session_id}")


if __name__ == "__main__":
    asyncio.run(main())
