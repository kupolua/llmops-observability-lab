import asyncio
import os
import uuid

from dotenv import load_dotenv
from langfuse import get_client
from opentelemetry import trace as otel_trace

from lf_client.client import ClaudeClient

load_dotenv()


async def main() -> None:
    langfuse = get_client()
    client = ClaudeClient(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        langfuse=langfuse,
    )

    session_id = f"session-{uuid.uuid4().hex[:8]}"
    user_id = "pavlo"

    print(f"\nSession: {session_id}\nUser: {user_id}\n")

    questions = [
        "Что такое HNSW в одном предложении?",
        "А зачем нужен реранкер в RAG?",
        "Какие популярные библиотеки реранкеров?",
    ]

    for i, q in enumerate(questions, 1):
        print(f"--- Вопрос {i}: {q} ---")

        # Корневой span обёрнутый Langfuse-методом
        with langfuse.start_as_current_observation(
            as_type="span",
            name=f"qa-{i}",
            input=q,
        ) as root_span:
            # Magic-attributes для Sessions/Users через чистый OTel API
            current_span = otel_trace.get_current_span()
            current_span.set_attribute("langfuse.session.id", session_id)
            current_span.set_attribute("langfuse.user.id", user_id)
            current_span.set_attribute("langfuse.tags", '["chat", "rag-learning"]')

            result = await client.ask(q, trace_name=f"qa-{i}-llm")
            root_span.update(output=result.text[:200])

            print(f"{result.text[:200]}...\n")

    langfuse.flush()
    print(f"\nОткрой Langfuse → Sessions → найди {session_id}")
    print(f"А также → Users → найди {user_id}")


if __name__ == "__main__":
    asyncio.run(main())
