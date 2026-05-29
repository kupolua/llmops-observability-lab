import uuid
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from lf_client.embedding_service import EmbeddingService
from lf_client.ipynb_parse import parse_notebook
from lf_client.md_chunking import MdChunk, chunk_markdown
from lf_client.md_parse import read_markdown

load_dotenv()


# Наш отобранный корпус
CORPUS_FILES = [
    # patterns/agents
    "patterns/agents/README.md",
    "patterns/agents/basic_workflows.ipynb",
    "patterns/agents/evaluator_optimizer.ipynb",
    "patterns/agents/orchestrator_workers.ipynb",
    # tool_use
    "tool_use/calculator_tool.ipynb",
    "tool_use/customer_service_agent.ipynb",
    "tool_use/extracting_structured_json.ipynb",
    "tool_use/parallel_tools.ipynb",
    "tool_use/tool_choice.ipynb",
    "tool_use/tool_use_with_pydantic.ipynb",
    "tool_use/tool_search_alternate_approaches.ipynb",
    # observability
    "observability/usage_cost_api.ipynb",
    # корневые
    "README.md",
    "CONTRIBUTING.md",
    "CLAUDE.md",
]


def load_file_as_text(path: Path) -> str:
    """Универсальный загрузчик — выбирает парсер по расширению."""
    if path.suffix == ".ipynb":
        return parse_notebook(path)
    if path.suffix == ".md":
        return read_markdown(path)
    raise ValueError(f"Неподдерживаемый формат: {path.suffix}")


def main() -> None:
    base = Path("data/anthropic-cookbook")

    # 1. Собираем чанки со всех файлов
    all_chunks: list[MdChunk] = []
    for rel_path in CORPUS_FILES:
        path = base / rel_path
        if not path.exists():
            print(f"SKIP (не найден): {rel_path}")
            continue

        try:
            text = load_file_as_text(path)
        except Exception as e:
            print(f"FAIL parsing {rel_path}: {e}")
            continue

        chunks = chunk_markdown(text, source=rel_path)
        all_chunks.extend(chunks)
        print(f"  {rel_path}: {len(chunks)} чанков")

    print(f"\nВсего чанков: {len(all_chunks)}")

    # 2. Эмбеддим (батчем — Voyage позволяет до 128 за раз)
    embedder = EmbeddingService()
    texts = [c.text for c in all_chunks]

    print(f"\nЭмбеддим {len(texts)} чанков...")
    # Батчим по 100 для безопасности
    all_vectors: list[NDArray[np.float32]] = []
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        vectors = embedder.embed(batch, input_type="document")
        all_vectors.extend(vectors)
        print(f"  batch {i // batch_size + 1}: {len(batch)} чанков")

    print(f"\nСтоимость эмбеддинга: ${embedder.total_cost_usd:.6f}")

    # 3. Индексируем в Qdrant
    client = QdrantClient(url="http://localhost:6333")
    collection_name = "cookbook"

    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vec.tolist(),
            payload={
                "text": chunk.text,
                "source": chunk.source,
                "section_path": chunk.section_path,
                "chunk_index": chunk.chunk_index,
                "token_count": chunk.token_count,
                # Категория по верхнему уровню папки
                "category": chunk.source.split("/")[0]
                if "/" in chunk.source
                else "root",
            },
        )
        for chunk, vec in zip(all_chunks, all_vectors, strict=True)
    ]

    # Заливаем батчами тоже (Qdrant позволяет, но большие upsert'ы могут таймаутить)
    for i in range(0, len(points), batch_size):
        client.upsert(
            collection_name=collection_name, points=points[i : i + batch_size]
        )

    info = client.get_collection(collection_name)
    print(f"\nКоллекция 'cookbook': {info.points_count} точек")
    print("Открой http://localhost:6333/dashboard")


if __name__ == "__main__":
    main()
