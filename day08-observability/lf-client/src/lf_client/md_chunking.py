import re
from dataclasses import dataclass

import tiktoken


@dataclass
class MdChunk:
    """Чанк из markdown-документа с метаданными."""

    text: str
    chunk_index: int
    source: str
    section_path: str  # например: "Workflows > Prompt Chaining > Examples"
    token_count: int


_ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


def split_by_headers(
    text: str,
    max_tokens: int = 500,
) -> list[tuple[str, str]]:
    """Разбить markdown по заголовкам H1/H2/H3.

    Возвращает список (section_path, content), где section_path — иерархия
    заголовков, под которой находится этот контент.

    Если секция больше max_tokens, она будет дальше резаться recursive splitter'ом
    в вызывающем коде.
    """
    lines = text.split("\n")

    sections: list[tuple[str, str]] = []
    current_h1 = ""
    current_h2 = ""
    current_h3 = ""
    buffer: list[str] = []

    def flush() -> None:
        if not buffer:
            return
        content = "\n".join(buffer).strip()
        if not content:
            return
        # Формируем path
        parts = [p for p in (current_h1, current_h2, current_h3) if p]
        section_path = " > ".join(parts) if parts else "(root)"
        sections.append((section_path, content))

    for line in lines:
        # Проверяем, не заголовок ли это
        h1_match = re.match(r"^# (.+)$", line)
        h2_match = re.match(r"^## (.+)$", line)
        h3_match = re.match(r"^### (.+)$", line)

        if h1_match:
            flush()
            buffer = []
            current_h1 = h1_match.group(1).strip()
            current_h2 = ""
            current_h3 = ""
        elif h2_match:
            flush()
            buffer = []
            current_h2 = h2_match.group(1).strip()
            current_h3 = ""
        elif h3_match:
            flush()
            buffer = []
            current_h3 = h3_match.group(1).strip()
        else:
            buffer.append(line)

    flush()
    return sections


def recursive_split_by_tokens(
    text: str,
    max_tokens: int = 500,
    overlap_tokens: int = 50,
) -> list[str]:
    """Если секция получилась слишком большой — режем рекурсивно.

    Упрощённая версия из дня 16. Режем по двойным переносам, потом одиночным.
    """
    if count_tokens(text) <= max_tokens:
        return [text]

    # Разделители в порядке предпочтения
    for sep in ["\n\n", "\n", ". ", " "]:
        if sep in text:
            parts = text.split(sep)
            break
    else:
        # Совсем нечем резать — режем по токенам грубо
        tokens = _ENCODING.encode(text)
        return [
            _ENCODING.decode(tokens[i : i + max_tokens])
            for i in range(0, len(tokens), max_tokens)
        ]

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for part in parts:
        part_tokens = count_tokens(part)
        if current_tokens + part_tokens <= max_tokens:
            current.append(part)
            current_tokens += part_tokens
        else:
            if current:
                chunks.append(sep.join(current))
            current = [part]
            current_tokens = part_tokens

    if current:
        chunks.append(sep.join(current))

    return chunks


def chunk_markdown(
    text: str,
    source: str,
    max_tokens: int = 500,
) -> list[MdChunk]:
    """Полный pipeline: markdown → чанки с метаданными.

    1. Режем по заголовкам (документ-aware)
    2. Большие секции дополнительно режем рекурсивно
    """
    sections = split_by_headers(text, max_tokens=max_tokens)

    all_chunks: list[MdChunk] = []
    for section_path, content in sections:
        section_tokens = count_tokens(content)

        if section_tokens <= max_tokens:
            # Целиком — один чанк
            all_chunks.append(
                MdChunk(
                    text=content,
                    chunk_index=len(all_chunks),
                    source=source,
                    section_path=section_path,
                    token_count=section_tokens,
                )
            )
        else:
            # Большая секция — режем дальше
            sub_chunks = recursive_split_by_tokens(content, max_tokens=max_tokens)
            for sub in sub_chunks:
                all_chunks.append(
                    MdChunk(
                        text=sub,
                        chunk_index=len(all_chunks),
                        source=source,
                        section_path=section_path,
                        token_count=count_tokens(sub),
                    )
                )

    return all_chunks


def main() -> None:
    from pathlib import Path
    from lf_client.ipynb_parse import parse_notebook

    sample = Path("data/anthropic-cookbook/patterns/agents/basic_workflows.ipynb")
    text = parse_notebook(sample)

    chunks = chunk_markdown(text, source=str(sample))

    print(f"Получено чанков: {len(chunks)}\n")

    # Статистика
    sizes = [c.token_count for c in chunks]
    print("Размеры чанков (токены):")
    print(f"  min={min(sizes)}, max={max(sizes)}, avg={sum(sizes) // len(sizes)}")

    # Покажем 2-3 примера
    print(f"\n=== Чанк 0 ({chunks[0].token_count} tok) ===")
    print(f"section: {chunks[0].section_path}")
    print(chunks[0].text[:300])
    print("...")

    mid = len(chunks) // 2
    print(f"\n=== Чанк {mid} ({chunks[mid].token_count} tok) ===")
    print(f"section: {chunks[mid].section_path}")
    print(chunks[mid].text[:300])
    print("...")


if __name__ == "__main__":
    main()
