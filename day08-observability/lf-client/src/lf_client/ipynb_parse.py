from pathlib import Path

import nbformat


def parse_notebook(path: Path) -> str:
    """Прочитать .ipynb и вернуть его как plain markdown-текст.

    Markdown-ячейки идут как есть.
    Code-ячейки оборачиваются в ```python ... ``` блоки.
    Outputs игнорируются (там часто мусор: base64-картинки, dataframes).
    """
    notebook = nbformat.read(str(path), as_version=4)  # type: ignore[no-untyped-call]

    parts: list[str] = []
    for cell in notebook.cells:
        cell_type = cell.get("cell_type")
        source = cell.get("source", "")

        # source может быть либо строкой, либо списком строк
        if isinstance(source, list):
            source_text = "".join(source)
        else:
            source_text = source

        if not source_text.strip():
            continue

        if cell_type == "markdown":
            parts.append(source_text)
        elif cell_type == "code":
            parts.append(f"```python\n{source_text}\n```")
        # raw и прочее игнорируем

    return "\n\n".join(parts)


def main() -> None:
    base = Path("data/anthropic-cookbook")

    # Один файл для проверки
    sample_path = base / "patterns/agents/basic_workflows.ipynb"
    if not sample_path.exists():
        print(f"ERROR: {sample_path} не найден")
        return

    text = parse_notebook(sample_path)
    print(f"Распарсили: {sample_path.relative_to(base)}")
    print(f"  размер: {len(text)} символов")
    print("\n=== Первые 1500 символов ===\n")
    print(text[:1500])
    print("\n...")


if __name__ == "__main__":
    main()
